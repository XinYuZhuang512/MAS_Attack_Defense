"""Extract action frames and bind entity arguments using only the observed trajectory prefix."""
from collections import defaultdict
from dataclasses import dataclass

import stanza
from stanza.models.common.doc import Document

from .boundaries import repaired_sentences
from .step_fragments import explicit_fragment
from .syntax_details import add_infinitive_modifiers, enrich_query


@dataclass(frozen=True)
class Mention:
    text: str
    lemma: str
    node: str | None
    word: int
    evidence: str


MODELS = {'tokenize': 'combined', 'mwt': 'combined', 'pos': 'combined_charlm',
          'lemma': 'combined_nocharlm', 'depparse': 'combined_charlm',
          'constituency': 'ptb3-revised_charlm'}
CONTROL = {'need', 'plan', 'intend', 'want', 'decide', 'aim'}
REPORT = {'say', 'suggest', 'ask', 'instruct', 'request', 'tell', 'claim', 'require'}
QUERY = {'check', 'verify', 'see', 'determine', 'confirm', 'investigate'}
CLAUSAL = {'advcl', 'acl', 'ccomp', 'xcomp', 'parataxis'}
MODES = {'will': 'stated_plan', 'shall': 'stated_plan', 'should': 'proposed',
         'must': 'required', 'can': 'possible', 'could': 'possible', 'may': 'possible',
         'might': 'possible', 'would': 'possible'}
DISCOURSE = {'wait', 'well', 'okay'}
PURPOSE_MARKERS = {'because', 'since', 'although', 'whereas', 'while', 'after',
                   'before', 'lest', 'until', 'unless', 'if', 'when', 'as'}


def constituent_spans(tree, words):
    """Return half-open, 1-based token intervals. Both parses must use the same words."""
    spans, leaves = [], []
    recipient_pp = []
    def visit(node, parent=None):
        start = len(leaves) + 1
        if not node.children:
            leaves.append(node.label)
        else:
            for child in node.children:
                visit(child, node)
            if node.label == 'PP' and parent is not None and parent.label == 'VP':
                recipient_pp.append((start, len(leaves) + 1))
            if node.label in {'S', 'SBAR'}:
                # A matrix S can begin with a fronted SBAR. Its main predicate
                # lies outside that SBAR and must survive.
                infinitival = node.label == 'S' and node.children[0].label == 'VP'
                spans.append((start, len(leaves) + 1, node.label, infinitival))
    visit(tree)
    if len(leaves) != len(words):
        raise ValueError('Constituency and dependency token counts differ')
    return spans, recipient_pp


class Sentence:
    def __init__(self, sent, thought, joint, refine=False):
        self.words = {w.id: w for w in sent.words}
        self.children = defaultdict(list)
        for w in sent.words:
            self.children[w.head].append(w)
        self.refine = refine
        self.spans, self.vp_pp = constituent_spans(sent.constituency, sent.words) if joint else ([], [])
        self.offsets = {w.id: (w.start_char if w.start_char is not None else t.start_char,
                               w.end_char if w.end_char is not None else t.end_char)
                        for t in sent.tokens for w in t.words}
        self.text = thought

    def excerpt(self, ids):
        # Preserve exact original text, including filenames split into multiple tokens.
        return self.text[self.offsets[min(ids)][0]:self.offsets[max(ids)][1]]

    def argument_text(self, ids):
        # Do not reintroduce omitted clauses between discontinuous token runs.
        runs = []
        for i in sorted(ids):
            if not runs or runs[-1][-1] + 1 != i:
                runs.append([])
            runs[-1].append(i)
        return ' … '.join(self.excerpt(run) for run in runs)

    def prepositional_argument(self, marker, ids, prefix=''):
        # UD attaches the other words of a compound preposition to its case marker.
        markers = {marker.id}
        markers.update(c.id for c in self.children[marker.id] if c.deprel == 'fixed')
        prep = ' '.join(self.words[i].text.lower() for i in sorted(markers))
        return {'role': prefix + prep, 'text': self.argument_text(set(ids) - markers),
                'surface': self.argument_text(ids)}

    def subtree(self, word, excluded=()):
        ids = [word.id]
        for c in self.children[word.id]:
            if c.deprel.split(':')[0] not in excluded:
                ids.extend(self.subtree(c, excluded))
        return ids

    def in_subordinate_clause(self, w, controller=None):
        for a,b,label,infinitival in self.spans:
            if not (a <= w.id < b) or label not in {'S','SBAR'}:
                continue
            if controller is not None and a <= controller.id < b:
                continue
            first = self.words[a].text.lower()
            if label == 'SBAR' and (first in PURPOSE_MARKERS or
                    (first == 'so' and a+1 < b and self.words[a+1].text.lower() == 'that')):
                return True
            if infinitival and first == 'to' and controller is None:
                return True
        return False

    def linked_antecedent(self, w):
        # Require an independent first-person commitment in the same sentence.
        children = self.children[w.id]
        if (w.deprel not in {'conj', 'parataxis'} or
                not any(c.deprel == 'nsubj' and c.text.lower() in {'i','we'} for c in children) or
                not any(c.deprel == 'aux' and c.lemma in {'will','should'} for c in children)):
            return None
        report = self.words.get(w.head)
        if report is None or report.lemma not in REPORT or report.id >= w.id:
            return None
        if any(c.lemma in {'not','never'} for c in self.children[report.id]):
            return None
        targets = [c for c in self.children[report.id] if c.deprel in {'xcomp','ccomp'} and c.id < w.id]
        if len(targets) != 1:
            return None
        target = targets[0]
        if target.lemma in CONTROL:
            if any(c.lemma in {'not','never'} for c in self.children[target.id]):
                return None
            options = [c for c in self.children[target.id] if c.deprel == 'xcomp']
            if len(options) != 1:
                return None
            target = options[0]
        if (target.upos != 'VERB' or any(c.deprel == 'conj' or c.lemma in {'not','never'}
                                        for c in self.children[target.id])):
            return None
        return report, target

    def recipient_ids(self, predicate, nominal):
        if not self.refine or predicate.lemma not in {'send','forward','email','deliver','write'}:
            return []
        # Recover a syntactic disagreement only when the nominal dependent is
        # a complete 'to' PP and the independent phrase tree puts it under VP.
        for c in self.children[nominal.id]:
            if c.deprel == 'nmod':
                ids = sorted(self.subtree(c, CLAUSAL))
                if (min(ids),max(ids)+1) in self.vp_pp and self.words[ids[0]].text.lower() == 'to':
                    return ids
        return []

    def infinitival_argument(self, clause, role):
        """Keep the infinitive's text and structure under its grammatical owner."""
        if clause.upos != 'VERB' or not any(
            marker.deprel == 'mark' and marker.lemma == 'to'
            for marker in self.children[clause.id]
        ):
            return None
        return {
            'role': role,
            'text': self.argument_text(self.subtree(clause, {'parataxis'})),
            'action': clause.lemma,
            'arguments': self.arguments(clause),
        }

    def object_clause_arguments(self, obj):
        """Emit object qualifiers, retaining clause text when they leave details unrepresented."""
        for clause in self.children[obj.id]:
            if clause.id <= obj.id or clause.deprel.split(':')[0] != 'acl':
                continue
            children = self.children[clause.id]
            if clause.deprel != 'acl:relcl':
                infinitive = self.infinitival_argument(clause, 'obj.infinitive')
                if infinitive is not None:
                    yield infinitive
                    continue
                if clause.xpos not in {'VBG', 'VBN'} or any(
                    w.deprel == 'mark' and w.lemma == 'to' for w in children
                ):
                    continue

            qualifiers = []
            covered = set()
            for child in children:
                if child.deprel.split(':')[0] == 'obl':
                    marker = next((w for w in self.children[child.id] if w.deprel == 'case'), None)
                    if marker is not None:
                        ids = self.subtree(child, CLAUSAL)
                        qualifiers.append(self.prepositional_argument(marker, ids, prefix='obj.'))
                        covered.update(ids)

            # A qualifier can stand alone for "associated with X". Keep the clause
            # if that would discard another detail such as a subject, time or negation.
            ids = self.subtree(clause, {'advcl', 'parataxis'})
            remainder = (self.words[i] for i in ids if i != clause.id and i not in covered)
            if not qualifiers or any(w.deprel.split(':')[0] not in {'aux', 'punct'} for w in remainder):
                yield {'role': 'obj.clause', 'text': self.argument_text(ids).strip(' ,—')}
            yield from qualifiers
            for child in children:
                if child.deprel.split(':')[0] == 'advcl':
                    purpose = self.infinitival_argument(child, 'obj.purpose')
                    if purpose is not None:
                        yield {**purpose, 'head': clause.text}

    def object_for_phrases(self, obj):
        """Yield direct nominal 'for' modifiers and their full prepositional spans."""
        for complement in self.children[obj.id]:
            if complement.deprel != 'nmod' or complement.upos not in {'NOUN', 'PROPN', 'PRON'}:
                continue
            marker = next((w for w in self.children[complement.id]
                           if w.deprel == 'case' and w.lemma == 'for' and w.id > obj.id), None)
            if marker is None:
                continue
            ids = sorted(self.subtree(complement, CLAUSAL))
            if ids[0] == marker.id and ids[-1] - ids[0] + 1 == len(ids):
                yield marker, ids

    def arguments(self, w):
        args = []
        for c in self.children[w.id]:
            rel = c.deprel.split(':')[0]
            case = [k for k in self.children[c.id] if k.deprel in {'case','mark'}]
            prep = case[0].text.lower() if case else None
            role = None
            excluded = CLAUSAL
            if rel in {'obj','iobj'}:
                role = rel
            elif rel == 'obl' and prep:
                args.append(self.prepositional_argument(case[0], self.subtree(c, excluded)))
                continue
            elif rel in {'xcomp','ccomp'} and w.lemma in REPORT | {'notify'}:
                role, excluded = 'requested_content', {'advcl','parataxis'}
            elif rel in {'advcl','ccomp'} and prep in {'if','whether','that'} and w.lemma in QUERY:
                role = 'question'
            elif rel == 'advcl' and prep in {'if','unless','when'}:
                role = 'condition'
            elif rel == 'advcl' and prep == 'as' and w.lemma == 'mark':
                role = 'result_state'
            elif rel == 'advcl' and prep == 'to':
                purpose = self.infinitival_argument(c, 'purpose')
                if purpose is not None:
                    args.append(purpose)
            elif rel == 'compound' and c.deprel == 'compound:prt':
                role = 'particle'
            if role:
                ids = self.subtree(c, excluded)
                for_phrases = []
                if role == 'obj':
                    recipient = self.recipient_ids(w, c)
                    if recipient:
                        args.append(self.prepositional_argument(self.words[recipient[0]], recipient))
                        ids = sorted(set(ids) - set(recipient))
                    for_phrases = list(self.object_for_phrases(c))
                    ids = sorted(set(ids).difference(*(span for _, span in for_phrases)))
                args.append({'role':role,'text':self.argument_text(ids)})
                if role == 'obj':
                    args.extend(self.prepositional_argument(marker, span) for marker, span in for_phrases)
                    args.extend(self.object_clause_arguments(c))
        return args

    def proceed_action(self, w):
        """Recognize the named activity in 'proceed to V' or 'proceed with V-ing'."""
        if w.lemma != 'proceed':
            return None
        candidates = []
        for child in self.children[w.id]:
            markers = [m for m in self.children[child.id] if m.deprel in {'mark', 'case'}]
            if len(markers) != 1 or child.id < w.id:
                continue
            marker = markers[0]
            infinitive = child.deprel == 'xcomp' and marker.lemma == 'to' and child.xpos == 'VB'
            gerund = child.deprel in {'advcl', 'obl', 'xcomp'} and marker.lemma == 'with' and child.xpos == 'VBG'
            if (infinitive or gerund) and child.upos == 'VERB':
                candidates.append(child)
        return candidates[0] if len(candidates) == 1 else None

    def simple_phrase(self, head, *, omit=()):
        """Exact, contiguous noun phrase, with no competing embedded entity or action."""
        ids = sorted(set(self.subtree(head)).difference(omit))
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            return None
        allowed = {'det', 'amod', 'compound', 'flat', 'nummod', 'punct', 'fixed', 'advmod'}
        if any(self.words[i].deprel.split(':')[0] not in allowed for i in ids if i != head.id):
            return None
        return self.excerpt(ids)

    def nominal_focus(self, predicate, boundary, node):
        """Find uniquely supported singular/plural entities among this predicate's earlier arguments."""
        choices = {'it': [], 'them': []}
        end_of_source = predicate.id
        for head in self.children[predicate.id]:
            rel = head.deprel.split(':')[0]
            if head.id >= boundary or rel not in {'obj', 'iobj', 'obl'}:
                continue
            ids = sorted(self.subtree(head, CLAUSAL))
            if ids[-1] >= boundary:
                return {}
            end_of_source = max(end_of_source, ids[-1])
            words = [self.words[i] for i in ids]
            if any(w.deprel == 'conj' or w.lemma in {'no', 'neither', 'either'} for w in words):
                return {}
            markers = [w for w in self.children[head.id] if w.deprel == 'case']
            search_object = (rel == 'obl' and predicate.lemma in {'search', 'look'}
                             and any(w.lemma == 'for' for w in markers))
            for word in words:
                if word.xpos not in {'NN', 'NNP', 'NNS', 'NNPS'} or word.deprel in {'compound','flat'}:
                    continue
                number = 'them' if word.xpos in {'NNS', 'NNPS'} else 'it'
                # All entities can compete. Only a common-noun object or search
                # target contributes a reusable referent (proper names may be people).
                phrase = None
                if word.id == head.id and (rel == 'obj' or search_object) and head.upos == 'NOUN':
                    core = [i for i in ids if not any(m.id == i for m in markers)]
                    if core and core == list(range(core[0], core[-1] + 1)):
                        end = core[-1]
                        # Include a contiguous naming phrase, but not its purpose clause.
                        for modifier in self.children[head.id]:
                            if modifier.deprel == 'acl' and modifier.lemma in {'name','call','title'}:
                                names = [w for w in self.children[modifier.id]
                                         if w.deprel in {'obj','xcomp'} and w.upos in {'NOUN','PROPN'}]
                                if len(names) == 1:
                                    alias = sorted([modifier.id, *self.subtree(names[0], CLAUSAL)])
                                    if alias == list(range(end+1, alias[-1]+1)) and alias[-1] < boundary:
                                        end = alias[-1]
                        end_of_source = max(end_of_source, end)
                        phrase = (self.excerpt([core[0], end]), head.lemma, head.id)
                choices[number].append(phrase)
        evidence = self.excerpt([predicate.id, end_of_source])
        focus = {}
        for number, options in choices.items():
            if len(options) == 1 and options[0]:
                text, lemma, word = options[0]
                focus[number] = Mention(text, lemma, node, word, evidence)
        return focus

    def simple_context(self, action, controller=None):
        # Avoid borrowing entities across quotations, hypotheticals, or multiple actions.
        verbs = {action.id}
        if controller is not None:
            verbs.add(controller.id)
        root = next(w for w in self.words.values() if w.head == 0)
        return root.id in verbs and not any(
            w.deprel.split(':')[0] in CLAUSAL | {'conj'} and w.id not in verbs
            or w.lemma in {'not', 'never', 'if', 'unless'}
            for w in self.words.values())

    def fronted_possession(self, action, pronoun, node):
        """Bind the object in 'Now that I have X, I should review it/them'."""
        if action.deprel != 'root' or not any(w.deprel == 'nsubj' and w.lemma.lower() in {'i', 'we'}
                                              for w in self.children[action.id]):
            return None
        before = [w for w in self.words.values() if w.id < action.id]
        if [w.text.lower() for w in before[:2]] != ['now', 'that']:
            return None
        for source in before:
            if source.lemma != 'have' or source.upos != 'VERB':
                continue
            link = self.words.get(source.head)
            attached = source.head == action.id and source.deprel == 'advcl'
            via_now = source.deprel == 'ccomp' and link and link.id == 1 and link.head == action.id
            if not (attached or via_now):
                continue
            cs = self.children[source.id]
            if any(w.deprel not in {'mark', 'nsubj', 'obj', 'punct'} for w in cs):
                continue
            if not any(w.deprel == 'nsubj' and w.lemma.lower() in {'i', 'we'} for w in cs):
                continue
            objects = [w for w in cs if w.deprel == 'obj']
            if len(objects) != 1 or any(w.deprel not in {'det', 'amod', 'compound', 'nmod:poss', 'nummod'}
                                        for w in self.children[objects[0].id]):
                continue
            source_ids = set(self.subtree(source))
            if any(w.upos in {'NOUN', 'PROPN', 'VERB'} and w.id not in source_ids for w in before):
                continue
            return self.nominal_focus(source, action.id, node).get(pronoun)
        return None

    def bind_nominals(self, action, controller, args, emitted, previous, node):
        governor = controller if controller is not None else action
        bound = {}
        for arg in args:
            role = arg['role']
            if role not in {'obj', 'to', 'into'}:
                continue
            tokens = arg['text'].lower().split()
            if tokens not in (['it'], ['them']):
                continue
            pronoun = tokens[0]
            arg.setdefault('surface', arg['text'])
            arg['resolution'] = 'unresolved'
            # In "find the files and read them", the overt lexical action may
            # itself govern the coordination even when it is inside "need to".
            anchor = action if action.deprel in {'conj', 'parataxis'} else governor
            parallel = anchor.deprel in {'conj', 'parataxis'}
            source = self.words.get(anchor.head)
            referent = None
            if parallel and source is not None and source.id < anchor.id:
                accepted = source.id in emitted
                past = (source.deprel == 'root' and source.xpos in {'VBD','VBN'} and any(
                    c.deprel == 'nsubj' and c.text.lower() in {'i','we'} for c in self.children[source.id]))
                intervening = any(c.id < anchor.id and (
                    c.deprel.split(':')[0] in CLAUSAL | {'conj'} or c.xpos == 'MD' and past
                    or c.lemma in {'not','never','no'}) for c in self.children[source.id])
                disjunctive = any(c.deprel == 'cc' and c.lemma in {'or','nor','but'}
                                  for c in self.children[anchor.id])
                if (accepted or past) and not intervening and not disjunctive:
                    referent = self.nominal_focus(source, anchor.id, node).get(pronoun)
            elif role == 'obj' and self.words[1].lemma.lower() == 'now':
                referent = self.fronted_possession(action, pronoun, node)
            if referent is None and not parallel and previous and self.simple_context(action, controller):
                # A simple transitive action can continue the previous entity
                # even when the verb changes, e.g. find -> read -> archive.
                candidate = previous.get(pronoun)
                # Local competing nouns take precedence over an earlier node.
                target = next((c for c in self.children[action.id] if c.text.lower() == pronoun), None)
                if target is None:
                    continue
                wanted = {'NNS','NNPS'} if pronoun == 'them' else {'NN','NNP'}
                competing = any(
                    q.id < target.id and q.xpos in wanted and q.deprel not in {'compound','flat'}
                    and not (role == 'into' and q.head == action.id and q.deprel == 'obj')
                    for q in self.words.values())
                if candidate and not competing:
                    noun = candidate.lemma
                    direct_object = role == 'obj'
                    destination = (role == 'into' and action.lemma == 'move' and
                                   noun in {'folder','directory','mailbox','archive'})
                    reply_target = (role == 'to' and action.lemma == 'reply' and
                                    noun in {'message','email','letter','thread'})
                    if direct_object or destination or reply_target:
                        referent = candidate
            if referent is not None:
                arg.update(text=referent.text, resolution='resolved',
                           inferred_referent=referent.text, referent_evidence=referent.evidence,
                           referent_word=referent.word)
                if referent.node is not None:
                    arg['referent_node'] = referent.node
                bound[pronoun] = referent
        return bound

    def bind_comparison_topic(self, action, frame):
        # A reported proposition can supply a topic for a bare "compare". Keep
        # it as attributed evidence, not as an observed destination or tool call.
        if action.deprel not in {'parataxis','conj'}:
            return
        if any(c.deprel.split(':')[0] in {'obj','iobj','obl','ccomp','xcomp','advcl','orphan'}
               for c in self.children[action.id]):
            return
        report = self.words.get(action.head)
        if report is None or report.deprel != 'root' or report.lemma not in REPORT or report.id >= action.id:
            return
        earlier = [w for w in self.children[report.id] if w.id < action.id]
        clauses = [w for w in earlier if w.deprel == 'ccomp']
        speakers = [w for w in earlier if w.deprel == 'nsubj']
        if len(clauses) != 1 or len(speakers) != 1 or any(
                w.deprel in {'conj','parataxis','advcl'} or w.lemma in {'not','never'} for w in earlier):
            return
        clause = clauses[0]
        ids = sorted(self.subtree(clause))
        if ids[-1] >= action.id or ids != list(range(ids[0], ids[-1] + 1)):
            return
        words = [self.words[i] for i in ids]
        degree = {w.id for w in words if w.xpos in {'JJR','RBR','JJS','RBS'}}
        if (any(w.deprel.split(':')[0] in CLAUSAL | {'conj'} or w.lemma in {'no','not','never','neither'}
                for w in words if w.id != clause.id) or
                not degree):
            return
        speaker = self.simple_phrase(speakers[0])
        if not speaker:
            return
        topic = {'source': speaker, 'evidence': self.excerpt(ids), 'inferred': True}
        for dep, field in [('nsubj', 'claim_subject'), ('obj', 'claim_object')]:
            heads = [c for c in self.children[clause.id] if c.deprel == dep]
            if len(heads) == 1:
                phrase = self.simple_phrase(heads[0])
                if phrase:
                    topic[field] = phrase
        frame['inferred_topic'] = topic

        # Anchor the comparative to the object it modifies ("a cheaper plan")
        # or to the subject of a copular adjective ("the plan is cheaper").
        children = self.children[clause.id]
        objects = [c for c in children if c.deprel == 'obj']
        subject = topic.get('claim_subject')
        if len(objects) == 1:
            anchor = objects[0]
            properties = {c.id for c in self.children[anchor.id] if c.deprel in {'amod','nummod','det'}}
        elif clause.upos == 'ADJ' and subject and any(c.deprel == 'cop' for c in children):
            anchor = clause
            properties = {clause.id}
        else:
            return
        graded = degree | {w.head for w in words if w.id in degree and w.deprel == 'advmod'}
        if properties.isdisjoint(graded):
            return

        # A nominal "than X" supplies a second operand only when attached to
        # this comparison. Keep its words separate from the first operand.
        markers = [w for w in words if w.lemma == 'than']
        omitted = ()
        standard = None
        if markers:
            if len(markers) != 1 or markers[0].deprel != 'case':
                return
            marker = markers[0]
            other = self.words[marker.head]
            if other.deprel not in {'nmod','obl'} or other.head not in properties | {anchor.id}:
                return
            standard = self.simple_phrase(other, omit=[marker.id])
            if not standard:
                return
            omitted = self.subtree(other)
        item = self.simple_phrase(anchor, omit=omitted) if objects else subject
        if not item:
            return

        args = frame['arguments']
        args.append({'role': 'obj', 'text': item, 'inferred': True})
        if standard:
            args.append({'role': 'obj', 'text': standard, 'inferred': True})

    def evaluated_infinitive(self, root):
        """An impersonal positive evaluation can introduce a contemplated action."""
        cs = self.children[root.id]
        it = next((w for w in cs if w.deprel == 'expl' and w.lemma.lower() == 'it'), None)
        if it is None or root.upos != 'ADJ':
            return None
        conditional = any(w.deprel == 'aux' and w.lemma == 'would' and it.id < w.id for w in cs)
        necessity = (root.lemma in {'necessary', 'essential', 'crucial'} and
                     any(w.deprel == 'cop' and w.xpos in {'VBZ','VBP'} for w in cs) and
                     not any(w.deprel == 'aux' for w in cs))
        if necessity and any(w.lemma in {'no','not','never'} and w.id < root.id for w in self.words.values()):
            return None
        if necessity and (any(c.lemma in {'hardly','rarely','never','not','unclear'} or
                              c.deprel.split(':')[0] == 'obl' for c in cs) or
                          any(w.text in {'"', '“', '”', "''"} for w in self.words.values())):
            return None
        if not necessity and not (conditional and root.lemma in
                                  {'convenient', 'helpful', 'useful', 'beneficial', 'advisable'}):
            return None
        if not any(w.deprel == 'cop' and w.lemma == 'be' for w in cs):
            return None
        if any(w.text == '?' or w.lemma in {'if', 'unless'} for w in self.words.values()):
            return None
        if any(w.lemma in {'not', 'never', 'hardly', 'scarcely', 'less'} for w in cs):
            return None
        for verb in cs:
            if verb.deprel == 'csubj' and verb.xpos == 'VB':
                children = self.children[verb.id]
                if any(w.deprel.startswith('nsubj') or w.lemma in {'not', 'never'} for w in children):
                    continue
                if any(w.deprel == 'mark' and w.lemma == 'to' for w in children):
                    return verb
        return None

    def independent_decision(self, report, verb):
        """Recover a comma-separated agent decision attached as a second complement."""
        if (report.deprel != 'root' or report.lemma not in REPORT | {'recommend', 'indicate'} or
                verb.upos != 'VERB'):
            return False
        speakers = [c for c in self.children[report.id] if c.deprel == 'nsubj']
        if len(speakers) != 1 or speakers[0].text.lower() in {'i', 'we', 'you'}:
            return False
        children = self.children[verb.id]
        subject = next((c for c in children if c.deprel == 'nsubj' and c.text.lower() in {'i','we'}), None)
        modals = {c.lemma for c in children if c.deprel == 'aux'}
        if subject is None or not (modals & {'will','shall','must','should'} or verb.lemma in CONTROL):
            return False
        boundary = subject.id - 1
        if boundary > 0 and self.words[boundary].text.lower() in {'so','but'}:
            boundary -= 1
        if boundary <= report.id or self.words[boundary].text != ',':
            return False
        # A comma after a bare reporting verb or a parenthetical does not end reported speech.
        # Require a substantive complement before the comma. Abstain on quoted,
        # conditional and question contexts, or an earlier first-person instruction.
        if any(w.text == '?' or w.lemma.lower() in {'if','unless'} for w in self.words.values()):
            return False
        quote_tokens = [w for w in self.words.values() if w.upos == 'PUNCT' and
                        w.text in {'"', '“', '”', '‘', '’', "'", "``", "''"}]
        if len(quote_tokens) % 2 or any(w.id > boundary for w in quote_tokens):
            return False
        if 'will' not in modals and any(w.text.lower() in {'i','we'} and w.deprel == 'nsubj'
                                        for w in self.words.values() if report.id < w.id < boundary):
            return False
        return any(report.id < c.id < boundary and (
            c.deprel in {'obj','ccomp','xcomp'} or
            c.deprel == 'advcl' and any(k.deprel == 'mark' and k.lemma == 'to' for k in self.children[c.id]) or
            c.deprel == 'obl' and any(k.xpos == 'VBG' and k.deprel == 'case'
                                      for k in self.children[c.id]))
                   for c in self.children[report.id])

    def frames(self, previous=None, node=None, *, imperative_list=False):
        found = []
        emitted = {}
        self.next_focus = {}
        bindings = {}
        def walk(w, inherited=None, controller=None, relation=None):
            children = self.children[w.id]
            subjects = [x for x in children if x.deprel.startswith('nsubj')]
            first = any(x.text.lower() in {'i','we'} for x in subjects)
            aux = [x for x in children if x.deprel == 'aux']
            mode = next((MODES[x.lemma] for x in aux if x.lemma in MODES), None)
            locally_negated = any(x.lemma in {'not','never'} and x.deprel == 'advmod' for x in children)
            subordinate = self.in_subordinate_clause(w, controller)
            if subordinate and w.deprel == 'parataxis' and first and mode == 'stated_plan':
                # A preceding reason clause can overrun the comma in the constituency tree.
                parent = self.words[w.head]
                start = subjects[0].id
                if (parent.deprel == 'root' and parent.lemma not in REPORT and start > 1 and
                        self.words[start-1].text == ',' and
                        any(c.deprel == 'obj' and c.id < start for c in self.children[parent.id]) and
                        not any(c.lemma in {'if','unless'} or c.text in {'?', '"', '“', '”', "'", "''"}
                                for c in self.words.values())):
                    subordinate = False
            allowed = w.upos == 'VERB' and not locally_negated and not subordinate
            own = None
            if allowed:
                if first and (mode or w.lemma in CONTROL):
                    own = mode or 'stated_goal'
                elif not subjects and inherited:
                    own = inherited
                elif not subjects and w.deprel == 'root' and w.xpos in {'VB','VBG'} and w.lemma not in DISCOURSE:
                    own = 'implicit_subject'
                elif first and w.xpos == 'VBG' and any(x.lemma == 'be' for x in aux):
                    own = 'ongoing'
            # Stanza expands Let's into let + a first-person object with lemma we.
            is_control = w.lemma in CONTROL or (w.lemma == 'let' and any(
                x.deprel in {'obj', 'iobj'} and (
                    x.text.lower() in {'me', 'us'} or (x.lemma == 'we' and x.text.lower() in {"'s", '’s'})
                ) for x in children))
            if (allowed and not own and is_control and w.lemma == 'let' and
                    w.deprel == 'parataxis' and self.words[w.head].deprel == 'root' and
                    any(c.deprel == 'expl' or (c.deprel == 'nsubj' and c.text.lower() in {'i','we'})
                        for c in self.children[w.head]) and self.words[w.head].lemma not in REPORT and
                    not subjects and
                    not any(c.text in {'"', '“', '”', '?'} for c in self.words.values())):
                own = 'implicit_subject'
            parent_for_coord = relation
            activity = self.proceed_action(w) if own and self.refine else None
            if activity is not None:
                walk(activity, own, w, relation)
                parent_for_coord = activity.lemma
            elif own and is_control:
                for c in children:
                    if c.deprel == 'xcomp':
                        walk(c, own, w, relation)
            elif own:
                args = self.arguments(w)
                mode_name = own
                if w.lemma == 'do' and any(a['role']=='obj' and a['text'].lower() in {'that','it','this'} for a in args):
                    mode_name = 'unresolved_reference'
                frame = {'action':w.lemma,'mode':mode_name,'evidence':self.excerpt([w.id]),'arguments':args}
                if self.refine and mode_name == 'unresolved_reference':
                    link = self.linked_antecedent(w)
                    if link:
                        report, target = link
                        frame.update(action=target.lemma, mode=own, arguments=self.arguments(target),
                                     reference={'report':report.text,'antecedent':target.text,
                                                'commitment':self.excerpt(self.subtree(w, CLAUSAL))})
                if self.refine:
                    if mode_name != 'unresolved_reference' and w.lemma != 'do':
                        bindings[w.id] = self.bind_nominals(w, controller, args, emitted, previous, node)
                    if w.lemma == 'compare' and not args:
                        self.bind_comparison_topic(w, frame)
                if relation is not None:
                    frame['coordination'] = relation
                parent = self.words.get(w.head)
                if parent and parent.lemma in REPORT and w.deprel == 'parataxis':
                    if any(a <= w.id < b and a > parent.id and label == 'S' and
                           any(k.deprel == 'ccomp' and a <= k.id < b for k in self.children[parent.id])
                           for a,b,label,_ in self.spans):
                        frame['scope_warning'] = 'constituency_groups_with_reported_clause'
                emitted[w.id] = controller
                found.append(frame)
                parent_for_coord = w.lemma
            for c in children:
                if c.deprel == 'conj':
                    # An explicit new subject resets control. Negation is never propagated as a plan.
                    has_subject = any(x.deprel.startswith('nsubj') for x in self.children[c.id])
                    if own or has_subject:
                        conjunction = next((x.lemma for x in self.children[c.id] if x.deprel=='cc'), 'and')
                        edge = {'head':parent_for_coord,'connective':conjunction} if parent_for_coord and own and not has_subject else None
                        walk(c, own, controller, edge)
                elif c.deprel == 'parataxis':
                    walk(c)
        root = next(w for w in self.words.values() if w.head == 0)
        candidate = self.evaluated_infinitive(root) if self.refine else None
        if candidate is not None:
            mode = 'required' if root.lemma in {'necessary','essential','crucial'} else 'possible'
            walk(candidate, mode, root)
        elif self.refine and root.lemma in {'step','essential','crucial','necessary'}:
            cs = self.children[root.id]
            if ((root.upos == 'ADJ' or any(c.deprel == 'amod' and c.lemma == 'next' for c in cs)) and
                    any(c.deprel == 'cop' and c.xpos in {'VBZ','VBP'} for c in cs) and
                    not any(w.lemma in {'not','never','if','unless'} or w.text == '?' for w in self.words.values())):
                for action in cs:
                    if action.deprel == 'csubj' and action.xpos == 'VBG':
                        walk(action, 'stated_plan' if root.lemma == 'step' else 'required', root)
        walk(root, controller=root if imperative_list else None)
        if self.refine and not found and root.xpos == 'VBN':
            cs = self.children[root.id]
            modal = next((c.lemma for c in cs if c.deprel == 'aux' and c.lemma in {'should','must'}), None)
            patient = next((c for c in cs if c.deprel == 'nsubj:pass'), None)
            attributed = any(w.text.lower() in {'according','per'}
                             for w in self.words.values() if patient and w.id < patient.id)
            if (modal and patient and not attributed and
                    any(c.lemma == modal and c.id > patient.id for c in cs) and
                    any(c.deprel == 'aux:pass' and c.xpos == 'VB' for c in cs)
                    and not any(w.lemma in {'not','never','if','unless','by'} or
                                w.text in {'?', '"', '“', '”'} for w in self.words.values())
                    and not any(c.deprel.split(':')[0] in {'conj','parataxis','advcl'}
                                and c.id < root.id for c in cs)):
                obj = self.argument_text(self.subtree(patient, {'punct'}))
                found.append({'action': root.lemma, 'mode': MODES[modal], 'actor': 'unspecified',
                              'evidence': self.excerpt([root.id]),
                              'arguments': [{'role':'obj', 'text':obj}, *self.arguments(root)]})
        if self.refine and not found and root.lemma in REPORT | {'recommend','indicate'}:
            # UD may attach the decision to the reporting verb or its first complement.
            for verb in self.words.values():
                parent = self.words.get(verb.head)
                direct = verb.head == root.id and verb.deprel == 'ccomp'
                nested = (verb.deprel == 'parataxis' and parent is not None and parent.head == root.id
                          and parent.deprel in {'xcomp','ccomp'})
                if (direct or nested) and self.independent_decision(root, verb):
                    walk(verb)
        if self.refine and len(emitted) == 1 and found[0]['mode'] != 'possible':
            ident, controller = next(iter(emitted.items()))
            w = self.words[ident]
            if self.simple_context(w, controller):
                self.next_focus = self.nominal_focus(w, len(self.words) + 1, node)
                # A sole resolved argument keeps its original mention alive.
                # Mixed or ambiguous arguments cannot relay that binding.
                arguments = [a for a in found[0]['arguments'] if a['role'] != 'particle']
                if len(arguments) == 1 and arguments[0].get('resolution') == 'resolved':
                    self.next_focus = bindings[ident]
        return found


class IntentExtractor:
    """One instance per execution chain. Call extract_step in execution order."""

    def __init__(self, pipeline=None, *, include_modifiers=False):
        if pipeline is None:
            pipeline = stanza.Pipeline('en', processors=','.join(MODELS), package=MODELS,
                                       download_method=None, use_gpu=False, verbose=False)
        self.pipeline = pipeline
        self.include_modifiers = include_modifiers
        self._focus = {}

    def extract_step(self, text, node_id):
        """Return (node ID, original sentence, frames) records for one incoming step."""
        return self.extract_document(self.pipeline(text), node_id)

    def extract_query(self, text, node_id="Q"):
        """Start a new task with a user request, including polite requests and object qualifiers."""
        return self.extract_document(self.pipeline(text), node_id, query=True)

    def extract_step_candidates(self, text, node_id, candidates):
        """Extract one step sentence from scored CT/UD candidates on the original text."""
        from .candidates import weighted_frames
        frames, focus, diagnostics = weighted_frames(
            self.pipeline, text, node_id, candidates, self._focus, self.include_modifiers)
        self._focus = focus
        return [(node_id, text, frames)], diagnostics

    def repair_fragment(self, sentence, text):
        # A bare VP with a nominal-tagged head contradicts the constituency parse.
        # Retag only the head and rerun lemma/UD on the original words and offsets.
        phrase = sentence.constituency
        while phrase.label in {'ROOT', 'S'} and phrase.children:
            phrase = phrase.children[0]
        first = sentence.words[0]
        # In elliptical transfer actions, a second endpoint can disambiguate a
        # nominal analysis (e.g. "Syncing email with compliance").
        endpoint = {'syncing': {'with'}, 'transferring': {'to','from'},
                    'moving': {'to','into','from'}, 'copying': {'to','into','from'}}.get(first.text.lower())
        root = next(w for w in sentence.words if w.head == 0)
        paired = (endpoint and first.deprel == 'compound' and first.head == root.id and
                  first.xpos in {'NN','NNS','VBG'} and root.upos in {'NOUN','PROPN'} and
                  any(w.deprel == 'case' and w.lemma in endpoint and
                      sentence.words[w.head-1].head == root.id for w in sentence.words))
        if ((not paired and (phrase.label != 'VP' or len(phrase.children) < 2 or
                phrase.children[0].label not in {'NN', 'NNS'} or
                first.head != 0 or first.xpos != phrase.children[0].label or
                not any(child.label in {'NP', 'PP'} for child in phrase.children[1:]))) or
                any(w.xpos == 'MD' or w.deprel.startswith(('nsubj', 'csubj')) or
                    w.text in {'?', '"', '“', '”'} for w in sentence.words)):
            return None
        gerund = first.text.lower().endswith('ing')
        if not gerund and not any(child.label == 'NP' for child in phrase.children[1:]):
            return None
        repaired = Document([sentence.to_dict()], text=text)
        head = repaired.sentences[0].words[0]
        head.upos, head.xpos = 'VERB', 'VBG' if gerund else 'VB'
        self.pipeline.processors['lemma'].process(repaired)
        self.pipeline.processors['depparse'].process(repaired)
        candidate = repaired.sentences[0]
        if (candidate.words[0].head != 0 or
                gerund and candidate.words[0].lemma.lower() == first.text.lower()):
            return None
        if paired:
            objects = {w.id for w in candidate.words if w.head == 1 and w.deprel == 'obj'}
            if not objects:
                return None
            mark = next((w for w in candidate.words if w.deprel == 'case' and w.lemma in endpoint), None)
            if mark is None:
                return None
            target = candidate.words[mark.head-1]
            if target.head in objects and target.deprel == 'nmod':
                # Verify disputed attachment in an explicit progressive sentence;
                # transfer only the dependency between aligned original words.
                start, end = sentence.tokens[0].start_char, sentence.tokens[-1].end_char
                variant = self.pipeline('I am ' + text[start:start+1].lower() + text[start+1:end])
                if len(variant.sentences) != 1:
                    return None
                probe = variant.sentences[0].words
                if (len(probe) != len(candidate.words) + 2 or
                        any(a.text.casefold() != b.text.casefold() for a,b in zip(candidate.words, probe[2:])) or
                        probe[2].head != 0 or probe[target.id+1].head != 3 or
                        probe[target.id+1].deprel != 'obl'):
                    return None
                target.head, target.deprel = 1, 'obl'
            if target.head != 1 or target.deprel != 'obl':
                return None
            if first.text.lower() == 'syncing' and candidate.words[0].lemma == 'synce':
                candidate.words[0].lemma = 'sync'
        candidate.constituency = sentence.constituency
        return Sentence(candidate, text, joint=True, refine=True)

    def imperative_reading(self, document):
        # Test a case variant when a bare directive initially looks nominal.
        # Require a verb reading with an overt argument on the unchanged source words.
        text = document.text
        words = [w for s in document.sentences for w in s.words]
        if (len(words) < 2 or words[0].start_char != 0 or not text[0].isupper() or
                not text[0].isascii() or words[0].xpos not in {'NN','NNS','JJ','RB','VBG'} or
                words[0].deprel not in {'root','compound','amod','advmod'} or
                any(w.text in {'!', '?', ';', '"', '“', '”', '.'} for w in words[:-1])):
            return None
        original_root = next(w for w in words if w.head == 0)
        if any(w.head == original_root.id and w.deprel.startswith(('nsubj','csubj','cop'))
               for w in document.sentences[0].words):
            return None
        variant = text[0].lower() + text[1:]
        if not variant.rstrip().endswith(('.', '!', '?')):
            variant = variant.rstrip() + '.'
        doc = self.pipeline(variant)
        sentences = repaired_sentences(doc, self.pipeline)
        if len(sentences) != 1:
            return None
        sentence = sentences[0]
        verb = sentence.words[0]
        if (words[0].text.lower().endswith('ing') and variant.endswith('.') and
                not text.rstrip().endswith('.') and (verb.xpos != 'VBG' or verb.head != 0)):
            plain = self.pipeline(variant[:-1])
            options = repaired_sentences(plain, self.pipeline)
            if len(options) == 1:
                sentence = options[0]
                verb = sentence.words[0]
        if (verb.start_char != 0 or verb.end_char != words[0].end_char or
                verb.head != 0 or verb.xpos not in {'VB','VBG'} or verb.upos != 'VERB'):
            return None
        parsed = Sentence(sentence, text, joint=True, refine=True)
        obj = next((w for w in parsed.children[verb.id] if w.deprel == 'obj'), None)
        if (any(w.deprel.startswith(('nsubj','csubj','cop')) or w.lemma in {'not','never'}
                for w in parsed.children[verb.id]) or
                any(w.text == '?' for w in sentence.words)):
            return None
        # A capitalized noun followed by another bare noun is ambiguous ("Record player").
        if obj is not None and (verb.xpos == 'VBG' or
                any(c.deprel in {'det','amod','nummod','nmod:poss'} for c in parsed.children[obj.id]) or
                any(c.deprel in {'obl','advcl','conj'} for c in parsed.children[verb.id])):
            return parsed
        if verb.xpos == 'VBG' and verb.lemma == 'proceed' and any(c.deprel == 'xcomp' for c in parsed.children[verb.id]):
            return parsed
        if not (verb.lemma in {'login','logout'} and any(c.deprel == 'obl' for c in parsed.children[verb.id])):
            return None
        return parsed

    def directed_command(self, document):
        """Validate an ambiguous imperative with an object and a separate role."""
        if len(document.sentences) != 1:
            return None
        sent = document.sentences[0]
        words = sent.words
        roles = {'forward': {'to'}, 'record': {'for','with'}, 'schedule': {'for','with'},
                 'update': {'with'}}.get(words[0].text.lower())
        if (not roles or words[0].xpos not in {'NN','JJ','RB'} or
                not any(w.lemma in roles for w in words) or
                any(w.deprel.startswith(('nsubj','csubj','cop')) or w.xpos == 'MD' or
                    w.lemma in {'if','unless','not','never'} or w.text in {'?', '"', '“', '”'}
                    for w in words)):
            return None
        text = document.text[sent.tokens[0].start_char:sent.tokens[-1].end_char]
        probe = self.pipeline('Please ' + text[0].lower() + text[1:])
        if len(probe.sentences) != 1:
            return None
        ps = probe.sentences[0].words
        if (len(ps) != len(words) + 1 or ps[1].head != 0 or ps[1].xpos != 'VB' or
                any(x.text.casefold() != y.text.casefold() for x,y in zip(words,ps[1:])) or
                any(w.head == 1 for w in ps[1:])):
            return None
        obj = next((w for w in ps if w.head == 2 and w.deprel == 'obj'), None)
        if obj is None or not any(w.deprel == 'case' and w.lemma in roles and
                ps[w.head-1].head in {2,obj.id} for w in ps):
            return None
        if ps[1].lemma == 'record' and obj.xpos == 'NN' and not any(
                w.head == obj.id and w.deprel in {'amod','det','compound','nummod'} for w in ps):
            return None
        # Project the verified UD edges onto the original sentence, retaining
        # its source offsets and omitting the synthetic politeness marker.
        copy = Document([sent.to_dict()], text=document.text).sentences[0]
        for w, alternative in zip(copy.words, ps[1:]):
            w.upos, w.xpos, w.lemma = alternative.upos, alternative.xpos, alternative.lemma
            w.head = alternative.head - 1 if alternative.head else 0
            w.deprel = alternative.deprel
        copy.constituency = sent.constituency
        return Sentence(copy, document.text, joint=True, refine=True)

    def modal_acronym(self, sentence, text):
        root = next(w for w in sentence.words if w.head == 0)
        cs = [w for w in sentence.words if w.head == root.id]
        token = root
        if root.xpos == 'MD' and root.id < len(sentence.words):
            token = sentence.words[root.id]
            if token.text.lower() not in {'cc','bcc'}:
                return None
        if (token.upos != 'PROPN' or not token.text.isascii() or not token.text.isalpha()
                or not token.text.isupper() or len(token.text) < 2):
            return None
        if (not any(c.deprel == 'nsubj' and c.text.lower() in {'i','we'} for c in cs) or
                not (root.xpos == 'MD' and root.lemma in {'should','will','must'} or
                     any(c.deprel == 'aux' and c.lemma in {'should','will','must'} for c in cs)) or
                any(w.lemma in {'if','unless','not','never'} or w.text in {'?', '"', '“', '”'}
                    for w in sentence.words)):
            return None
        # The modal selects a bare infinitive. Require an overt object after UD reparsing.
        repaired = Document([sentence.to_dict()], text=text)
        target = repaired.sentences[0].words[token.id-1]
        target.upos, target.xpos, target.lemma = 'VERB', 'VB', target.text.lower()
        self.pipeline.processors['depparse'].process(repaired)
        candidate = repaired.sentences[0]
        if target.head != 0 or not any(w.head == target.id and w.deprel == 'obj' for w in candidate.words):
            return None
        candidate.constituency = sentence.constituency
        return Sentence(candidate, text, joint=True, refine=True)

    def imperative_list(self, document, reading):
        """Keep the original coordination when both parses support a leading command."""
        if len(document.sentences) != 1:
            return reading
        source = document.sentences[0]
        head = source.words[0]
        if head.head != 0 or not any(w.head == head.id and w.xpos == 'VB' and
                                     w.deprel in {'parataxis','conj'} for w in source.words):
            return reading
        copy = Document([source.to_dict()], text=document.text).sentences[0]
        copy.constituency = source.constituency
        verb = reading.words[1]
        copy.words[0].upos, copy.words[0].xpos, copy.words[0].lemma = verb.upos, verb.xpos, verb.lemma
        for w in copy.words:
            if w.head != head.id:
                continue
            if w.deprel == 'nmod' and any(c.deprel == 'case' and c.head == w.id for c in copy.words):
                w.deprel = 'obl'
            if (w.deprel == 'parataxis' and w.xpos == 'VB' and
                    not any(c.head == w.id and c.deprel.startswith(('nsubj','csubj','mark')) for c in copy.words)):
                w.deprel = 'conj'
        return Sentence(copy, document.text, joint=True, refine=True)

    def extract_document(self, document, node_id, *, query=False):
        """Use the same extractor when a caller already has a parsed Stanza document."""
        nodes = []
        focus = {} if query else self._focus
        repaired_fragment = False
        sentences = repaired_sentences(document, self.pipeline)
        for i, sentence in enumerate(sentences, 1):
            key = node_id if len(sentences) == 1 else f'{node_id}.{i}'
            parser = Sentence(sentence, document.text, joint=True, refine=True)
            frames = parser.frames(previous=focus, node=key)
            if not frames:
                alternate = self.repair_fragment(sentence, document.text) or self.modal_acronym(sentence, document.text)
                if alternate is not None:
                    repaired = alternate.frames(previous=focus, node=key)
                    if repaired:
                        parser, frames = alternate, repaired
                        repaired_fragment = True
            if not query and not frames:
                frames = explicit_fragment(parser, focus, key)
            if query:
                frames = enrich_query(parser, frames, key)
            if self.include_modifiers:
                add_infinitive_modifiers(parser, frames)
            focus = parser.next_focus
            original = document.text[sentence.tokens[0].start_char:sentence.tokens[-1].end_char]
            nodes.append((key, original, frames))
        if nodes and (not any(frames for _, _, frames in nodes) or
                                     not query and len(nodes) == 1 and repaired_fragment):
            alternate = self.imperative_reading(document) or self.directed_command(document)
            if alternate is not None:
                coordinated = self.imperative_list(document, alternate) if query else alternate
                repaired = coordinated.frames(previous={} if query else self._focus, node=node_id,
                                              imperative_list=coordinated is not alternate)
                alternate = coordinated
                if repaired:
                    # These gerunds have irregular predictions in the English lemmatizer.
                    first_frame = repaired[0]
                    correction = {('syncing', 'synce'): 'sync', ('downloading', 'downloading'): 'download'}
                    key = (first_frame['evidence'].lower(), first_frame['action'])
                    if key in correction:
                        first_frame['action'] = correction[key]
                    if query:
                        repaired = enrich_query(alternate, repaired, node_id)
                    if self.include_modifiers:
                        add_infinitive_modifiers(alternate, repaired)
                    nodes = [(node_id, document.text.strip(), repaired)]
                    focus = alternate.next_focus
        self._focus = focus if nodes else {}
        return nodes


def project_frames(frames):
    """Return the action/role/text representation used by the coherence experiments."""
    return [
        [frame["action"], [[arg["role"], arg["text"]] for arg in frame["arguments"]]]
        for frame in frames
    ]


def chain_frames(pipeline, episode, *, enhance_query=False):
    """Yield Q then agent steps. Opt into the qualified Q used in the query-repair trial."""
    extractor = IntentExtractor(pipeline)
    read_query = extractor.extract_query if enhance_query else extractor.extract_step
    yield from read_query(episode["task"], "Q")
    for i, step in enumerate(episode["steps"], 1):
        yield from extractor.extract_step(step["thought"], f"S{i:02d}")
