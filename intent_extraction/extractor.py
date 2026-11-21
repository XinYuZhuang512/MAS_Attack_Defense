"""Extract action frames and bind entity arguments using only the observed trajectory prefix."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import stanza


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
                if prep == 'for' and w.lemma not in {'ask','search','look','pay','apply',
                                                       'check','wait','file','register'}:
                    continue
                role = prep
            elif rel in {'xcomp','ccomp'} and w.lemma in REPORT | {'notify'}:
                role, excluded = 'requested_content', {'advcl','parataxis'}
            elif rel in {'advcl','ccomp'} and prep in {'if','whether','that'} and w.lemma in QUERY:
                role = 'question'
            elif rel == 'advcl' and prep in {'if','unless','when'}:
                role = 'condition'
            elif rel == 'advcl' and prep == 'as' and w.lemma == 'mark':
                role = 'result_state'
            elif rel == 'compound' and c.deprel == 'compound:prt':
                role = 'particle'
            if role:
                ids = self.subtree(c, excluded)
                if role == 'obj':
                    recipient = self.recipient_ids(w, c)
                    if recipient:
                        args.append({'role':'to','text':self.excerpt(recipient)})
                        ids = sorted(set(ids) - set(recipient))
                # Some nested nodes can be non-projective. Emit the selected runs individually.
                runs = []
                for i in sorted(ids):
                    if not runs or runs[-1][-1] + 1 != i:
                        runs.append([])
                    runs[-1].append(i)
                args.append({'role':role,'text':' … '.join(self.excerpt(run) for run in runs)})
        if self.refine:
            for arg in args:
                # Role encodes the preposition. Give explicit and pronominal
                # destinations the same entity representation.
                if arg['role'] in {'to', 'into'}:
                    prep, space, value = arg['text'].partition(' ')
                    if space and prep.lower() == arg['role']:
                        arg['surface'] = arg['text']
                        arg['text'] = value
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

    def simple_phrase(self, head):
        """Exact, contiguous noun phrase, with no competing embedded entity or action."""
        ids = sorted(self.subtree(head))
        if ids != list(range(ids[0], ids[-1] + 1)):
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

    def bind_nominals(self, action, controller, args, emitted, previous, node):
        governor = controller if controller is not None else action
        bound = {}
        for arg in args:
            role = arg['role']
            if role not in {'obj', 'to', 'into'}:
                continue
            tokens = arg['text'].lower().split()
            if role != 'obj' and tokens[:1] == [role]:
                tokens = tokens[1:]
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
            elif previous and self.simple_context(action, controller):
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

    def comparison_topic(self, action):
        # A reported proposition can supply a topic for a bare "compare". Keep
        # it as attributed evidence, not as an observed destination or tool call.
        if action.deprel not in {'parataxis','conj'}:
            return None
        if any(c.deprel.split(':')[0] in {'obj','iobj','obl','ccomp','xcomp','advcl','orphan'}
               for c in self.children[action.id]):
            return None
        report = self.words.get(action.head)
        if report is None or report.deprel != 'root' or report.lemma not in REPORT or report.id >= action.id:
            return None
        earlier = [w for w in self.children[report.id] if w.id < action.id]
        clauses = [w for w in earlier if w.deprel == 'ccomp']
        speakers = [w for w in earlier if w.deprel == 'nsubj']
        if len(clauses) != 1 or len(speakers) != 1 or any(
                w.deprel in {'conj','parataxis','advcl'} or w.lemma in {'not','never'} for w in earlier):
            return None
        clause = clauses[0]
        ids = sorted(self.subtree(clause))
        if ids[-1] >= action.id or ids != list(range(ids[0], ids[-1] + 1)):
            return None
        words = [self.words[i] for i in ids]
        if (any(w.deprel.split(':')[0] in CLAUSAL | {'conj'} or w.lemma in {'not','never'}
                for w in words if w.id != clause.id) or
                not any(w.xpos in {'JJR','RBR','JJS','RBS'} for w in words)):
            return None
        speaker = self.simple_phrase(speakers[0])
        if not speaker:
            return None
        topic = {'source': speaker, 'evidence': self.excerpt(ids), 'inferred': True}
        for dep, field in [('nsubj', 'claim_subject'), ('obj', 'claim_object')]:
            heads = [c for c in self.children[clause.id] if c.deprel == dep]
            if len(heads) == 1:
                phrase = self.simple_phrase(heads[0])
                if phrase:
                    topic[field] = phrase
        return topic

    def frames(self, previous=None, node=None):
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
            allowed = w.upos == 'VERB' and not locally_negated and not self.in_subordinate_clause(w,controller)
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
            is_control = w.lemma in CONTROL or (w.lemma == 'let' and any(
                x.text.lower() in {'me','us'} and x.deprel in {'obj','iobj'} for x in children))
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
                        topic = self.comparison_topic(w)
                        if topic:
                            frame['inferred_topic'] = topic
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
        walk(next(w for w in self.words.values() if w.head==0))
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

    def __init__(self, pipeline=None):
        if pipeline is None:
            pipeline = stanza.Pipeline('en', processors=','.join(MODELS), package=MODELS,
                                       download_method=None, use_gpu=False, verbose=False)
        self.pipeline = pipeline
        self._focus = {}

    def extract_step(self, text, node_id):
        """Return (node ID, original sentence, frames) records for one incoming step."""
        return self.extract_document(self.pipeline(text), node_id)

    def extract_document(self, document, node_id):
        """Use the same extractor when a caller already has a parsed Stanza document."""
        nodes = []
        focus = self._focus
        for i, sentence in enumerate(document.sentences, 1):
            key = node_id if len(document.sentences) == 1 else f'{node_id}.{i}'
            parser = Sentence(sentence, document.text, joint=True, refine=True)
            frames = parser.frames(previous=focus, node=key)
            focus = parser.next_focus
            original = document.text[sentence.tokens[0].start_char:sentence.tokens[-1].end_char]
            nodes.append((key, original, frames))
        self._focus = focus if nodes else {}
        return nodes


def chain_frames(pipeline, episode):
    """Yield nodes for one AgentDrift episode using the normal incremental extractor."""
    extractor = IntentExtractor(pipeline)
    steps = ((f'S{i:02d}', step['thought']) for i, step in enumerate(episode['steps'], 1))
    for key, text in chain([('Q', episode['task'])], steps):
        yield from extractor.extract_step(text, key)
