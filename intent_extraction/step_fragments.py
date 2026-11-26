"""Recover explicit step predicates from subjectless execution notes."""


def _marker(tree, word, lemma):
    return any(w.deprel in {'mark', 'case'} and w.lemma == lemma for w in tree.children[word.id])


def _verification(tree, root):
    # "Verifying the payment has been processed correctly." The following
    # proposition is the object of verification, including its polarity.
    first = tree.words[1]
    if first.xpos != 'VBG' or first.lemma not in {'verify', 'confirm', 'check', 'ensure'}:
        return []
    words = list(tree.words.values())[1:]
    if not words or any(w.text in {'?', ';', ':', '“', '”', '"'} or
                        w.lemma in {'but', 'although', 'unless'} for w in words):
        return []
    complement = any(c.deprel == 'ccomp' for c in tree.children[first.id])
    finite = any(w.xpos in {'VBD','VBZ','VBP','MD'} for w in words)
    # An incorrectly attached passive is recoverable only with an overt
    # completion/success modifier. "Verifying the payment was delayed" is ambiguous.
    passive = (root.xpos == 'VBN' and any(c.deprel == 'aux:pass' for c in tree.children[root.id])
               and any(c.deprel == 'advmod' and c.lemma in {'successfully','correctly','properly'}
                       for c in tree.children[root.id]))
    if not finite or not (first.id == root.id and complement or
                          first.head == root.id and passive and
                          any(c.deprel == 'obj' for c in tree.children[first.id])):
        return []
    if any(w.deprel in {'parataxis','punct'} and w.text not in {'.', "'", '"'}
           for w in words):
        return []
    if any(w.deprel == 'nsubj' and w.text.lower() in {'i','we'} for w in words):
        return []
    while words and words[-1].deprel == 'punct':
        words.pop()
    if not words:
        return []
    return [{'action':first.lemma, 'evidence':first.text, 'mode':'ongoing',
             'arguments':[{'role':'question', 'text':tree.argument_text([w.id for w in words])}]}]


def explicit_fragment(tree, previous, node):
    root = next(w for w in tree.words.values() if w.head == 0)
    first = tree.words[1]
    if first.lemma in {'verify','confirm','check','ensure'}:
        return _verification(tree, root)
    children = tree.children[root.id]
    action = None
    if root.id == 1 and root.text.lower() in {'need','proceeding'}:
        if any(c.deprel.startswith(('nsubj','csubj','cop')) for c in children):
            return []
        gerund = root.text.lower() == 'proceeding'
        mode = 'ongoing' if gerund else 'stated_goal'
        options = [c for c in children if c.deprel in {'xcomp','acl','advcl','obl'} and
                   c.upos == 'VERB' and (c.xpos == 'VB' and _marker(tree,c,'to') or
                                          gerund and c.xpos == 'VBG' and _marker(tree,c,'with'))]
        if len(options) == 1:
            action = options[0]
    elif root.xpos == 'VB' and _marker(tree,root,'to'):
        subject = next((c for c in children if c.deprel.startswith('nsubj') and c.lemma == 'step'),None)
        if (subject and any(c.deprel == 'amod' and c.lemma == 'next' for c in tree.children[subject.id])
                and any(c.deprel == 'cop' and c.xpos in {'VBZ','VBP'} for c in children)):
            action = root
        mode = 'stated_plan'
    if action is None:
        return []
    if any(w.text in {'?', '"', '“','”'} or w.lemma in {'if','unless','not','never','or','but'}
           for w in tree.words.values()):
        return []
    if any(c.deprel.startswith(('nsubj','csubj')) and c.lemma != 'step' for c in tree.children[action.id]):
        return []
    if any(c.deprel in {'aux','advcl','parataxis'} and c.id < action.id and c != action for c in children):
        return []
    if any(c.deprel == 'nmod:poss' and c.text.lower() not in {'my','our'}
           for w in children if w.lemma == 'step' for c in tree.children[w.id]):
        return []
    def actions(w):
        yield w
        for c in tree.children[w.id]:
            if (c.deprel == 'conj' and c.upos == 'VERB' and c.xpos in {'VB','VBG'} and
                    not any(k.deprel.startswith(('nsubj','csubj')) for k in tree.children[c.id])):
                yield from actions(c)
    frames = []
    emitted = {}
    for w in sorted(actions(action),key=lambda w:w.id):
        args = tree.arguments(w)
        tree.bind_nominals(w, root, args, emitted, previous, node)
        frames.append({'action':w.lemma,'evidence':w.text,'mode':mode,'arguments':args})
        emitted[w.id] = root
    if len(frames) == 1 and tree.simple_context(action,root):
        tree.next_focus = tree.nominal_focus(action,len(tree.words)+1,node)
    return frames
