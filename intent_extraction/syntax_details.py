"""Query-specific predicates and optional grammatical metadata for action frames."""

WISH_VERBS = {"want", "need", "like", "wish"}


def _anchor(tree, frame):
    matches = [
        word
        for word in tree.words.values()
        if word.lemma == frame["action"] and word.text == frame["evidence"]
    ]
    return matches[0] if len(matches) == 1 else None


def _requested_predicates(tree):
    selected = {}

    def positive(word):
        return not any(
            child.lemma in {"not", "never"} for child in tree.children[word.id]
        )

    def collect(word):
        if word.upos != "VERB" or not positive(word):
            return
        for child in tree.children[word.id]:
            if word.lemma in WISH_VERBS and child.deprel == "xcomp":
                collect(child)
            elif child.deprel == "conj" and not any(
                term.deprel.startswith("nsubj") for term in tree.children[child.id]
            ):
                collect(child)
        if word.lemma not in WISH_VERBS:
            selected[word.id] = word

    for word in tree.words.values():
        if word.deprel not in {"root", "parataxis", "conj"}:
            continue
        children = tree.children[word.id]
        subjects = {c.text.lower() for c in children if c.deprel.startswith("nsubj")}
        auxiliaries = {c.lemma for c in children if c.deprel == "aux"}
        if ("you" in subjects and auxiliaries & {"can", "could", "will", "would"}) or (
            subjects & {"i", "we"} and word.lemma in WISH_VERBS
        ):
            collect(word)
    return selected


def _object_qualifiers(tree, verb):
    for obj in tree.children[verb.id]:
        if obj.deprel != "obj":
            continue
        for child in tree.children[obj.id]:
            if child.deprel.split(":")[0] != "nmod":
                continue
            case = next(
                (w.lemma for w in tree.children[child.id] if w.deprel == "case"), None
            )
            if case:
                ids = sorted(tree.subtree(child, {"acl", "advcl", "ccomp", "conj"}))
                if ids[-1] - ids[0] + 1 == len(ids):
                    yield {"role": "obj." + case, "text": tree.excerpt(ids)}


def enrich_query(tree, original, node):
    """Keep rich frame metadata while projecting the qualified query from the pilot."""
    selected = _requested_predicates(tree)
    anchored = [
        (frame, _anchor(tree, frame))
        for frame in original
        if not (selected and frame["action"] in WISH_VERBS and not frame["arguments"])
    ]
    for word_id, word in sorted(selected.items()):
        if any(
            (
                anchor.id == word_id
                if anchor is not None
                else frame["action"] == word.lemma
            )
            for frame, anchor in anchored
        ):
            continue
        frame = {
            "action": word.lemma,
            "mode": "requested",
            "evidence": word.text,
            "arguments": tree.arguments(word),
        }
        anchored.append((frame, word))

    if len(anchored) != 1:
        tree.next_focus = {}
    elif anchored[0][0]["mode"] == "requested":
        word = anchored[0][1]
        controller = tree.words.get(word.head) if word.deprel == "xcomp" else None
        if tree.simple_context(word, controller):
            tree.next_focus = tree.nominal_focus(word, len(tree.words) + 1, node)

    for frame, word in anchored:
        if word is not None:
            for qualifier in _object_qualifiers(tree, word):
                if qualifier not in frame["arguments"]:
                    frame["arguments"].append(qualifier)
    return [frame for frame, _ in anchored]


def add_infinitive_modifiers(tree, frames):
    """Attach contiguous infinitives to their governor without creating action nodes."""
    for frame in frames:
        anchor = _anchor(tree, frame)
        if anchor is None:
            continue
        heads = [(anchor, "advcl", "infinitival_action_modifier")]
        heads.extend(
            (w, "acl", "infinitival_object_modifier")
            for w in tree.children[anchor.id]
            if w.deprel == "obj"
        )
        modifiers = []
        for head, relation, role in heads:
            for word in tree.children[head.id]:
                if word.upos != "VERB" or word.deprel != relation:
                    continue
                if not any(
                    c.text.lower() == "to" and c.deprel == "mark"
                    for c in tree.children[word.id]
                ):
                    continue
                ids = sorted(tree.subtree(word))
                if ids[-1] - ids[0] + 1 == len(ids):
                    modifiers.append(
                        (
                            word.id,
                            {
                                "role": role,
                                "head": head.text,
                                "text": tree.excerpt(ids),
                            },
                        )
                    )
        if modifiers:
            frame["modifiers"] = [
                item for _, item in sorted(modifiers, key=lambda pair: pair[0])
            ]
