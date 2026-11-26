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
        separated = {marker.id for marker, _ in tree.object_for_phrases(obj)}
        for child in tree.children[obj.id]:
            if child.deprel.split(":")[0] != "nmod":
                continue
            marker = next((w for w in tree.children[child.id] if w.deprel == "case"), None)
            if marker and marker.id not in separated:
                ids = sorted(tree.subtree(child, {"acl", "advcl", "ccomp", "conj"}))
                if ids[-1] - ids[0] + 1 == len(ids):
                    yield tree.prepositional_argument(marker, ids, prefix="obj.")


def _copular_queries(tree):
    """Normalize direct 'what is/are NP' questions, including coordinated questions."""
    root = next(w for w in tree.words.values() if w.head == 0)
    # UD can make either "what" or its nominal answer slot the predicate.
    first = tree.words[1]
    inverse = (root.upos in {"NOUN", "PROPN"} and first.lemma.lower() == "what" and
               first.head == root.id and first.deprel == "nsubj" and
               any(w.deprel == "cop" and w.id == first.id + 1 for w in tree.children[root.id]))
    primary = first if inverse else root
    if primary.lemma.lower() != "what":
        return
    for wh in tree.words.values():
        if wh.id != primary.id and not (wh.head == primary.id and wh.deprel == "conj" and wh.lemma.lower() == "what"):
            continue
        children = tree.children[wh.id]
        if inverse and wh.id == first.id:
            subject = root
        else:
            subjects = [w for w in children if w.deprel == "nsubj" and w.upos in {"NOUN", "PROPN"}]
            if len(subjects) != 1:
                continue
            subject = subjects[0]
        # In a coordinated question Stanza may attach the copula to the nominal subject.
        clause = children + tree.children[subject.id]
        if not any(w.deprel == "cop" and w.lemma == "be" and w.xpos in {"VBP", "VBZ"} for w in clause):
            continue
        if any(w.deprel in {"aux", "advcl"} or w.lemma in {"not", "never"} for w in clause):
            continue
        ids = set(tree.subtree(subject, {"cop", "punct"}))
        if inverse and wh.id == first.id:
            ids.difference_update(tree.subtree(wh))
        qualifiers = []
        for child in tree.children[subject.id]:
            if child.deprel == "nmod":
                marker = next((w for w in tree.children[child.id] if w.deprel == "case"), None)
                if marker is not None:
                    phrase = tree.subtree(child)
                    qualifiers.append(tree.prepositional_argument(marker, phrase, prefix="obj."))
                    ids.difference_update(phrase)
        yield ({"action": "query", "mode": "requested", "evidence": wh.text,
                "arguments": [{"role": "obj", "text": tree.argument_text(ids)}, *qualifiers]}, wh)


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
    questions = list(_copular_queries(tree))
    if questions:
        tree.next_focus = {}
        anchored.extend(questions)
        anchored.sort(key=lambda item: item[1].id if item[1] is not None else len(tree.words) + 1)
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
