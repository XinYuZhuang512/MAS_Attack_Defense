"""Project a constituency tree and its aligned basic dependencies into action frames."""
from collections import defaultdict


ARGUMENTS = {"obj", "iobj", "obl", "xcomp", "ccomp"}
MODIFIERS = {"det", "amod", "nummod", "compound", "nmod", "acl", "appos",
             "conj", "flat", "fixed", "advmod", "quantmod"}
SCOPE = {"nsubj", "csubj", "aux", "cop", "mark", "neg", "advmod", "cc"}


class Reading:
    def __init__(self, tokens, tree):
        self.tokens = tokens
        self.tags = tree["tags"]
        self.children = defaultdict(list)
        self.parent = {}
        for head, rel, child in tree["edges"]:
            self.children[head].append((child, rel))
            self.parent[child] = head, rel
        self.phrases = defaultdict(list)
        for phrase, start, stop in tree["phrases"]:
            self.phrases[start, stop].append(phrase)

    def text(self, ids):
        chunks = []
        last = None
        for i in sorted(ids):
            word, before = self.tokens[i]
            if last is not None:
                chunks.append(before if i == last + 1 else " … ")
            chunks.append(word)
            last = i
        return "".join(chunks)

    def subtree(self, head):
        yield head
        for child, _ in self.children[head]:
            yield from self.subtree(child)

    def phrase(self, head, ids):
        # Select the largest CT constituent headed here within the UD projection.
        # For PP arguments prefer the nominal inside the case marker.
        tags = {"NP","NML","NX","WHNP","QP"} if self.tags[head][0].startswith(
            ("NN","PRP","WP","CD")) else {"VP","S","SBAR","ADJP","ADVP","PP"}
        spans = [(b-a, a, b, tag) for (a,b), labels in self.phrases.items()
                 if a <= head < b and all(i in ids for i in range(a,b))
                 for tag in labels if tag in tags]
        if not spans:
            return None
        _, a, b, tag = max(spans)
        return {"label": tag, "text": self.text(range(a,b))}

    def branch(self, head, rel, path, facts):
        markers = sorted(i for i, r in self.children[head] if r == "case")
        marker_ids = {j for i in markers for j in self.subtree(i)}
        key = ()
        if facts is not None:
            key = path + ((rel, head, tuple(markers)),)
            facts.add(key)
        ids = set(self.subtree(head))
        item = {"token": head, "head": self.tokens[head][0], "relation": rel,
                "text": self.text(ids)}
        if facts is not None:
            item["_key"] = key
        if markers:
            item["preposition"] = self.text(marker_ids)
        phrase = self.phrase(head, ids - marker_ids)
        if phrase:
            item["constituent"] = phrase
        nested = [self.branch(i, r, key, facts) for i,r in self.children[head]
                  if r.split(":")[0] in MODIFIERS]
        if nested:
            item["modifiers"] = nested
        return item

    def coordinated_predicate(self, i, head, rel):
        # A nominal tag can conflict with both a complete VP and its own
        # direct object. Require parallelism with an uninflected verbal head.
        if rel != "conj" or head < 0 or self.tags[head][0] != "VB":
            return False
        objects = [j for j, role in self.children[i] if role in {"obj","iobj"}]
        return any(start == i and "VP" in labels and any(i < j < stop for j in objects)
                   for (start, stop), labels in self.phrases.items())

    def frames(self, *, record_paths=True):
        for i,(tag,lemma) in enumerate(self.tags):
            head, rel = self.parent[i]
            if rel.split(":")[0] in {"aux","cop","amod","compound"}:
                continue
            if not tag.startswith("VB"):
                if not tag.startswith("NN") or not self.coordinated_predicate(i, head, rel):
                    continue
                lemma = self.tokens[i][0].casefold()
            action = (i, lemma)
            facts = set() if record_paths else None
            args = []
            for j, role in self.children[i]:
                if role.split(":")[0] in ARGUMENTS or role in {"nsubj:pass","csubj:pass","compound:prt"}:
                    args.append(self.branch(j, role, (), facts))
            # Keep the grammatical governors and their local scope markers.
            # This also exposes a predicate inside a report, condition or modifier.
            scope = []
            cursor = i
            while cursor >= 0:
                governor, link = self.parent[cursor]
                scope.append({"token": cursor, "word": self.tokens[cursor][0], "relation": link,
                              "cues": [{"relation":r, "text":self.text(self.subtree(j))}
                                       for j,r in self.children[cursor] if r.split(":")[0] in SCOPE]})
                cursor = governor
            yield action, facts, {"token": i, "action": lemma, "surface": self.tokens[i][0],
                                  "arguments": args, "scope": scope}
