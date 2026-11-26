"""Extract predicate, argument and modifier attachments from scored English parses.

All token IDs in the result are zero based within the sentence. Supports are
relative to the supplied derivations. An action is a syntactic predicate
candidate; scope is retained for downstream interpretation.
"""
from collections import Counter, defaultdict
import json
import math
import subprocess

from .corenlp import java_command
from .reading import Reading


def aggregate(sentence, min_support=0.5):
    """Marginalize over derivations, returning a coherent attachment set per predicate."""
    if not 0 <= min_support <= 1:
        raise ValueError("min_support must be between 0 and 1")
    trees = sentence["trees"]
    if not trees or not all(math.isfinite(t["score"]) for t in trees):
        raise ValueError("Expected at least one finite scored parse")
    peak = max(t["score"] for t in trees)
    weights = [math.exp(t["score"] - peak) for t in trees]
    total = math.fsum(weights)
    actions = Counter()
    edges = defaultdict(Counter)
    readings = defaultdict(list)
    for rank,(tree, weight) in enumerate(zip(trees, weights), 1):
        mass = weight / total
        for action, facts, frame in Reading(sentence["tokens"], tree).frames():
            actions[action] += mass
            readings[action].append((facts, mass, rank, frame))
            for edge in facts:
                edges[action][edge] += mass

    def attach_support(arg, counts):
        arg["support"] = min(1.0, counts[arg.pop("_key")])
        for modifier in arg.get("modifiers", []):
            attach_support(modifier, counts)

    accepted, alternatives = [], []
    for action,p in sorted(actions.items()):
        counts = edges[action]
        # Minimum expected number of missing + spurious attachments, conditional
        # on this action. Restrict selection to one jointly observed reading.
        def agreement(item):
            facts, mass, rank, _ = item
            return math.fsum(2 * counts[key] - p for key in sorted(facts)), mass, -rank
        chosen, _, rank, frame = max(readings[action], key=agreement)
        for arg in frame["arguments"]:
            attach_support(arg, counts)
        frame.update(support=min(1.0, p), tree_rank=rank)
        frame["alternative_attachments"] = [
            {"path": [{"relation": r, "token": i, "head": sentence["tokens"][i][0],
                       "case_tokens": list(markers)} for r,i,markers in key], "support": min(1.0,v)}
            for key,v in sorted(counts.items()) if key not in chosen]
        (accepted if p >= min_support - 1e-12 else alternatives).append(frame)
    return {"sentence": sentence["sentence"], "derivations": len(trees),
            "frames": accepted, "lower_support_frames": alternatives}


class PCFGIntentExtractor:
    """Batch sentences in one local JVM. Requires a JDK 11+ and CoreNLP English PCFG."""
    def __init__(self, k=16, min_support=0.5, corenlp_home=None, java=None):
        if type(k) is not int or k < 1 or not 0 <= min_support <= 1:
            raise ValueError("Expected k >= 1 and 0 <= min_support <= 1")
        self.command = java_command(str(k), corenlp_home, java)
        self.min_support = min_support

    def extract_many(self, texts):
        texts = list(texts)
        if not texts:
            return []
        payload = "".join(json.dumps({"text": t}) + "\n" for t in texts)
        proc = subprocess.run(self.command, input=payload, text=True, encoding="utf-8", capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        rows = proc.stdout.splitlines()
        if len(rows) != len(texts):
            raise RuntimeError("Missing parser results")
        return [[aggregate(s, self.min_support) for s in json.loads(row)] for row in rows]

    def extract(self, text):
        return self.extract_many([text])[0]

    def extract_record(self, record):
        """Use aligned scored CT/Basic-UD pairs exported by show_cached_trees.py."""
        text = record["text"]
        trees = record["syntax"]["trees"]
        graphs = record["dependencies"]
        if not trees or len(trees) != len(graphs):
            raise ValueError("Missing aligned tree/graph pairs")
        spans = trees[0]["spans"]
        tokens = []
        previous = 0
        for start, end in spans:
            if not previous <= start < end <= len(text):
                raise ValueError("Invalid token offsets")
            tokens.append([text[start:end], text[previous:start]])
            previous = end
        parses = []
        pairs = {}
        for tree, graph in zip(trees, graphs):
            if tree["spans"] != spans or len(tree["pos"]) != len(tokens):
                raise ValueError("Candidates must use the same token alignment")
            constituents = []
            leaves = []

            def walk(node):
                start = len(leaves)
                children = node["children"]
                if len(children) == 1 and isinstance(children[0], str):
                    leaves.append(node["label"])
                else:
                    for child in children:
                        walk(child)
                    constituents.append((node["label"], start, len(leaves)))

            walk(tree["node"])
            if leaves != tree["pos"]:
                raise ValueError("CT leaves and POS tags differ")
            tagged = [(word,tag) for (word,_),tag in zip(tokens, leaves)]
            pairs.update(dict.fromkeys(tagged))
            parses.append({"score":tree["parse_score"], "phrases":constituents,
                           "tagged":tagged, "edges":[(int(h)-1,r,int(j)-1) for h,r,j in graph["Basic"]["edges"]]})
        proc = subprocess.run(self.command[:-1] + ["--lemmas"], text=True, capture_output=True,
                              input=json.dumps({"pairs": list(pairs)}) + "\n", encoding="utf-8")
        if proc.returncode:
            raise RuntimeError(proc.stderr.strip())
        lemmas = dict(zip(pairs, json.loads(proc.stdout), strict=True))
        for tree in parses:
            tree["tags"] = [(tag,lemmas[word,tag]) for word,tag in tree.pop("tagged")]
        return [aggregate({"sentence":text, "tokens":tokens, "trees":parses}, self.min_support)]
