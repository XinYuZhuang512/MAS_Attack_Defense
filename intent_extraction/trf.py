"""Project spaCy's English transformer dependencies into sentence-level frames."""

MODEL = "en_core_web_trf"

ARGUMENTS = {"dobj", "iobj", "dative", "attr", "oprd", "ccomp", "xcomp", "nsubjpass", "prt"}
MODIFIERS = {"poss", "amod", "det", "nummod", "compound", "appos", "acl", "relcl",
             "conj", "quantmod", "advmod"}
ARGUMENT_ROLES = {"dobj": "obj", "nsubjpass": "nsubj:pass", "dative": "iobj"}
SCOPE_CUES = {"nsubj", "csubj", "nsubjpass", "csubjpass", "aux", "auxpass",
              "cop", "mark", "neg", "advmod", "cc"}


def _span(token):
    return token.doc[token.left_edge.i:token.right_edge.i + 1].text


def _argument(head, role, offset, preposition=None):
    source = preposition if preposition is not None else head
    result = {"token": head.i - offset, "head": head.text, "relation": role, "text": _span(source)}
    if preposition is not None:
        result["preposition"] = preposition.lower_
    modifiers = []
    for child in head.children:
        if child.dep_ == "prep":
            modifiers.extend(_argument(obj, "nmod", offset, child)
                             for obj in child.children if obj.dep_ == "pobj")
        elif child.dep_ in MODIFIERS:
            modifiers.append(_argument(child, child.dep_, offset))
    if modifiers:
        result["modifiers"] = modifiers
    return result


def _arguments(verb, offset):
    for child in verb.children:
        if child.dep_ == "dative" and child.pos_ == "ADP":
            objects = [obj for obj in child.children if obj.dep_ == "pobj"]
            if objects:
                for obj in objects:
                    yield _argument(obj, "obl", offset, child)
                continue
        if child.dep_ in ARGUMENTS:
            yield _argument(child, ARGUMENT_ROLES.get(child.dep_, child.dep_), offset)
        elif child.dep_ in {"prep", "agent"}:
            for obj in child.children:
                if obj.dep_ in {"pobj", "pcomp"}:
                    yield _argument(obj, "obl", offset, child)


def _sentence_result(sentence):
    frames = []
    for verb in sentence:
        if verb.pos_ != "VERB" or verb.dep_ in {"aux", "auxpass", "amod", "compound"}:
            continue
        scope = [
            {"token": word.i - sentence.start, "word": word.text, "relation": word.dep_.lower(),
             "cues": [{"relation": c.dep_, "text": _span(c)}
                      for c in word.children if c.dep_ in SCOPE_CUES]}
            for word in (verb, *verb.ancestors)
        ]
        frames.append({"token": verb.i - sentence.start, "action": verb.lemma_, "surface": verb.text,
                       "arguments": list(_arguments(verb, sentence.start)), "scope": scope})
    return {"sentence": sentence.text, "frames": frames}


class TrfIntentExtractor:
    """Reuse one en_core_web_trf pipeline across queries and execution steps."""

    def __init__(self):
        import spacy
        self.pipeline = spacy.load(MODEL, exclude=["ner"])

    def extract(self, text):
        if not isinstance(text, str):
            raise TypeError("Expected an English text string")
        if not text.strip():
            return []
        return [_sentence_result(sentence) for sentence in self.pipeline(text).sents
                if sentence.text.strip()]

    def extract_many(self, texts):
        return [self.extract(text) for text in texts]
