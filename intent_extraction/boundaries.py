"""Reparse a sentence boundary that Stanza put inside a file name or a possessive personal name."""
import re

from stanza.models.common.doc import Document

# A suffix alone is ambiguous. Require a zero-space sentence break inside
# the name, and complete token boundaries around the name, before merging.
FILE_NAME = re.compile(
    r"[\w-]+(?:\.[\w-]+)*\.(?:py|js|jsx|ts|tsx|java|cpp|html|css|json|yaml|yml|"
    r"toml|xml|sql|sh|md|txt|csv|tsv|log|pdf|docx?|xlsx?|pptx?|jpg|jpeg|png|svg|zip|gz)"
    r"(?![\w-])", re.I)


def repaired_sentences(document, pipeline, *, join_possessives=True):
    sentences = document.sentences
    if len(sentences) < 2:
        return sentences
    cuts = {left.tokens[-1].end_char: i for i, (left, right) in
            enumerate(zip(sentences, sentences[1:])) if left.tokens[-1].end_char == right.tokens[0].start_char}
    joins = set()
    for i, (left, right) in enumerate(zip(sentences, sentences[1:])):
        # "Reading Kevin | White's email": respect original punctuation/newlines,
        # and require the surname to actually be parsed as a possessive modifier.
        first = right.words[0]
        if (join_possessives and len(right.words) > 1 and left.words[-1].xpos == first.xpos == 'NNP' and
                first.deprel == 'nmod:poss' and right.words[1].xpos == 'POS' and
                right.words[1].head == first.id and
                document.text[left.tokens[-1].end_char:right.tokens[0].start_char] == ' '):
            joins.add(i)
    if not cuts and not joins:
        return sentences
    starts = {t.start_char for s in sentences for t in s.tokens}
    ends = {t.end_char for s in sentences for t in s.tokens}
    names = {}
    for match in FILE_NAME.finditer(document.text):
        a, b = match.span()
        internal = [i for offset, i in cuts.items() if a < offset < b]
        if internal and a in starts and b in ends:
            names[a] = b
            joins.update(internal)
    if not joins:
        return sentences
    result = []
    i = 0
    while i < len(sentences):
        j = i
        while j in joins:
            j += 1
        if i == j:
            result.append(sentences[i])
        else:
            rows = []
            skip_to = 0
            word_id = 1
            for sentence in sentences[i:j + 1]:
                for token in sentence.tokens:
                    if token.start_char < skip_to:
                        continue
                    if token.start_char in names:
                        skip_to = names[token.start_char]
                        rows.append(dict(id=word_id, text=document.text[token.start_char:skip_to],
                                         start_char=token.start_char, end_char=skip_to))
                        word_id += 1
                        continue
                    # Keep the original multi-word expansions (e.g. "I'll") and offsets.
                    for row in token.to_dict():
                        row = dict(row)
                        row.pop('head', None)
                        row.pop('deprel', None)
                        row.pop('deps', None)
                        if isinstance(row['id'], (list, tuple)) and len(row['id']) == 2:
                            row['id'] = (word_id, word_id + len(token.words) - 1)
                        else:
                            row['id'] = word_id
                            word_id += 1
                        rows.append(row)
            doc = Document([rows], text=document.text)
            doc = pipeline(doc, processors=['pos', 'lemma', 'depparse', 'constituency'])
            result.extend(doc.sentences)
        i = j + 1
    return result


def filename_sentences(document, pipeline):
    return repaired_sentences(document, pipeline, join_possessives=False)
