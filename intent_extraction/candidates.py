"""Project scored constituency/Basic UD pairs onto the same action frames as a step."""
from collections import defaultdict
import json
import math

from stanza.models.common.doc import Document
from stanza.models.constituency.parse_tree import Tree

from .extractor import Sentence, project_frames
from .syntax_details import add_infinitive_modifiers


def penn_upos(tag, relation):
    if tag in {'ADD', 'NNP', 'NNPS'}:
        return 'PROPN'
    if tag.startswith('NN'):
        return 'NOUN'
    if tag in {'PRP', 'PRP$', 'WP', 'WP$'}:
        return 'PRON'
    if tag == 'MD' or relation.split(':')[0] in {'aux', 'cop'}:
        return 'AUX'
    if tag.startswith('VB'):
        return 'VERB'
    if tag.startswith('JJ'):
        return 'ADJ'
    if tag.startswith('RB') or tag == 'WRB':
        return 'ADV'
    if tag in {'DT', 'WDT', 'PDT'}:
        return 'DET'
    return {'IN': 'ADP', 'TO': 'PART', 'RP': 'PART', 'POS': 'PART', 'CC': 'CCONJ',
            'CD': 'NUM', 'EX': 'PRON', 'UH': 'INTJ'}.get(tag, 'PUNCT' if not tag.isalnum() else 'X')


def stanza_tree(node):
    if isinstance(node, str):
        return Tree(node)
    return Tree(node['label'], [stanza_tree(child) for child in node['children']])


def weighted_frames(pipeline, text, node_id, candidates, previous, include_modifiers):
    candidates = sorted(candidates, key=lambda t: t['score'], reverse=True)
    if not candidates or not all(math.isfinite(t['score']) for t in candidates):
        raise ValueError('Expected one or more trees with finite log scores')
    mass = defaultdict(float)
    representatives = {}
    lemmas = {}
    for item in candidates:
        edges = {int(child): (int(head), rel) for head, rel, child in item['dependencies']}
        words = []
        for i, ((a, b), tag) in enumerate(zip(item['spans'], item['pos'], strict=True), 1):
            head, relation = edges[i]
            words.append(dict(id=i, text=text[a:b], xpos=tag, upos=penn_upos(tag, relation),
                              start_char=a, end_char=b, head=head, deprel=relation))
        doc = Document([words], text=text)
        sentence = doc.sentences[0]
        tags = tuple((w.text, w.xpos, w.upos) for w in sentence.words)
        if tags not in lemmas:
            pipeline.processors['lemma'].process(doc)
            lemmas[tags] = [w.lemma for w in sentence.words]
        else:
            for word, lemma in zip(sentence.words, lemmas[tags], strict=True):
                word.lemma = lemma
        sentence.constituency = stanza_tree(item['node'])
        parser = Sentence(sentence, text, joint=True, refine=True)
        frames = parser.frames(previous=previous, node=node_id)
        if include_modifiers:
            add_infinitive_modifiers(parser, frames)
        # Vote on the representation consumed downstream. Keep scope/mode distinct.
        key = json.dumps([project_frames(frames), [f['mode'] for f in frames]], ensure_ascii=False)
        mass[key] += math.exp(item['score'] - candidates[0]['score'])
        if key not in representatives:
            representatives[key] = (frames, parser.next_focus)
    winner = max(mass, key=mass.get)
    share = mass[winner] / sum(mass.values())
    frames, focus = representatives[winner] if share > 0.5 else ([], {})
    return frames, focus, dict(tree_count=len(candidates), winning_mass=share,
                               has_majority=share > 0.5)


def viewer_candidates(record):
    for tree, graph in zip(record['syntax']['trees'], record['dependencies'], strict=True):
        yield dict(score=tree['parse_score'], spans=tree['spans'], pos=tree['pos'],
                   node=tree['node'], dependencies=graph['Basic']['edges'])
