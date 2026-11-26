"""Prioritize a review queue with Q-to-step intent matching."""
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import unicodedata

from .trf import TrfIntentExtractor


@dataclass(frozen=True)
class Intent:
    verb: str
    object: str
    descriptions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StepReview:
    priority: int
    reason: str
    score: float | None
    query_matches: tuple[int, ...]


@lru_cache(maxsize=50_000)
def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@lru_cache(maxsize=150_000)
def similarity(left, right):
    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0.0
    return sum(SequenceMatcher(None, a, b, autojunk=False).ratio()
               for a, b in ((left, right), (right, left))) / 2


def field_role(arg):
    return arg.get("preposition") or arg["relation"]


def nested_values(arg, parent_role):
    for modifier in arg.get("modifiers", []):
        role = parent_role + "." + field_role(modifier)
        yield role, modifier["text"]
        yield from nested_values(modifier, role)


def project_intents(rows, *, detailed=False):
    """Keep one tuple per direct object and name prepositional fields by their marker."""
    unique = {}
    for row in rows:
        for frame in row["frames"]:
            args = frame["arguments"]
            verb = " ".join([frame["action"], *(a["text"] for a in args if a["relation"] == "prt")])
            fields = [(field_role(a), a["text"]) for a in args if a["relation"] not in {"obj", "prt"}]
            if detailed:
                for arg in args:
                    if arg["relation"] not in {"obj", "prt"}:
                        fields.extend(nested_values(arg, field_role(arg)))
                for depth, scope in enumerate(frame.get("scope", ())):
                    if depth:
                        fields.append((f"scope.{depth}", scope["word"]))
                    for cue in scope["cues"]:
                        if (cue["relation"] in {"neg", "nsubj", "nsubjpass"} and
                                normalize(cue["text"]) not in {"i", "we", "you"}):
                            fields.append((f"scope.{depth}.{cue['relation']}", cue["text"]))
            for obj in [a for a in args if a["relation"] == "obj"] or [None]:
                details = fields + (list(nested_values(obj, "obj")) if obj and detailed else [])
                unique[Intent(verb, obj["text"] if obj else "", tuple(details))] = None
    return tuple(unique)


def role_values(intent):
    by_role = defaultdict(list)
    if intent.object:
        by_role["obj"].append(intent.object)
    for role, text in intent.descriptions:
        by_role[role].append(text)
    return by_role


def best_alignment(expected, actual):
    """Maximize the lowest score over distinct same-role matches."""
    scores = [[similarity(source, target) for source in expected] for target in actual]
    if len(scores) == 1:
        return max(scores[0])
    for cutoff in sorted({value for row in scores for value in row}, reverse=True):
        assigned = {}

        def assign(index, visited):
            for column, score in enumerate(scores[index]):
                if score < cutoff or column in visited:
                    continue
                visited.add(column)
                if column not in assigned or assign(assigned[column], visited):
                    assigned[column] = index
                    return True
            return False

        if all(assign(index, set()) for index in range(len(scores))):
            return cutoff
    raise AssertionError("expected at least as many query values as observed values")


def subset_score(query, observed):
    score = similarity(query.verb, observed.verb)
    fields = role_values(query)
    for role, values in role_values(observed).items():
        candidates = fields[role]
        if len(candidates) < len(values):
            return 0.0
        score = min(score, best_alignment(candidates, values))
    return score


@lru_cache(maxsize=50_000)
def signature(intent):
    return (normalize(intent.verb), normalize(intent.object),
            tuple(sorted((role, normalize(text)) for role, text in intent.descriptions)))


def exact_match(query_rows, step_rows):
    query = set(map(signature, project_intents(query_rows, detailed=True)))
    for intent in project_intents(step_rows, detailed=True):
        has_argument = intent.object or any(not role.startswith("scope.") for role, _ in intent.descriptions)
        if not has_argument or signature(intent) not in query:
            return False
    return True


def prioritize(query_rows, step_rows, *, threshold=0.90):
    """Return one review entry for every step, ordered by ascending priority."""
    if not 0 <= threshold < 1:
        raise ValueError("Expected a threshold in [0, 1)")
    query, observed = project_intents(query_rows), project_intents(step_rows)
    if not query or not observed:
        return StepReview(0, "missing_intent", None, ())
    matches = [max((subset_score(candidate, current), -index)
                   for index, candidate in enumerate(query)) for current in observed]
    score = min(value for value, _ in matches)
    indexes = tuple(-index for _, index in matches)
    if score <= threshold:
        return StepReview(0, "field_mismatch", score, indexes)
    if any(not row["frames"] for row in (*query_rows, *step_rows)):
        return StepReview(0, "partial_extraction", score, indexes)
    if exact_match(query_rows, step_rows):
        return StepReview(2, "exact_match", score, indexes)
    return StepReview(1, "fields_match", score, indexes)


class StringStepScreen:
    """Parse Q once and return a review entry for each subsequent step."""

    def __init__(self, query, *, extractor=None, threshold=0.90):
        self.extractor = TrfIntentExtractor() if extractor is None else extractor
        if not 0 <= threshold < 1:
            raise ValueError("Expected a threshold in [0, 1)")
        self.threshold = threshold
        try:
            self.query = self.extractor.extract(query)
        except (RuntimeError, ValueError):
            self.query = None

    def review(self, step):
        if self.query is None:
            return StepReview(0, "query_extraction_error", None, ())
        try:
            observed = self.extractor.extract(step)
        except (RuntimeError, ValueError):
            return StepReview(0, "step_extraction_error", None, ())
        return prioritize(self.query, observed, threshold=self.threshold)
