"""One greedy constituency parse and its basic dependency tree per sentence."""
import json
import subprocess
import tempfile
from threading import Lock
from weakref import finalize

from .corenlp import java_command
from .reading import Reading
from .stanza_beam import local_stanza


def _shutdown(worker, log):
    try:
        try:
            worker.stdin.close()
        except BrokenPipeError:
            pass
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    finally:
        worker.stdout.close()
        log.close()


class SingleTreeIntentExtractor:
    """Reuse one local parser for a stream of raw English inputs."""
    def __init__(self, *, parser="stanza", corenlp_home=None, java=None, project=None):
        if parser not in {"stanza","shift-reduce"}:
            raise ValueError("Expected stanza or shift-reduce")
        self.parser = parser
        self.command = java_command("--syntax" if parser == "stanza" else "--single", corenlp_home, java)
        self.project = project
        self._model = None
        self._lock = Lock()
        self._worker = None
        self._cleanup = None

    def _start(self):
        log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        try:
            worker = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=log, text=True, encoding="utf-8")
        except OSError:
            log.close()
            raise
        self._worker = worker
        self._log = log
        self._cleanup = finalize(self, _shutdown, worker, log)

    def _stop(self):
        if self._cleanup is not None:
            self._cleanup()
            self._cleanup = None
            self._worker = None

    def close(self):
        with self._lock:
            self._stop()
            self._model = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _request(self, payload):
        if self._worker is None:
            self._start()
        try:
            self._worker.stdin.write(json.dumps(payload) + "\n")
            self._worker.stdin.flush()
            response = self._worker.stdout.readline()
            if not response:
                raise EOFError("CoreNLP worker exited")
        except (OSError, EOFError) as exc:
            self._log.seek(0)
            message = self._log.read()[-2000:]
            self._stop()
            raise RuntimeError(f"Single-tree parser failed. {message}") from exc
        return json.loads(response)

    def _parse(self, text):
        sentences = self._request({"text": text})
        if self.parser == "shift-reduce" or not sentences:
            return sentences
        if self._model is None:
            _, viewer = local_stanza(self.project)
            _, self._model, _ = viewer.stanza_engine()
        def encode(node):
            return [node.label, *(encode(c) for c in node.children)]
        for sentence in sentences:
            tagged = sentence.pop("tagged")
            tree, = self._model.parse_tagged_words([tagged], batch_size=1)
            if list(tree.leaf_labels()) != [w for w, tag in tagged]:
                raise ValueError("Stanza constituency leaves and CoreNLP tokens differ")
            sentence["tree"] = self._request({"tree": encode(tree), "size": len(tagged)})
        return sentences

    def extract(self, text):
        if not isinstance(text, str):
            raise TypeError("Expected an English text string")
        if not text.strip():
            return []
        with self._lock:
            sentences = self._parse(text)
        result = []
        for sentence in sentences:
            reading = Reading(sentence["tokens"], sentence["tree"])
            frames = [frame for _, _, frame in reading.frames(record_paths=False)]
            result.append({"sentence": sentence["sentence"], "frames": frames})
        return result

    def extract_many(self, texts):
        return [self.extract(text) for text in texts]
