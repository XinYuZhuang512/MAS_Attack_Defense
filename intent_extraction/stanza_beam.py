"""Optional adapter for the existing KAL Stanza k-best sentence decoder."""
import importlib.util
import os
from pathlib import Path
import sys

from .probabilistic import PCFGIntentExtractor


def local_stanza(project=None):
    root = Path(project or os.environ.get("KAL2INU_HOME") or Path.home() / "KAL/kal2inu").resolve()
    source = root / "tmp/show_cached_trees.py"
    if not source.is_file():
        raise FileNotFoundError(f"Set KAL2INU_HOME to the checkout containing {source.name}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("_intent_stanza_kbest", source)
    viewer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viewer)
    return root, viewer


class StanzaBeamExtractor:
    """Parse one pre-segmented English sentence at a time; reuse the loaded model."""
    def __init__(self, k=16, min_support=0.5, project=None):
        self.reader = PCFGIntentExtractor(k=k, min_support=min_support)
        root, viewer = local_stanza(project)
        java, classpath = viewer.runtime_paths(root, None, None)
        self.parse = viewer.fresh_parse
        self.options = dict(k_best=k, engine=viewer.stanza_engine(), java=java,
                            classpath=classpath, timeout=120)

    def extract(self, text):
        return self.reader.extract_record(self.parse(text, **self.options))

    def extract_many(self, sentences):
        return [self.extract(text) for text in sentences]
