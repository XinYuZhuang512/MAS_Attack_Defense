"""Local CoreNLP source-file launcher."""
import os
from pathlib import Path
import shutil


def java_command(mode, corenlp_home=None, java=None):
    home = Path(corenlp_home or os.environ.get("CORENLP_HOME") or Path.home() / "stanza_corenlp")
    local_java = Path("/opt/homebrew/opt/openjdk/bin/java")
    executable = java or os.environ.get("JAVA") or (str(local_java) if local_java.is_file() else shutil.which("java"))
    if not executable or not any(home.glob("stanford-corenlp-*.jar")):
        raise RuntimeError("Set CORENLP_HOME to the CoreNLP installation and JAVA to a JDK executable")
    return [executable, "-Xmx2g", "-cp", str(home / "*"),
            str(Path(__file__).with_name("IntentParses.java")), mode]
