import os
import subprocess
import sys
import time
from pathlib import Path

# Runs the whole training pipeline as fast as possible:
# build_labels.py (prices) and oss_labeler.py (text scores) at the same time, then train_models.py.
# Run from the repo root:  python src/data/run_training.py

HERE = Path(__file__).resolve().parent


def run(script):
    return subprocess.Popen([sys.executable, str(HERE / script)])


# Catch setup problems before anything runs
missing = [k for k in ("MASSIVE_API_KEY", "OSS_API_KEY") if not os.environ.get(k)]
if missing:
    sys.exit(f"Set {' and '.join(missing)} in this terminal first (see the run steps).")
needed = ["build_labels.py", "intraday_entry.py", "oss_labeler.py", "train_models.py"]
absent = [f for f in needed if not (HERE / f).exists()]
if absent:
    sys.exit(f"Missing from {HERE}: {', '.join(absent)}")

start = time.time()
labels, scores = run("build_labels.py"), run("oss_labeler.py")
codes = labels.wait(), scores.wait()
print(f"\nLabels and scores finished in {(time.time() - start) / 60:.0f} min")
if any(codes):
    sys.exit("One step failed (see its output above). Re-run this script; both steps resume where they stopped.")

subprocess.run([sys.executable, str(HERE / "train_models.py")], check=True)
print(f"\nTotal time: {(time.time() - start) / 60:.0f} min")
