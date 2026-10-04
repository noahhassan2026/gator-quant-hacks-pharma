"""Reproduces the quant note's headline numbers with one command:  python run_all.py
Needs MASSIVE_API_KEY, OSS_API_KEY, OSS_BASE_URL, OSS_MODEL and SEC_USER_AGENT set (see .env.example).
Steps that already produced their output file are skipped, so a re-run resumes where it stopped."""
import os
import subprocess
import sys

STEPS = [
    ("historical_sec_8k_2016_2024.csv", ["src/data/sec_edgar_8k.py"]),
    ("filings_2025_2026.csv", ["src/data/sec_edgar_8k.py", "2025", "2026", "filings_2025_2026.csv"]),
    ("direction_model.joblib", ["src/data/run_training.py"]),
    ("backtest_equity.csv", ["src/data/backtest.py"]),
    ("report_metrics.json", ["src/data/report_metrics.py"]),
]

for output, cmd in STEPS:
    if os.path.exists(output):
        print(f"Skipping {cmd[0]} ({output} already exists)")
        continue
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    if subprocess.run([sys.executable, *cmd]).returncode != 0:
        sys.exit(f"{cmd[0]} failed; fix the error above and run again.")
print("\nDone. Headline numbers: backtest.py output above, backtest_report.png and report_metrics.json.")
