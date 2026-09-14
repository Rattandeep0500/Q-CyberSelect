import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]

experiments = {
    "DkS": root / "qiskit" / "experiments" / "dks_reproduction.py",
    "MIS": root / "qiskit" / "experiments" / "mis_reproduction.py",
    "MVC": root / "qiskit" / "experiments" / "mvc_reproduction.py"
}

results = {}

for name, script in experiments.items():
    process = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True
    )

    results[name] = {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr
    }

output = root / "results" / "benchmark_runs.json"

with open(output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

for name, result in results.items():
    print(f"=== {name} ===")
    print(result["stdout"])
    print()
