"""Stages 3-4: run the injection attempts in outputs/injection/plan.json.

Usage (from the repository root):
    python -m scripts.run_injection --only 18:R.2 66:R.3   # specific attempts (e.g. a pilot)
    python -m scripts.run_injection                        # every attempt in the plan
    python -m scripts.run_injection --verbose              # also print each agent step

Each attempt is saved to outputs/injection/runs/<task_id>_<mode>.json; finished attempts
are skipped, so an interrupted batch can be resumed without paying twice.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from repro.dabench import load_tasks
from repro.injection import _is_api_error, run_attempt

ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = ROOT / "outputs" / "seeds" / "runs"
INJ_DIR = ROOT / "outputs" / "injection"


def attempt_path(task_id: int, mode: str) -> Path:
    return INJ_DIR / "runs" / f"{task_id}_{mode.replace('.', '')}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", help="attempts to run, as <task_id>:<mode>, e.g. 18:R.2")
    ap.add_argument("--max-cost", type=float, default=2.00, help="stop once this much USD is spent")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    load_dotenv(find_dotenv())
    plan = json.loads((INJ_DIR / "plan.json").read_text())
    if args.only:
        wanted = {(int(a.split(":")[0]), a.split(":")[1]) for a in args.only}
        plan = [p for p in plan if (p["task_id"], p["mode"]) in wanted]
        missing = wanted - {(p["task_id"], p["mode"]) for p in plan}
        if missing:
            raise SystemExit(f"not in plan.json: {sorted(missing)}")
    by_id = {t.id: t for t in load_tasks()}
    (INJ_DIR / "runs").mkdir(parents=True, exist_ok=True)

    spent = 0.0
    for i, p in enumerate(plan, 1):
        path = attempt_path(p["task_id"], p["mode"])
        label = f"[{i}/{len(plan)}] q{p['task_id']} {p['mode']} t={p['t']}"
        if path.exists():
            print(f"{label}: already done, skipping")
            continue
        if spent >= args.max_cost:
            print(f"stopping: spent ${spent:.2f} >= --max-cost ${args.max_cost:.2f}")
            break
        seed = json.loads((SEEDS_DIR / f"{p['task_id']}.json").read_text())
        try:
            rec = run_attempt(seed, by_id[p["task_id"]], p["mode"], p["t"], verbose=args.verbose)
        except Exception as e:
            if not _is_api_error(e):
                raise
            print(f"{label}: API error, stopping without saving this attempt "
                  f"(check credit/rate limits, then rerun to resume): {type(e).__name__}: {str(e)[:200]}")
            break
        path.write_text(json.dumps(rec, indent=1))
        spent += rec["cost_usd"]
        n = len(rec["steps"]) if rec["steps"] else 0
        print(f"{label}: {rec['status']}, {n} steps, ${rec['cost_usd']:.4f} (batch total ${spent:.2f})"
              + (f"  {rec['error'].splitlines()[0][:120]}" if rec["error"] else ""), flush=True)

    recs = [json.loads(attempt_path(p["task_id"], p["mode"]).read_text())
            for p in plan if attempt_path(p["task_id"], p["mode"]).exists()]
    print(f"\n{len(recs)} attempts: {dict(Counter(r['status'] for r in recs))}; "
          f"total cost ${sum(r['cost_usd'] for r in recs):.2f}")


if __name__ == "__main__":
    main()
