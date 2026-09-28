"""Pick the injection step t for each (seed, error mode) from the eligibility judgments.

Reads outputs/injection/eligibility_response.txt (the reply to eligibility_prompt.txt),
checks it against candidate_steps.json, and samples t uniformly from each eligible list
with a fixed random seed. Seeds with no eligible step for a mode are skipped for that mode.

Usage (from whowhen_pro_small_scale_reproduction/):
    python -m scripts.pick_injection_steps
Writes outputs/injection/plan.json.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INJ_DIR = ROOT / "outputs" / "injection"
MODES = ("R.2", "R.3")
PICK_SEED = 0


def parse_response(text: str) -> dict:
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if not blocks:
        raise ValueError("no ```json block in the eligibility response")
    return json.loads(blocks[-1])


def main():
    candidates = {int(k): v for k, v in json.loads((INJ_DIR / "candidate_steps.json").read_text()).items()}
    eligible = {int(k): v for k, v in parse_response((INJ_DIR / "eligibility_response.txt").read_text()).items()}

    if set(eligible) != set(candidates):
        raise ValueError(f"seed ids differ: missing {sorted(set(candidates) - set(eligible))}, "
                         f"extra {sorted(set(eligible) - set(candidates))}")
    for task_id, per_mode in eligible.items():
        for mode in MODES:
            bad = set(per_mode[mode]) - set(candidates[task_id])
            if bad:
                raise ValueError(f"q{task_id} {mode}: non-candidate steps {sorted(bad)}")

    rng = random.Random(PICK_SEED)
    plan, skipped = [], []
    for task_id in sorted(eligible):
        for mode in MODES:
            steps = sorted(eligible[task_id][mode])
            if steps:
                plan.append({"task_id": task_id, "mode": mode, "t": rng.choice(steps), "eligible": steps})
            else:
                skipped.append(f"q{task_id} {mode}")

    (INJ_DIR / "plan.json").write_text(json.dumps(plan, indent=1))
    for mode in MODES:
        ts = [p["t"] for p in plan if p["mode"] == mode]
        dist = {t: ts.count(t) for t in sorted(set(ts))}
        print(f"{mode}: {len(ts)} attempts, t distribution {dist}")
    print(f"skipped (no eligible step): {', '.join(skipped)}")
    print(f"{len(plan)} attempts -> {(INJ_DIR / 'plan.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
