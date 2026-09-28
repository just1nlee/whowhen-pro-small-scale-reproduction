"""Check that every seed can be warm-start replayed exactly. Makes no API calls.

Replays ALL recorded steps of each seed (no live model) and checks that every
observation matches the recording byte for byte and that the final answer is
reproduced and still grades correct.

Usage (from the repository root):
    python -m scripts.check_replay
"""

import json
from pathlib import Path

from repro.dabench import load_tasks
from repro.grader import grade
from repro.replay_model import ReplayFidelityError, replay

RUNS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "seeds" / "runs"


def main():
    by_id = {t.id: t for t in load_tasks()}
    seeds = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"), key=lambda p: int(p.stem))]
    seeds = [s for s in seeds if s["is_seed"]]

    failures = []
    for seed in seeds:
        task = by_id[seed["task_id"]]
        try:
            _, answer = replay(seed, task, upto=seed["n_steps"])
            problem = None
            if str(answer) != seed["final_answer"]:
                problem = f"final answer {answer!r} != recorded {seed['final_answer']!r}"
            elif not grade(task, answer).correct:
                problem = "replayed answer no longer grades correct"
        except ReplayFidelityError as e:
            problem = f"fidelity: {e}"
        print(f"q{seed['task_id']} ({seed['n_steps']} steps): {'OK' if problem is None else 'FAIL'}", flush=True)
        if problem:
            failures.append((seed["task_id"], problem))

    print(f"\n{len(seeds) - len(failures)}/{len(seeds)} seeds replay exactly")
    for task_id, problem in failures:
        print(f"\n--- q{task_id} ---\n{problem[:2000]}")


if __name__ == "__main__":
    main()
