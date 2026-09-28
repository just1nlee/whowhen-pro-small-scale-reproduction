"""Stage 1: run the smolagents CodeAgent on DA-Bench questions and record full trajectories.

Usage (from whowhen_pro_small_scale_reproduction/):
    python -m scripts.run_seeds --n 40            # select 40 questions and run them
    python -m scripts.run_seeds --n 40 --verbose  # also print each step's thought/code/output
    python -m scripts.run_seeds --ids 0 5 6       # run specific question ids

Each run is saved to outputs/seeds/runs/<id>.json; already-finished ids are skipped,
so an interrupted batch can be resumed without paying twice.
"""

import argparse
import json
import random
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from repro.agent import MAX_STEPS, MODEL_ID, make_agent, record_steps, task_workspace
from repro.claude_model import ClaudeModel
from repro.dabench import load_tasks
from repro.grader import grade

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "seeds"
PAPER_TRACES = ROOT.parent / "whowhen-pro" / "data" / "text.jsonl"

MIN_SEED_STEPS = 2
SELECTION_SEED = 0
# Runs that pass grading but are excluded as seeds by hand, with the reason.
EXCLUDED_SEEDS = {
    321: "used all MAX_STEPS steps; no step budget left for a post-injection rollout, "
         "so an injected run could fail from the step cap rather than the injected error",
    590: "fails the warm-start replay fidelity check: step 1's error message lists the authorized "
         "imports in set order, which varies with Python's hash randomization, so the pre-injection "
         "context can't be reproduced byte for byte",
}


def paper_question_ids(tasks) -> list[int]:
    """DA-Bench ids used by the paper's released smolagents traces (matched on prompt text)."""
    by_prompt = {t.prompt(): t.id for t in tasks}
    ids = set()
    with open(PAPER_TRACES) as f:
        for line in f:
            r = json.loads(line)
            if r["benchmark"] == "dabench" and r["framework"] == "smolagents":
                ids.add(by_prompt[json.loads(r["trajectory"])[0]["content"]])
    return sorted(ids)


def select_ids(tasks, n: int) -> list[int]:
    """n paper questions, balanced across easy/medium/hard, fixed random seed."""
    level = {t.id: t.level for t in tasks}
    pool = paper_question_ids(tasks)
    rng = random.Random(SELECTION_SEED)
    by_level = {lv: [i for i in pool if level[i] == lv] for lv in ("easy", "medium", "hard")}
    for ids in by_level.values():
        rng.shuffle(ids)
    picked = []
    while len(picked) < n and any(by_level.values()):
        for lv in ("easy", "medium", "hard"):
            if by_level[lv] and len(picked) < n:
                picked.append(by_level[lv].pop())
    return sorted(picked)


def run_one(task, verbose: bool) -> dict:
    model = ClaudeModel(MODEL_ID)
    agent = make_agent(model, verbose)
    with task_workspace(task):
        try:
            answer, run_error = agent.run(task.prompt()), None
        except Exception as e:  # API failures etc. -- record and move on
            answer, run_error = None, f"{type(e).__name__}: {e}"
    steps = record_steps(agent)

    g = grade(task, answer)
    return {
        "task_id": task.id,
        "level": task.level,
        "model": MODEL_ID,
        "max_steps": MAX_STEPS,
        "system_prompt": agent.memory.system_prompt.system_prompt,
        "task_prompt": task.prompt(),
        "steps": steps,
        "final_answer": None if answer is None else str(answer),
        "correct": g.correct,
        "per_answer": g.per_answer,
        "gold": task.gold,
        "n_steps": len(steps),
        "any_step_error": any(s["error"] for s in steps),
        "run_error": run_error,
        "is_seed": (g.correct and len(steps) >= MIN_SEED_STEPS and run_error is None
                    and task.id not in EXCLUDED_SEEDS),
        "excluded_reason": EXCLUDED_SEEDS.get(task.id),
        "usage": model.usage_log,
        "cost_usd": model.total_cost(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="number of paper questions to select and run")
    ap.add_argument("--ids", type=int, nargs="+", help="run these question ids instead")
    ap.add_argument("--max-cost", type=float, default=2.00, help="stop the batch once this much USD is spent")
    ap.add_argument("--verbose", action="store_true", help="print each step's thought, code, and output")
    args = ap.parse_args()

    load_dotenv(find_dotenv())
    tasks = load_tasks()
    by_id = {t.id: t for t in tasks}
    ids = args.ids or select_ids(tasks, args.n)

    runs_dir = OUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not args.ids:  # --ids is for ad-hoc runs; don't overwrite the batch's selection
        (OUT_DIR / "selected_ids.json").write_text(json.dumps(ids))

    spent = 0.0
    for i, task_id in enumerate(ids, 1):
        path = runs_dir / f"{task_id}.json"
        if path.exists():
            print(f"[{i}/{len(ids)}] q{task_id}: already done, skipping")
            continue
        if spent >= args.max_cost:
            print(f"stopping: spent ${spent:.2f} >= --max-cost ${args.max_cost:.2f}")
            break
        rec = run_one(by_id[task_id], args.verbose)
        path.write_text(json.dumps(rec, indent=1))
        spent += rec["cost_usd"]
        status = "SEED" if rec["is_seed"] else ("correct" if rec["correct"] else "wrong")
        print(f"[{i}/{len(ids)}] q{task_id} ({rec['level']}): {status}, {rec['n_steps']} steps, "
              f"${rec['cost_usd']:.4f} (batch total ${spent:.2f})"
              + (f"  run error: {rec['run_error']}" if rec["run_error"] else ""), flush=True)

    recs = []
    for i in ids:
        path = runs_dir / f"{i}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        # Apply exclusions to runs saved before they were added to EXCLUDED_SEEDS.
        if rec["task_id"] in EXCLUDED_SEEDS and rec.get("excluded_reason") is None:
            rec["is_seed"], rec["excluded_reason"] = False, EXCLUDED_SEEDS[rec["task_id"]]
            path.write_text(json.dumps(rec, indent=1))
        recs.append(rec)
    n_seed = sum(r["is_seed"] for r in recs)
    n_excl = sum(r["correct"] and r.get("excluded_reason") is not None for r in recs)
    print(f"\n{len(recs)} runs: {sum(r['correct'] for r in recs)} correct, {n_excl} excluded by hand, "
          f"{n_seed} usable seeds (correct and >= {MIN_SEED_STEPS} steps); "
          f"total cost ${sum(r['cost_usd'] for r in recs):.2f}")


if __name__ == "__main__":
    main()
