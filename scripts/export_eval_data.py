"""Stage 4: filter the injected runs and export them in the Who&When Pro release format.

Keeps attempts that ended in failure, then applies the paper's post-hoc filters
(Section 3.3): drop traces whose injected step leaks construction artifacts, and
traces where the correct answer was already visible before the injected step.
Also records (without filtering on) whether a failure was format-only, whether the
run hit the step limit, and whether the injected step called final_answer.

Output, laid out as a dataset checkout for whowhen_eval (--data-root outputs/eval_data):
    outputs/eval_data/data/text.jsonl   one row per kept trace, release format
    outputs/eval_data/taxonomy.yaml     copied from the paper's release
    outputs/eval_data/manifest.json     every attempt: kept/dropped, reason, flags,
                                        injection details (never shown to the judge)

Usage (from the repository root):
    python -m scripts.export_eval_data
"""

import json
import re
import shutil
from pathlib import Path

from repro.agent import MAX_STEPS
from repro.dabench import load_tasks
from repro.grader import grade

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "injection" / "runs"
OUT_DIR = ROOT / "outputs" / "eval_data"
TAXONOMY = ROOT / "whowhen-pro" / "taxonomy.yaml"

# Phrases in the injected step that would reveal the injection prompt to a reader.
LEAK_PATTERNS = [r"inject", r"instruct", r"red[- ]?team", r"\bflaw", r"deliberate", r"intentional",
                 r"on purpose", r"pretend", r"as (?:told|directed)", r"the error described",
                 r"make (?:a|this|the) (?:specific )?(?:reasoning|calculation) error"]


def leak_hits(text: str) -> list[str]:
    return [p for p in LEAK_PATTERNS if re.search(p, text, flags=re.I)]


MIN_SALIENT_LEN = 3  # shorter values (e.g. counts like "4") match unrelated numbers in printed tables


def answer_salient(task, prior_steps: list[dict]) -> bool | None:
    """True if every gold value already appears verbatim in an observation before step t.
    None if the gold values are too short to check by string matching."""
    values = [str(v) for _, v in task.gold]
    if any(len(v) < MIN_SALIENT_LEN for v in values):
        return None
    seen = "\n".join(s["observation"] or "" for s in prior_steps)
    return all(re.search(rf"(?<![\w.]){re.escape(v)}(?![\w])", seen) for v in values)


def to_release_row(rec: dict, task) -> dict:
    trajectory = [{"kind": "user", "content": task.prompt()}]
    for s in rec["steps"]:
        trajectory.append({"kind": "action", "step_number": s["step_number"], "reasoning": s["model_output"],
                           "code": s["code"], "observation": s["observation"], "observation_images": [],
                           "error": s["error"], "is_final_answer": s["is_final_answer"]})
    answer = ", ".join(f"@{name}[{value}]" for name, value in task.gold)
    return {
        "id": f"dabench_smolagents_repro_{rec['task_id']}_{rec['mode'].replace('.', '')}",
        "framework": "smolagents",
        "benchmark": "dabench",
        "task": json.dumps({"query": task.question, "answer": answer}),
        "trajectory": json.dumps(trajectory),
        "ground_truth": json.dumps({"agent": None, "step": rec["t"], "mode": rec["mode"]}),
        "extras": json.dumps({}),
    }


def main():
    by_id = {t.id: t for t in load_tasks()}
    recs = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"))]

    manifest, rows = [], []
    for rec in recs:
        task = by_id[rec["task_id"]]
        entry = {"task_id": rec["task_id"], "mode": rec["mode"], "t": rec["t"], "status": rec["status"],
                 "flaw": rec["flaw"], "wrong_answer": rec["wrong_answer"], "final_answer": rec["final_answer"]}
        if rec["status"] != "failed":
            entry.update(kept=False, reason=f"not a failure ({rec['status']})")
            manifest.append(entry)
            continue

        injected = rec["steps"][rec["t"] - 1]
        g = grade(task, rec["final_answer"])
        missing = [name for name, _ in task.gold if name not in g.extracted]
        entry["flags"] = {
            "leak_patterns": leak_hits(injected["model_output"]),
            "answer_salient_before_t": answer_salient(task, rec["steps"][:rec["t"] - 1]),
            "format_only_failure": bool(missing) and all(g.per_answer[n] for n, _ in task.gold if n not in missing),
            "missing_answer_names": missing,
            "hit_step_limit": len(rec["steps"]) > MAX_STEPS or any("max steps" in (s["error"] or "").lower()
                                                                    for s in rec["steps"]),
            "injected_step_called_final_answer": injected["is_final_answer"],
        }
        if entry["flags"]["leak_patterns"]:
            entry.update(kept=False, reason="injected step leaks construction artifacts")
        elif entry["flags"]["answer_salient_before_t"]:
            entry.update(kept=False, reason="correct answer already visible before step t")
        else:
            row = to_release_row(rec, task)
            entry.update(kept=True, reason=None, trace_id=row["id"])
            rows.append(row)
        manifest.append(entry)

    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "data" / "text.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    shutil.copy(TAXONOMY, OUT_DIR / "taxonomy.yaml")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))

    failed = [m for m in manifest if m["status"] == "failed"]
    print(f"{len(recs)} attempts, {len(failed)} failed, {len(rows)} kept")
    for m in failed:
        fl = m["flags"]
        notes = ([ "salience_unchecked"] if fl["answer_salient_before_t"] is None else []) + [k for k in ("format_only_failure", "hit_step_limit", "injected_step_called_final_answer") if fl[k]]
        print(f"  q{m['task_id']} {m['mode']} t={m['t']}: {'KEPT' if m['kept'] else 'DROPPED - ' + m['reason']}"
              + (f"  leak={fl['leak_patterns']}" if fl["leak_patterns"] else "")
              + (f"  flags={notes}" if notes else ""))
    for mode in ("R.2", "R.3"):
        print(f"{mode}: {sum(1 for m in manifest if m['mode'] == mode and m.get('kept'))} kept")


if __name__ == "__main__":
    main()
