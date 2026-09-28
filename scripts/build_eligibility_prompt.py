"""Render the step-eligibility prompt over all seeds, to paste into claude.ai (no API cost).

Candidate injection steps are 2..T-1 with no execution error, matching the paper's
released DA-Bench traces (never step 1, never the final-answer step) and its rule of
excluding steps with execution errors. The model then says which candidates suit R.2 / R.3.

Usage (from the repository root):
    python -m scripts.build_eligibility_prompt
Writes outputs/injection/eligibility_prompt.txt. Save the model's full reply to
outputs/injection/eligibility_response.txt.
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "outputs" / "seeds" / "runs"
OUT_DIR = ROOT / "outputs" / "injection"
TEMPLATE = ROOT / "prompts" / "step_eligibility.txt"
TAXONOMY = ROOT / "whowhen-pro" / "taxonomy.yaml"


def candidate_steps(seed: dict) -> list[int]:
    return [s["step_number"] for s in seed["steps"][1:-1] if s["error"] is None]


def render_seed(seed: dict) -> str:
    cands = set(candidate_steps(seed))
    parts = [f"##### Seed {seed['task_id']} #####", "TASK:", seed["task_prompt"].strip(),
             f"CORRECT ANSWER: {' '.join(f'@{n}[{v}]' for n, v in seed['gold'])}"]
    for s in seed["steps"]:
        tag = " [CANDIDATE]" if s["step_number"] in cands else ""
        parts += [f"--- Step {s['step_number']}{tag} ---", s["model_output"].strip()]
        if s["error"]:
            parts.append(f"ERROR: {s['error']}")
        if s["observation"]:
            parts.append(f"OBSERVATION:\n{s['observation'].strip()}")
    return "\n".join(parts)


def main():
    taxonomy = yaml.safe_load(TAXONOMY.read_text())
    seeds = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"), key=lambda p: int(p.stem))]
    seeds = [s for s in seeds if s["is_seed"]]

    prompt = TEMPLATE.read_text().format(
        r2_definition=taxonomy["R.2"]["description"].strip(),
        r3_definition=taxonomy["R.3"]["description"].strip(),
        seeds="\n\n".join(render_seed(s) for s in seeds),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eligibility_prompt.txt").write_text(prompt)
    (OUT_DIR / "candidate_steps.json").write_text(
        json.dumps({s["task_id"]: candidate_steps(s) for s in seeds}, indent=1))
    n_cands = sum(len(candidate_steps(s)) for s in seeds)
    print(f"{len(seeds)} seeds, {n_cands} candidate steps; ~{len(prompt) // 4:,} tokens -> "
          f"{(OUT_DIR / 'eligibility_prompt.txt').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
