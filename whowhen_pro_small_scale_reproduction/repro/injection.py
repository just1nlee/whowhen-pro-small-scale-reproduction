"""Two-stage error injection at step t of a seed (Who&When Pro, Section 3.3 / Appendix K).

Stage 1: a frontier model reads the task, the correct answer, and the observations from
steps 1..t-1, and describes a specific flaw (FLAW) plus the wrong answer it leads to.
Stage 2: the base agent model gets its normal context at step t with an injection prompt
(built around FLAW) appended, and writes the corrupted step itself. The injection prompt
is only added to that one model call; it never enters the agent's memory, so later steps
and the final trace contain only the agent's own reply.

Steps 1..t-1 are replayed from the seed (see replay_model.py); steps t+1.. run live.

Departures from the paper's published templates (after pilot 1): stage 1 also sees the
seed's original step t (Thought + code) and must place the flaw in that step, and the R.2
stage 2 prompt requires the step's code to act on the flaw.
"""

import re
from pathlib import Path

import anthropic

from smolagents.models import ChatMessage, MessageRole, Model

from .agent import MODEL_ID, record_steps, task_workspace
from .claude_model import ClaudeModel
from .dabench import DABenchTask
from .grader import grade
from .replay_model import ReplayFidelityError, build_replay_agent

FRONTIER_MODEL_ID = "claude-sonnet-5"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
TEMPLATES = {mode: {stage: (PROMPTS_DIR / f"{mode.replace('.', '').lower()}_stage{stage}.txt").read_text()
                    for stage in (1, 2)}
             for mode in ("R.2", "R.3")}


def format_gold(task: DABenchTask) -> str:
    return " ".join(f"@{name}[{value}]" for name, value in task.gold)


def format_observations(steps: list[dict]) -> str:
    parts = []
    for s in steps:
        body = s["observation"] or ""
        if s["error"]:
            body = (body + "\n" if body else "") + f"Error: {s['error']}"
        parts.append(f"[Step {s['step_number']} observation]\n{body.strip()}")
    return "\n\n".join(parts)


def parse_stage1(text: str) -> tuple[str, str]:
    flaw = re.search(r"FLAW:\s*(.+?)\s*(?=WRONG_ANSWER:|$)", text, flags=re.S)
    wrong = re.search(r"WRONG_ANSWER:\s*(.+)", text, flags=re.S)
    if not flaw or not wrong:
        raise ValueError("stage 1 output is missing FLAW or WRONG_ANSWER")
    return flaw.group(1).strip(), wrong.group(1).strip()


class InjectingModel(Model):
    """Wraps the base agent model: the first call gets `injection_prompt` appended as a
    user message; every later call is passed through unchanged."""

    def __init__(self, base: Model, injection_prompt: str):
        super().__init__(model_id=base.model_id)
        self.base = base
        self.injection_prompt = injection_prompt
        self.injected = False

    def generate(self, messages, **kwargs):
        if not self.injected:
            self.injected = True
            messages = list(messages) + [ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": self.injection_prompt}])]
        return self.base.generate(messages, **kwargs)


def _is_api_error(e: BaseException) -> bool:
    """True if e (or anything it wraps) is an Anthropic API/connection error, e.g. no credit left."""
    while e is not None:
        if isinstance(e, anthropic.APIError):
            return True
        e = e.__cause__ or e.__context__
    return False


def run_attempt(seed: dict, task: DABenchTask, mode: str, t: int, verbose: bool = False) -> dict:
    """One injection attempt: stage 1, replay 1..t-1, inject at t, run live to the end, grade."""
    rec = {"task_id": task.id, "mode": mode, "t": t, "frontier_model": FRONTIER_MODEL_ID,
           "base_model": MODEL_ID, "stage1_prompt": None, "stage1_output": None, "flaw": None,
           "wrong_answer": None, "stage2_prompt": None, "steps": None, "final_answer": None,
           "correct": None, "per_answer": None, "status": None, "error": None}
    frontier = ClaudeModel(FRONTIER_MODEL_ID)
    base = ClaudeModel(MODEL_ID)
    prior = seed["steps"][:t - 1]

    # Stage 1: frontier model describes the flaw.
    rec["stage1_prompt"] = TEMPLATES[mode][1].format(
        query=seed["task_prompt"], ground_truth=format_gold(task), observations=format_observations(prior),
        current_step=seed["steps"][t - 1]["model_output"].strip())
    try:
        out = frontier.generate([ChatMessage(role=MessageRole.USER,
                                             content=[{"type": "text", "text": rec["stage1_prompt"]}])])
        rec["stage1_output"] = out.content
        rec["flaw"], rec["wrong_answer"] = parse_stage1(out.content)
    except Exception as e:  # refusal or unparseable output; API errors propagate
        if _is_api_error(e):
            raise
        rec["status"] = "stage1_refused" if "refused" in str(e) else "stage1_failed"
        rec["error"] = f"{type(e).__name__}: {e}"
        return _finish(rec, frontier, base)

    # Stage 2 + rollout: replay 1..t-1, inject at t, continue live.
    last = prior[-1]
    last_obs = (last["observation"] or "") + (f"\nError: {last['error']}" if last["error"] else "")
    rec["stage2_prompt"] = TEMPLATES[mode][2].format(
        query=seed["task_prompt"], last_observation=last_obs.strip(), flaw=rec["flaw"])
    agent = build_replay_agent(seed, task, upto=t - 1,
                               live_model=InjectingModel(base, rec["stage2_prompt"]), verbose=verbose)
    answer = None
    with task_workspace(task):
        try:
            answer = agent.run(seed["task_prompt"])
        except ReplayFidelityError as e:
            rec["status"], rec["error"] = "replay_diverged", str(e)
        except Exception as e:
            if _is_api_error(e):  # e.g. out of credit: don't record a half-run attempt
                raise
            rec["status"], rec["error"] = "rollout_error", f"{type(e).__name__}: {e}"
    rec["steps"] = record_steps(agent)

    if rec["status"] is None:
        g = grade(task, answer)
        rec["final_answer"] = None if answer is None else str(answer)
        rec["correct"], rec["per_answer"] = g.correct, g.per_answer
        rec["status"] = "recovered" if g.correct else "failed"
    return _finish(rec, frontier, base)


def _finish(rec: dict, frontier: ClaudeModel, base: ClaudeModel) -> dict:
    rec["usage"] = frontier.usage_log + base.usage_log
    rec["cost_usd"] = frontier.total_cost() + base.total_cost()
    return rec
