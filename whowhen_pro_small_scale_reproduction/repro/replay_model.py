"""Warm-start replay of a recorded seed run (Who&When Pro, Section 3.3 / Appendix F.3).

Steps 1..k are replayed through the normal smolagents loop: at each step the
agent asks its model for a reply, and ReplayModel answers with the reply
recorded in the seed run instead of calling the LLM. smolagents then parses and
executes that code as usual, so both the conversation and the Python state
(variables like `df`) are rebuilt. From step k+1 on, ReplayModel forwards calls
to a live model.

DA-Bench agents make no external tool calls, so there is nothing for the
paper's SHA-256 tool cache to store. Code execution is a stateful environment,
which the paper handles by re-executing with fidelity checks: every replayed
step's observation and error must match the recording byte for byte, or the
attempt is aborted with ReplayFidelityError.
"""

from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import TokenUsage

from .agent import make_agent, task_workspace
from .dabench import DABenchTask


class ReplayFidelityError(Exception):
    """A replayed step did not reproduce the seed's recorded observation."""


class ReplayModel(Model):
    """Answers the first len(recorded_outputs) calls from the recording, then defers to `live_model`."""

    def __init__(self, recorded_outputs: list[str], live_model: Model | None = None):
        super().__init__(model_id="replay")
        self.recorded_outputs = recorded_outputs
        self.live_model = live_model
        self.n_calls = 0

    def generate(self, messages, stop_sequences=None, **kwargs):
        i = self.n_calls
        self.n_calls += 1
        if i < len(self.recorded_outputs):
            return ChatMessage(role=MessageRole.ASSISTANT, content=self.recorded_outputs[i],
                               token_usage=TokenUsage(input_tokens=0, output_tokens=0))
        if self.live_model is None:
            raise RuntimeError(f"recording exhausted after {len(self.recorded_outputs)} steps and no live model given")
        return self.live_model.generate(messages, stop_sequences=stop_sequences, **kwargs)


def _fidelity_check(recorded_steps: list[dict], upto: int):
    """Step callback: compare each replayed step (1..upto) with the recording."""
    def check(memory_step, agent=None):
        if not isinstance(memory_step, ActionStep) or memory_step.step_number > upto:
            return
        rec = recorded_steps[memory_step.step_number - 1]
        observation = memory_step.observations
        error = str(memory_step.error) if memory_step.error else None
        for field, got, want in (("observation", observation, rec["observation"]), ("error", error, rec["error"])):
            if got != want:
                raise ReplayFidelityError(
                    f"step {memory_step.step_number}: {field} differs from the seed\n"
                    f"--- recorded ---\n{want}\n--- replayed ---\n{got}")
    return check


def build_replay_agent(seed: dict, task: DABenchTask, upto: int, live_model: Model | None = None,
                       verbose: bool = False):
    """A CodeAgent that will replay the seed's steps 1..upto, then continue with `live_model`.
    Run it inside task_workspace(task) with agent.run(seed["task_prompt"])."""
    if not 0 <= upto <= seed["n_steps"]:
        raise ValueError(f"upto={upto} outside 0..{seed['n_steps']}")
    if task.prompt() != seed["task_prompt"]:
        raise ReplayFidelityError("task prompt differs from the seed")

    recorded = seed["steps"][:upto]
    model = ReplayModel([s["model_output"] for s in recorded], live_model)
    agent = make_agent(model, verbose=verbose, step_callbacks=[_fidelity_check(recorded, upto)])
    if agent.system_prompt != seed["system_prompt"]:
        raise ReplayFidelityError("smolagents system prompt differs from the seed")
    return agent


def replay(seed: dict, task: DABenchTask, upto: int, live_model: Model | None = None, verbose: bool = False):
    """Replay the seed's steps 1..upto, then continue with `live_model` (if given).

    Returns (agent, final_answer). Raises ReplayFidelityError if any replayed
    step diverges from the seed.
    """
    agent = build_replay_agent(seed, task, upto, live_model, verbose)
    with task_workspace(task):
        answer = agent.run(seed["task_prompt"])
    return agent, answer

