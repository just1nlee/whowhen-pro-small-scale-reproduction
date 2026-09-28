"""Shared smolagents CodeAgent setup, so seed runs and replays use identical agent settings."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from smolagents import CodeAgent
from smolagents.memory import ActionStep

from .dabench import DABenchTask

MODEL_ID = "claude-haiku-4-5"
MAX_STEPS = 15
# Libraries the paper's own DA-Bench traces imported.
AUTHORIZED_IMPORTS = ["pandas", "numpy", "scipy", "scipy.*", "sklearn", "sklearn.*",
                      "statsmodels", "statsmodels.*", "statistics", "re", "datetime"]


def make_agent(model, verbose: bool = False, step_callbacks=None) -> CodeAgent:
    return CodeAgent(tools=[], model=model, max_steps=MAX_STEPS, verbosity_level=2 if verbose else 0,
                     additional_authorized_imports=AUTHORIZED_IMPORTS, step_callbacks=step_callbacks)


@contextmanager
def task_workspace(task: DABenchTask):
    """Run inside a scratch folder where ./data/<file> links to the task's CSV,
    since the task prompt points the agent at that relative path."""
    workspace = Path(tempfile.mkdtemp(prefix=f"dabench_{task.id}_"))
    (workspace / "data").mkdir()
    (workspace / "data" / task.file_name).symlink_to(task.csv_path)
    prev_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        yield workspace
    finally:
        os.chdir(prev_cwd)


def record_step(s: ActionStep) -> dict:
    return {
        "step_number": s.step_number,
        "model_output": s.model_output,
        "code": s.code_action,
        "observation": s.observations,
        "error": str(s.error) if s.error else None,
        "is_final_answer": s.is_final_answer,
    }


def record_steps(agent: CodeAgent) -> list[dict]:
    return [record_step(s) for s in agent.memory.steps if isinstance(s, ActionStep)]
