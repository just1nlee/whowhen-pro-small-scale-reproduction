"""DA-Bench (InfiAgent-DABench) data loader.

Joins the dev questions with their gold labels and renders the task prompt
given to the agent. The prompt template is copied from the smolagents x
DA-Bench traces released with Who&When Pro (all 327 traces use it verbatim).
"""

import json
from dataclasses import dataclass
from pathlib import Path

DA_BENCH_DIR = Path(__file__).resolve().parent.parent / "da-bench"
DATA_DIR = DA_BENCH_DIR / "data"

PROMPT_TEMPLATE = """\
You are a data analyst. Answer the following data analysis question by writing Python code to load and analyze the provided CSV file.

## Available data files
  - {data_path}

## Question
{question}

## Constraints
{constraints}

## Expected answer format
{format}

After analyzing the data, submit your answer with final_answer(your_answer_here) in the specified format."""


@dataclass
class DABenchTask:
    id: int
    question: str
    constraints: str
    format: str
    file_name: str
    level: str  # easy / medium / hard
    concepts: list[str]
    csv_path: Path  # absolute path to the CSV on disk
    gold: list[list[str]]  # [[answer_name, value], ...]; never shown to the agent

    def prompt(self, data_path: str | None = None) -> str:
        """Task prompt for the agent. `data_path` is the CSV path as the agent
        should see it; defaults to ./data/<file_name>, as in the paper's traces."""
        return PROMPT_TEMPLATE.format(
            data_path=data_path or f"./data/{self.file_name}",
            question=self.question,
            constraints=self.constraints,
            format=self.format,
        )


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tasks(data_dir: Path = DATA_DIR) -> list[DABenchTask]:
    """Load all dev-set tasks, sorted by id."""
    labels = {r["id"]: r["common_answers"] for r in _read_jsonl(data_dir / "da-dev-labels.jsonl")}
    tables = data_dir / "da-dev-tables"

    tasks = []
    for q in _read_jsonl(data_dir / "da-dev-questions.jsonl"):
        csv_path = tables / q["file_name"]
        if not csv_path.exists():
            raise FileNotFoundError(f"task {q['id']}: missing table {csv_path}")
        tasks.append(DABenchTask(
            id=q["id"],
            question=q["question"],
            constraints=q["constraints"],
            format=q["format"],
            file_name=q["file_name"],
            level=q["level"],
            concepts=q["concepts"],
            csv_path=csv_path.resolve(),
            gold=labels[q["id"]],
        ))
    return sorted(tasks, key=lambda t: t.id)


def load_task(task_id: int, data_dir: Path = DATA_DIR) -> DABenchTask:
    for t in load_tasks(data_dir):
        if t.id == task_id:
            return t
    raise KeyError(f"no DA-Bench task with id {task_id}")
