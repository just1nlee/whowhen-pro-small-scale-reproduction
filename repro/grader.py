"""Grade an agent's final answer with the official DA-Bench evaluator.

Thin wrapper around `evaluate_responses` from the vendored, unmodified
da-bench/eval_closed_form.py. A task passes only if every sub-answer is
correct (the benchmark's "Accuracy by Question" criterion).
"""

import sys
from dataclasses import dataclass

from .dabench import DA_BENCH_DIR, DABenchTask

# eval_closed_form.py does `from utils.utils import ...`, so its folder must
# be importable. Put it first so an unrelated `utils` package can't shadow it.
sys.path.insert(0, str(DA_BENCH_DIR))
from eval_closed_form import evaluate_responses  # noqa: E402
sys.path.remove(str(DA_BENCH_DIR))


@dataclass
class GradeResult:
    correct: bool
    per_answer: dict[str, bool]  # gold answer name -> correct?
    extracted: dict[str, str]  # @name[value] pairs parsed from the response


def grade(task: DABenchTask, response) -> GradeResult:
    """Grade one final answer. `response` is whatever the agent passed to
    final_answer(); non-strings are converted with str()."""
    text = "" if response is None else str(response)
    labels = [{"id": task.id, "common_answers": task.gold}]
    results = evaluate_responses(labels, [{"id": task.id, "response": text}])

    # The official evaluator silently skips empty responses; count them as wrong.
    if not results:
        return GradeResult(False, {name: False for name, _ in task.gold}, {})

    r = results[0]
    return GradeResult(all(r["correctness"].values()), r["correctness"], r["predicted_answers"])
