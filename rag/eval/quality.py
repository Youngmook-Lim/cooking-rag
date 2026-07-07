from dataclasses import dataclass
from rag.models import Retrieved


@dataclass
class QualityResult:
    question_id: int
    query: str
    answer: str
    cited_post_ids: list[int]
    correctness: int | None
    faithfulness: int | None
    rationale: str
    refused: bool = False
    judge_error: bool = False


@dataclass
class QualityReport:
    n_cases: int
    n_judged: int
    mean_correctness: float
    mean_faithfulness: float
    refusal_rate: float
    n_judge_errors: int


class JudgeParseError(ValueError):
    """The judge returned output that could not be parsed into scores."""


def render_sources(retrieved: list[Retrieved]) -> str:
    return "\n\n".join(f"[{r.chunk.post_id}] {r.chunk.text}" for r in retrieved)


def build_judge_prompt(question: str, answer: str, sources: str, reference: str) -> str:
    return (
        "You are evaluating a cooking assistant's answer.\n\n"
        f"Question: {question}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"Sources the answer was built from:\n{sources}\n\n"
        "Reference answer (one known-correct answer; OTHER correct answers are also "
        "valid — do not require the answer to match this one):\n"
        f"{reference}\n\n"
        "Score two things, each an integer 1 to 5:\n"
        "- correctness: is the answer a correct, complete answer to the question? Use "
        "the reference as a guide to the facts, but credit any answer that is correct "
        "even if it differs from the reference.\n"
        "- faithfulness: is every claim in the answer supported by the sources above?\n\n"
        "Respond with ONLY a JSON object and nothing else:\n"
        '{"correctness": N, "faithfulness": N, "rationale": "one sentence"}'
    )
