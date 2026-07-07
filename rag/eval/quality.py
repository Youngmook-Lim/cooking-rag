import json
import re
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


class Judge:
    def score(self, question: str, answer: str, sources: str,
              reference: str) -> tuple[int, int, str]:
        raise NotImplementedError


def _overlap_score(a: str, b: str) -> int:
    at = set(a.lower().split())
    bt = set(b.lower().split())
    if not at or not bt:
        return 1
    frac = len(at & bt) / len(at)
    return max(1, min(5, 1 + round(frac * 4)))


class FakeJudge(Judge):
    """Deterministic, offline judge: word-overlap proxy bucketed to 1-5."""

    def score(self, question, answer, sources, reference):
        correctness = _overlap_score(answer, reference)
        faithfulness = _overlap_score(answer, sources)
        return correctness, faithfulness, "fake judge: overlap-based score"


def parse_verdict(raw: str) -> tuple[int, int, str]:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise JudgeParseError(f"no JSON object in judge output: {raw!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON in judge output: {raw!r}") from exc
    try:
        correctness = int(data["correctness"])
        faithfulness = int(data["faithfulness"])
        rationale = str(data.get("rationale", ""))
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeParseError(f"missing/invalid fields in judge output: {raw!r}") from exc
    if not (1 <= correctness <= 5 and 1 <= faithfulness <= 5):
        raise JudgeParseError(f"scores out of range 1-5: {raw!r}")
    return correctness, faithfulness, rationale


class LLMJudge(Judge):
    def __init__(self, llm):
        self.llm = llm

    def score(self, question, answer, sources, reference):
        raw = self.llm.complete(
            build_judge_prompt(question, answer, sources, reference))
        return parse_verdict(raw)
