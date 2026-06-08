import re
from dataclasses import dataclass, field
from rag.models import Retrieved

REFUSAL_TEXT = "I couldn't find anything in the corpus that covers this question."


@dataclass
class Answer:
    text: str
    cited_post_ids: list[int] = field(default_factory=list)
    refused: bool = False


def build_prompt(question: str, retrieved: list[Retrieved]) -> str:
    sources = "\n\n".join(
        f"[{r.chunk.post_id}] {r.chunk.text}" for r in retrieved
    )
    return (
        "You are a cooking expert. Answer the question using ONLY the sources "
        "below. Cite the post id in square brackets, e.g. [5570], after each claim. "
        "If the sources do not contain the answer, say you don't know.\n\n"
        f"Sources:\n{sources}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


class LLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class ClaudeLLM(LLM):
    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model

    def complete(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text


class Generator:
    def __init__(self, llm: LLM, min_score: float = 0.2):
        self.llm = llm
        self.min_score = min_score

    def answer(self, question: str, retrieved: list[Retrieved]) -> Answer:
        usable = [r for r in retrieved if r.score >= self.min_score]
        if not usable:
            return Answer(text=REFUSAL_TEXT, cited_post_ids=[], refused=True)
        text = self.llm.complete(build_prompt(question, usable))
        cited = [int(pid) for pid in re.findall(r"\[(\d+)\]", text)]
        return Answer(text=text, cited_post_ids=cited, refused=False)
