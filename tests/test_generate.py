from types import SimpleNamespace
from rag.models import Chunk, Retrieved
from rag.generate import Generator, build_prompt, Answer, OpenRouterLLM


class FakeOpenAIClient:
    def __init__(self, reply):
        self.received = {}
        message = SimpleNamespace(content=reply)
        choice = SimpleNamespace(message=message)
        self._response = SimpleNamespace(choices=[choice])
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.received = kwargs
        return self._response


def test_openrouter_llm_sends_model_and_returns_text():
    client = FakeOpenAIClient(reply="Oil added too fast [5570].")
    llm = OpenRouterLLM(model="anthropic/claude-sonnet-4", client=client)
    out = llm.complete("why did my mayo break?")
    assert out == "Oil added too fast [5570]."
    assert client.received["model"] == "anthropic/claude-sonnet-4"
    assert client.received["messages"] == [
        {"role": "user", "content": "why did my mayo break?"}
    ]


class FakeLLM:
    def __init__(self, reply="Because the oil was added too fast [5570]."):
        self.reply = reply
        self.last_prompt = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.reply


def _r(text, post_id, score):
    return Retrieved(chunk=Chunk(text=text, post_id=post_id, question_id=1),
                     score=score)


def test_build_prompt_includes_question_and_sources():
    prompt = build_prompt("why did my mayo break?",
                          [_r("oil too fast", 5570, 0.9)])
    assert "why did my mayo break?" in prompt
    assert "5570" in prompt
    assert "oil too fast" in prompt


def test_generator_returns_answer_with_citations():
    gen = Generator(llm=FakeLLM(), min_score=0.2)
    ans = gen.answer("why did my mayo break?", [_r("oil too fast", 5570, 0.9)])
    assert isinstance(ans, Answer)
    assert 5570 in ans.cited_post_ids
    assert "oil" in ans.text.lower()


def test_generator_refuses_when_no_chunks():
    gen = Generator(llm=FakeLLM(), min_score=0.2)
    ans = gen.answer("how do I file taxes?", [])
    assert ans.refused is True
    assert ans.cited_post_ids == []


def test_generator_refuses_when_scores_too_low():
    gen = Generator(llm=FakeLLM(), min_score=0.5)
    ans = gen.answer("unrelated", [_r("noise", 1, 0.1)])
    assert ans.refused is True
