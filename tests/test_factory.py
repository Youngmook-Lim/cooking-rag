import pytest
from rag.factory import _model_name, build_retriever


def test_model_name_defaults_to_sonnet(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert _model_name() == "anthropic/claude-sonnet-4"


def test_model_name_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.8")
    assert _model_name() == "anthropic/claude-opus-4.8"


def test_build_retriever_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_retriever("bogus")


def test_build_llm_uses_model_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "some/model")
    from rag.factory import build_llm
    llm = build_llm()
    assert llm._model == "some/model"
    assert llm._temperature is None


def test_build_llm_forwards_temperature(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    from rag.factory import build_llm
    assert build_llm(temperature=0.0)._temperature == 0.0
