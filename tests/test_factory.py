from rag.factory import _model_name


def test_model_name_defaults_to_sonnet(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    assert _model_name() == "anthropic/claude-sonnet-4"


def test_model_name_reads_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-opus-4.8")
    assert _model_name() == "anthropic/claude-opus-4.8"
