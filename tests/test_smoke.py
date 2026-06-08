import rag


def test_package_imports():
    assert hasattr(rag, "__version__")
