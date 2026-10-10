import pytest


@pytest.fixture(autouse=True)
def _no_label_scheme_env(monkeypatch):
    # The label scheme defaults from JEV_LABEL_SCHEME; tests set it explicitly.
    monkeypatch.delenv("JEV_LABEL_SCHEME", raising=False)
