import joblib
import pytest
from fastapi.testclient import TestClient

from newsclf import serve
from newsclf.baseline import build_pipeline


@pytest.fixture()
def client(toy_df, tmp_path, monkeypatch):
    joblib.dump(build_pipeline().fit(toy_df["text"], toy_df["label"]), tmp_path / "baseline.joblib")
    monkeypatch.setattr(serve, "ARTIFACTS", tmp_path)
    monkeypatch.setenv("NEWSCLF_BACKEND", "linear")
    serve.get_predictor.cache_clear()
    yield TestClient(serve.app)
    serve.get_predictor.cache_clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "backend": "linear"}


def test_predict_returns_label_and_probabilities(client):
    res = client.post("/predict", json={"texts": ["the coach says the season goal", "chip internet software"]})
    assert res.status_code == 200
    preds = res.json()["predictions"]
    assert [p["label"] for p in preds] == ["Sports", "Sci/Tech"]
    assert abs(sum(preds[0]["probabilities"].values()) - 1) < 1e-3


@pytest.mark.parametrize("payload", [{"texts": []}, {"texts": ["  "]}, {"texts": ["x"] * 65}, {}])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
