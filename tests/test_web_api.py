from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from tdrag.models import Answer
from tdrag.web_api import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_serve_index_html(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "HISTORIA AQUILAE" in response.text or "TDRAG" in response.text


def test_get_stats_endpoint(client: TestClient):
    response = client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert "collection_name" in data
    assert "total_chunks" in data
    assert "num_ctx" in data
    assert data["num_ctx"] == 6144


def test_list_articles_endpoint(client: TestClient):
    response = client.get("/api/articles")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if len(data) > 0:
        assert "filename" in data[0]
        assert "size_kb" in data[0]
        assert "indexed" in data[0]


def test_chat_endpoint_valid_question(client: TestClient):
    fake_answer = Answer(
        text="Bu sahte bir test cevabıdır.",
        grounded=True,
        attempts=1,
        sources=["test_kaynak.pdf"],
    )

    with patch("tdrag.web_api.answer_question", return_value=fake_answer):
        response = client.post("/api/chat", json={"question": "Test sorusu nedir?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Bu sahte bir test cevabıdır."
        assert data["grounded"] is True
        assert data["attempts"] == 1
        assert data["sources"] == ["test_kaynak.pdf"]
        assert "duration_s" in data


def test_chat_endpoint_empty_question_fails(client: TestClient):
    response = client.post("/api/chat", json={"question": ""})
    assert response.status_code in (400, 422)


def test_upload_invalid_file_rejected(client: TestClient):
    response = client.post(
        "/api/upload",
        files=[("files", ("test.txt", b"dummy content", "text/plain"))],
    )
    assert response.status_code == 400


def test_chat_endpoint_with_source_file(client: TestClient):
    fake_answer = Answer(
        text="Normanlar hakkında filtrelenmiş test cevabı.",
        grounded=True,
        attempts=1,
        sources=["864443.pdf"],
    )

    with patch("tdrag.web_api.answer_question", return_value=fake_answer) as mock_ans:
        response = client.post(
            "/api/chat",
            json={"question": "Normanlar kimdir?", "source_file": "864443.pdf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Normanlar hakkında filtrelenmiş test cevabı."
        assert data["sources"] == ["864443.pdf"]
        # source_file parametresinin aktarıldığını doğrula
        mock_ans.assert_called_once()
        _, kwargs = mock_ans.call_args
        assert kwargs.get("source_file") == "864443.pdf"
