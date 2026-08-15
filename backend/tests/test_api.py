"""Integration tests for FastAPI REST API endpoints (SDD §12)."""
from fastapi.testclient import TestClient
import pytest

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_repositories():
    response = client.get("/api/repositories")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_nonexistent_repository():
    response = client.get("/api/repositories/999999")
    assert response.status_code == 404


def test_get_nonexistent_repo_graph():
    response = client.get("/api/repositories/999999/graph")
    assert response.status_code == 404


def test_get_nonexistent_repo_search():
    response = client.get("/api/repositories/999999/search?q=test&type=file")
    assert response.status_code == 404


def test_get_file_detail_and_module_resolution():
    # If Flask repo (id=2) is present in DB, test module and path resolution
    response = client.get("/api/repositories/2/files/src/flask")
    if response.status_code == 200:
        data = response.json()
        assert "path" in data
        assert data["path"].startswith("src/flask")
        assert "symbols" in data
        assert "imports" in data
