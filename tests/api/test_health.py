"""
Tests for health check and version endpoints.

These tests verify basic API functionality and metadata endpoints.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Test that health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_response_structure():
    """Test health check endpoint returns correct structure."""
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert "timestamp" in data
    assert "service" in data
    assert data["status"] == "healthy"


def test_version_endpoint_returns_200():
    """Test that version endpoint returns 200 OK."""
    response = client.get("/version")
    assert response.status_code == 200


def test_version_endpoint_response_structure():
    """Test version endpoint returns correct structure."""
    response = client.get("/version")
    data = response.json()

    assert "version" in data
    assert "api_version" in data
    assert "description" in data
    assert data["version"] == "2.3.0"
    assert data["api_version"] == "v1"


def test_openapi_schema_available():
    """Test that OpenAPI schema is generated."""
    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert schema["info"]["title"] == "Agentic Surveillance Research API"


def test_docs_endpoints_accessible():
    """Test that documentation endpoints are accessible."""
    # Swagger UI
    response = client.get("/docs")
    assert response.status_code == 200

    # ReDoc
    response = client.get("/redoc")
    assert response.status_code == 200
