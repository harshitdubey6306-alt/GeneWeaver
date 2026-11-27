from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verifies the /health endpoint returns online status and hardware info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "backend" in data["hardware"]


def test_single_alignment_endpoint():
    """Tests end-to-end single guide alignment and database logging via POST /api/v1/align."""
    payload = {
        "genome_sequence": "AGCTTAGCTAGCTACGATCGATCGATCGATCGTAGCTAGCTAG",
        "target_crispr": "CGATCGATCG",
        "max_mismatches": 1
    }
    response = client.post("/api/v1/align", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["match_count"] >= 1
    assert isinstance(data["run_id"], int)


def test_batch_alignment_endpoint():
    """Tests concurrent multi-guide alignment via POST /api/v1/align/batch."""
    payload = {
        "genome_sequence": "AGCTTAGCTAGCTACGATCGATCGATCGATCGTAGCTAGCTAG",
        "target_crisprs": ["CGATCGATCG", "AGCTTAGCTA"],
        "max_mismatches": 0
    }
    response = client.post("/api/v1/align/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_guides"] == 2
    assert data["completed_guides"] == 2


def test_invalid_sequence_returns_422():
    """Verifies invalid nucleotide characters trigger an HTTP 422 error."""
    payload = {
        "genome_sequence": "AGCTXYZTAGC",
        "target_crispr": "AGCTT",
        "max_mismatches": 0
    }
    response = client.post("/api/v1/align", json=payload)
    assert response.status_code == 422