import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.db.database import engine, Base

# Ensure DB schema tables are initialized for tests
Base.metadata.create_all(bind=engine)

client = TestClient(app)

def create_dummy_jpeg(width=600, height=600, color=(120, 140, 160)):
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

from unittest.mock import patch

def test_verify_document_success():
    jpeg_bytes = create_dummy_jpeg()
    files = {
        "document_file": ("passport.jpg", jpeg_bytes, "image/jpeg")
    }
    data = {
        "document_type": "Passport"
    }

    mock_ocr = {
        "fields": {"document_number": {"value": "A1234567"}},
        "ocr_confidence": 0.95,
        "id_checksum_valid": True,
        "mrz": {"present": True},
        "raw_text": ["PASSPORT", "A1234567"],
        "llm_validation": {"llm_available": True, "mismatches_or_anomalies": []}
    }
    mock_layout = {
        "layout_valid": True,
        "layout_score": 0.90,
        "aspect_ratio": 1.42,
        "aspect_ratio_ok": True,
        "emblem_detected": True,
        "spatial_geometry_valid": True,
        "layout_anomalies": []
    }
    mock_llm_val = {
        "llm_available": True,
        "mismatches_or_anomalies": []
    }

    mock_unified_llm = {
        "llm_available": True,
        "extracted_fields": {},
        "mismatches_or_anomalies": [],
        "schema_matched": True,
        "raw_llm_response": "",
        "officer_explanation": "Verified authentic passport document"
    }

    with patch("app.routers.documents.extract_ocr_data", return_value=mock_ocr), \
         patch("app.routers.documents.validate_document_layout", return_value=mock_layout), \
         patch("app.routers.documents.run_unified_llm_analysis", return_value=mock_unified_llm):
        response = client.post("/api/documents/verify", files=files, data=data)
        assert response.status_code == 200
        res_json = response.json()

        assert "id" in res_json
        assert res_json["document_type"] == "Passport"
        assert "risk_score" in res_json
        assert "prediction" in res_json
        assert "ocr_confidence" in res_json
        assert "extracted_fields" in res_json

def test_verify_document_unsupported_type():
    jpeg_bytes = create_dummy_jpeg()
    files = {
        "document_file": ("test.jpg", jpeg_bytes, "image/jpeg")
    }
    data = {
        "document_type": "InvalidDocType123"
    }

    response = client.post("/api/documents/verify", files=files, data=data)
    assert response.status_code == 422

def test_audit_trail_pagination_and_retrieval():
    # Fetch list
    list_res = client.get("/api/audit?page=1&limit=5")
    assert list_res.status_code == 200
    list_json = list_res.json()
    assert "total" in list_json
    assert "items" in list_json

    if list_json["items"]:
        first_id = list_json["items"][0]["id"]
        detail_res = client.get(f"/api/audit/{first_id}")
        assert detail_res.status_code == 200
        assert detail_res.json()["id"] == first_id

def test_audit_trail_filtering():
    res = client.get("/api/audit?document_type=Passport&risk_tier=high")
    assert res.status_code == 200
    assert "items" in res.json()
