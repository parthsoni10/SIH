from app.modules.explain import generate_officer_explanation, generate_template_explanation

def test_template_explanation_fraudulent():
    exp = generate_template_explanation(
        risk_score=85,
        prediction="fraudulent",
        failed_rules=["invalid_id_checksum", "expired_document"],
        tampering_score=0.6,
        face_match_score=0.4,
        blacklist_hit=True,
        document_type="Passport"
    )
    assert "Flagged" in exp
    assert "checksum" in exp or "expired" in exp or "blacklist" in exp

def test_template_explanation_genuine():
    exp = generate_template_explanation(
        risk_score=10,
        prediction="genuine",
        failed_rules=[],
        tampering_score=0.05,
        face_match_score=0.95,
        blacklist_hit=False,
        document_type="Passport"
    )
    assert "Verified" in exp
