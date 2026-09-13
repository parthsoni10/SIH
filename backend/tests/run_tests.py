import sys
import os
import traceback
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

def run_all():
    print("==================================================")
    print(" Running Backend Unit Test Suite (Modules 1-8)    ")
    print("==================================================")
    
    passed = 0
    failed = 0

    # 1. Test Risk Scoring (Module 6)
    try:
        from tests.test_risk_scoring import (
            test_feature_vector_structure,
            test_predict_risk_genuine,
            test_predict_risk_fraudulent_blacklist,
            test_model_self_test
        )
        test_feature_vector_structure()
        test_predict_risk_genuine()
        test_predict_risk_fraudulent_blacklist()
        test_model_self_test()
        print("[PASS] Module 6: test_risk_scoring")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 6: test_risk_scoring:\n{traceback.format_exc()}")
        failed += 1

    # 2. Test Preprocessing (Module 1)
    try:
        from tests.test_preprocessing import (
            test_preprocess_synthetic_jpeg,
            test_preprocess_resizing,
            test_empty_bytes_raises
        )
        test_preprocess_synthetic_jpeg()
        test_preprocess_resizing()
        test_empty_bytes_raises()
        print("[PASS] Module 1: test_preprocessing")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 1: test_preprocessing:\n{traceback.format_exc()}")
        failed += 1

    # 3. Test OCR & MRZ (Module 2)
    try:
        from tests.test_ocr import (
            test_mrz_check_digit_calculation,
            test_mrz_td3_parsing,
            test_verhoeff_checksum,
            test_pan_number_regex,
            test_extract_ocr_data_dummy_image
        )
        test_mrz_check_digit_calculation()
        test_mrz_td3_parsing()
        test_verhoeff_checksum()
        test_pan_number_regex()
        test_extract_ocr_data_dummy_image()
        print("[PASS] Module 2: test_ocr & test_mrz")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 2: test_ocr & test_mrz:\n{traceback.format_exc()}")
        failed += 1

    # 4. Test Validation (Module 3)
    try:
        from tests.test_validation import (
            test_parse_date_safely,
            test_validate_document_valid_passport,
            test_validate_document_expired_passport,
            test_validate_document_blacklisted_id
        )
        test_parse_date_safely()
        test_validate_document_valid_passport()
        test_validate_document_expired_passport()
        test_validate_document_blacklisted_id()
        print("[PASS] Module 3: test_validation")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 3: test_validation:\n{traceback.format_exc()}")
        failed += 1

    # 5. Test Tampering (Module 4)
    try:
        from tests.test_tampering import (
            test_detect_tampering_structure,
            test_metadata_editing_software
        )
        test_detect_tampering_structure()
        test_metadata_editing_software()
        print("[PASS] Module 4: test_tampering")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 4: test_tampering:\n{traceback.format_exc()}")
        failed += 1

    # 6. Test Face Match (Module 5)
    try:
        from tests.test_face_match import (
            test_face_match_no_live_capture,
            test_detect_and_crop_face_blank_image
        )
        test_face_match_no_live_capture()
        test_detect_and_crop_face_blank_image()
        print("[PASS] Module 5: test_face_match")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 5: test_face_match:\n{traceback.format_exc()}")
        failed += 1

    # 7. Test Gemini Explanation (Module 7)
    try:
        from tests.test_explain import (
            test_template_explanation_fraudulent,
            test_template_explanation_genuine
        )
        test_template_explanation_fraudulent()
        test_template_explanation_genuine()
        print("[PASS] Module 7: test_explain")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 7: test_explain:\n{traceback.format_exc()}")
        failed += 1

    # 8. Test Orchestration (Module 8)
    try:
        from tests.test_orchestration import (
            test_verify_document_success,
            test_verify_document_unsupported_type,
            test_audit_trail_pagination_and_retrieval,
            test_audit_trail_filtering
        )
        test_verify_document_success()
        test_verify_document_unsupported_type()
        test_audit_trail_pagination_and_retrieval()
        test_audit_trail_filtering()
        print("[PASS] Module 8: test_orchestration")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Module 8: test_orchestration:\n{traceback.format_exc()}")
        failed += 1

    print("--------------------------------------------------")
    print(f" Summary: {passed} passed, {failed} failed.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    run_all()
