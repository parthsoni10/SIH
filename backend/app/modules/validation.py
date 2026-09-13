import re
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dateutil import parser as date_parser

# Mock seed database of blacklisted ID numbers and hashes
MOCK_BLACKLIST_NUMBERS = {
    "A1234567",
    "B9876543",
    "ABCDE1234F",
    "XYZ123456789",
    "123456789012",
    "DL1420110012345",
}

def parse_date_safely(date_str: str) -> Optional[date]:
    """Parses a date string safely into datetime.date object."""
    if not date_str or not isinstance(date_str, str):
        return None
    
    clean_str = date_str.strip()
    try:
        dt = date_parser.parse(clean_str, fuzzy=False)
        return dt.date()
    except Exception:
        # Fallback regex parsing for DD-MM-YYYY or YYYY-MM-DD
        try:
            dt = date_parser.parse(clean_str, fuzzy=True)
            return dt.date()
        except Exception:
            return None

def rule_doc_number_present(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    doc_num = fields.get("document_number", {}).get("value", "")
    if doc_num:
        return True, "doc_number_present", "Document number is present"
    return False, "missing_document_number", "Document number is missing"

def rule_document_number_pattern(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    doc_num = fields.get("document_number", {}).get("value", "").strip().upper()
    if not doc_num:
        return False, "invalid_document_number_format", "Document number missing"
    
    if doc_type == "Aadhaar":
        clean = re.sub(r'\D', '', doc_num)
        passed = len(clean) == 12
    elif doc_type == "PAN Card":
        passed = bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', doc_num))
    elif doc_type in ("Passport", "Visa"):
        passed = bool(re.match(r'^[A-Z0-9]{7,9}$', doc_num))
    elif doc_type == "Driving License":
        passed = len(re.sub(r'\s', '', doc_num)) >= 10
    else:
        passed = len(doc_num) >= 4

    return (True, "doc_number_pattern_valid", "Document number format is valid") if passed else (False, "invalid_document_number_format", f"Invalid format for {doc_type}")

def rule_id_checksum_valid(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    is_valid = ocr_result.get("id_checksum_valid", False)
    if is_valid:
        return True, "id_checksum_valid", "Checksum verification passed"
    return False, "invalid_id_checksum", "Checksum verification failed"

def rule_dob_plausible_range(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    dob_val = fields.get("dob", {}).get("value", "")
    if not dob_val:
        return False, "missing_or_invalid_dob", "DOB field is missing"
    
    dob_date = parse_date_safely(dob_val)
    if not dob_date:
        return False, "missing_or_invalid_dob", "DOB format is unparseable"
    
    today = date.today()
    if dob_date >= today:
        return False, "future_dob", "DOB is in the future"
    
    min_date = today - timedelta(days=int(365.25 * 120))
    if dob_date < min_date:
        return False, "implausible_dob_age", "DOB indicates age > 120 years"
    
    return True, "dob_valid", "DOB is valid and plausible"

def rule_expiry_date(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    exp_val = fields.get("expiry_date", {}).get("value", "")
    today = date.today()

    if doc_type in ("Passport", "Visa"):
        if not exp_val:
            return False, "missing_expiry_date", "Expiry date missing for document type requiring it"
        exp_date = parse_date_safely(exp_val)
        if not exp_date:
            return False, "missing_expiry_date", "Expiry date unparseable"
        if exp_date <= today:
            return False, "expired_document", f"Document expired on {exp_date}"
        return True, "expiry_valid", "Document is active"
    else:
        if exp_val:
            exp_date = parse_date_safely(exp_val)
            if exp_date and exp_date <= today:
                return False, "expired_document", f"Document expired on {exp_date}"
        return True, "expiry_valid", "Document is active or non-expiring"

def _extract_field_val(val: Any) -> str:
    if isinstance(val, dict):
        return str(val.get("value", "") or "")
    if isinstance(val, str):
        return val
    return ""

def rule_name_present(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    for key in ("name", "full_name", "surname", "given_names", "father_name", "holder_name"):
        val = _extract_field_val(fields.get(key))
        if val and len(val.strip()) >= 2:
            return True, "name_present", "Holder name present"
            
    for key, field_obj in fields.items():
        if "name" in key.lower():
            val = _extract_field_val(field_obj)
            if val and len(val.strip()) >= 2:
                return True, "name_present", "Holder name present"

    return False, "missing_name", "Holder name missing"

def rule_layout_template_valid(fields: Dict[str, Any], doc_type: str, ocr_result: Dict[str, Any]) -> Tuple[bool, str, str]:
    layout_info = ocr_result.get("layout_result", {})
    if not layout_info:
        return True, "layout_template_valid", "Layout check default pass"
    
    layout_score = layout_info.get("layout_score", 1.0)
    layout_valid = layout_info.get("layout_valid", True)
    if layout_valid and layout_score >= 0.50:
        return True, "layout_template_valid", "Document layout matches official template"
    return False, "invalid_government_layout_template", "Document layout or aspect ratio violates official template"

COMMON_RULES: List[Callable] = [
    rule_doc_number_present,
    rule_document_number_pattern,
    rule_id_checksum_valid,
    rule_dob_plausible_range,
    rule_expiry_date,
    rule_name_present,
    rule_layout_template_valid,
]

RULES_REGISTRY: Dict[str, List[Callable]] = {
    "Aadhaar": COMMON_RULES,
    "PAN Card": COMMON_RULES,
    "Passport": COMMON_RULES,
    "Driving License": COMMON_RULES,
    "Visa": COMMON_RULES,
    "Other": COMMON_RULES,
}

def validate_document(
    ocr_result: Dict[str, Any],
    document_type: str,
    custom_blacklist: Optional[set] = None
) -> Dict[str, Any]:
    """
    Evaluates extracted fields against rules registry.
    Returns validation_pass_rate, expiry_valid, failed_rules, and blacklist_hit status.
    """
    fields = ocr_result.get("fields", {})
    rules = RULES_REGISTRY.get(document_type, COMMON_RULES)

    rules_total = 0
    rules_passed = 0
    failed_rules: List[str] = []
    rule_details: Dict[str, bool] = {}
    expiry_valid = True

    for rule_fn in rules:
        rules_total += 1
        passed, rule_id, detail = rule_fn(fields, document_type, ocr_result)
        rule_details[rule_id] = passed
        if passed:
            rules_passed += 1
        else:
            failed_rules.append(rule_id)
            if rule_id in ("expired_document", "missing_expiry_date"):
                expiry_valid = False

    validation_pass_rate = round(rules_passed / float(rules_total), 4) if rules_total > 0 else 0.0

    # Blacklist check against mock DB
    doc_num_val = fields.get("document_number", {}).get("value", "")
    active_blacklist = custom_blacklist if custom_blacklist is not None else MOCK_BLACKLIST_NUMBERS
    clean_doc_num = re.sub(r'[^A-Z0-9]', '', doc_num_val.upper())
    blacklist_hit = clean_doc_num in active_blacklist if clean_doc_num else False

    if blacklist_hit and "blacklist_hit" not in failed_rules:
        failed_rules.append("blacklist_hit")
        rule_details["blacklist_hit"] = False

    return {
        "validation_pass_rate": validation_pass_rate,
        "expiry_valid": expiry_valid,
        "failed_rules": failed_rules,
        "blacklist_hit": blacklist_hit,
        "rule_details": rule_details
    }
