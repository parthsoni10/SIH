import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional
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

def validate_document(
    ocr_result: Dict[str, Any],
    document_type: str,
    custom_blacklist: Optional[set] = None
) -> Dict[str, Any]:
    """
    Evaluates extracted fields against document validation rules.
    Returns validation_pass_rate, expiry_valid, failed_rules, and blacklist_hit status.
    """
    fields = ocr_result.get("fields", {})
    id_checksum_valid = ocr_result.get("id_checksum_valid", False)
    
    rules_total = 0
    rules_passed = 0
    failed_rules: List[str] = []
    rule_details: Dict[str, bool] = {}

    today = date.today()

    # Rule 1: Document Number Presence & Pattern
    doc_num_val = fields.get("document_number", {}).get("value", "")
    rules_total += 1
    if doc_num_val:
        rule_details["doc_number_present"] = True
        rules_passed += 1
    else:
        rule_details["doc_number_present"] = False
        failed_rules.append("missing_document_number")

    # Rule 2: ID Checksum / Format Check
    rules_total += 1
    if id_checksum_valid:
        rule_details["id_checksum_valid"] = True
        rules_passed += 1
    else:
        rule_details["id_checksum_valid"] = False
        failed_rules.append("invalid_id_checksum")

    # Rule 3: Date of Birth Presence & Logical Check
    dob_val = fields.get("dob", {}).get("value", "")
    rules_total += 1
    dob_date = parse_date_safely(dob_val) if dob_val else None
    
    if dob_date:
        if dob_date < today:
            rule_details["dob_valid"] = True
            rules_passed += 1
        else:
            rule_details["dob_valid"] = False
            failed_rules.append("future_dob")
    else:
        rule_details["dob_valid"] = False
        failed_rules.append("missing_or_invalid_dob")

    # Rule 4: Expiry Date Check (required for Passport & Visa, optional for others)
    expiry_valid = True
    exp_val = fields.get("expiry_date", {}).get("value", "")
    exp_date = parse_date_safely(exp_val) if exp_val else None

    if document_type in ("Passport", "Visa"):
        rules_total += 1
        if exp_date:
            if exp_date > today:
                expiry_valid = True
                rule_details["expiry_valid"] = True
                rules_passed += 1
            else:
                expiry_valid = False
                rule_details["expiry_valid"] = False
                failed_rules.append("expired_document")
        else:
            expiry_valid = False
            rule_details["expiry_valid"] = False
            failed_rules.append("missing_expiry_date")
    elif exp_date:
        rules_total += 1
        if exp_date > today:
            expiry_valid = True
            rule_details["expiry_valid"] = True
            rules_passed += 1
        else:
            expiry_valid = False
            rule_details["expiry_valid"] = False
            failed_rules.append("expired_document")

    # Rule 5: Name Field Presence
    name_val = fields.get("name", {}).get("value", "")
    rules_total += 1
    if name_val and len(name_val.strip()) >= 2:
        rule_details["name_present"] = True
        rules_passed += 1
    else:
        rule_details["name_present"] = False
        failed_rules.append("missing_name")

    # Pass rate calculation
    validation_pass_rate = round(rules_passed / float(rules_total), 4) if rules_total > 0 else 0.0

    # Blacklist check against mock DB
    active_blacklist = custom_blacklist if custom_blacklist is not None else MOCK_BLACKLIST_NUMBERS
    clean_doc_num = re.sub(r'[^A-Z0-9]', '', doc_num_val.upper())
    blacklist_hit = clean_doc_num in active_blacklist if clean_doc_num else False
    if blacklist_hit:
        failed_rules.append("blacklist_hit")

    return {
        "validation_pass_rate": validation_pass_rate,
        "expiry_valid": expiry_valid,
        "failed_rules": failed_rules,
        "blacklist_hit": blacklist_hit,
        "rule_details": rule_details
    }
