import re
from typing import Dict, Any, List, Optional, Tuple

WEIGHTS = [7, 3, 1]

def compute_mrz_check_digit(data: str) -> int:
    """Computes ICAO 9303 check digit using weights [7, 3, 1]."""
    total = 0
    for i, char in enumerate(data):
        if char == '<':
            val = 0
        elif char.isdigit():
            val = int(char)
        elif 'A' <= char <= 'Z':
            val = ord(char) - ord('A') + 10
        else:
            val = 0
        total += val * WEIGHTS[i % 3]
    return total % 10

def parse_mrz_lines(lines: List[str]) -> Dict[str, Any]:
    """
    Detects and parses 2-line (TD3 / Passport) or 3-line (TD1 / Visa / ID) MRZ lines.
    Returns parsed fields and checksum validation status.
    """
    clean_lines = [re.sub(r'[^A-Z0-9<]', '', line.upper().strip()) for line in lines if len(line.strip()) >= 28]

    if len(clean_lines) >= 2 and len(clean_lines[0]) == 44 and len(clean_lines[1]) == 44:
        return parse_td3_mrz(clean_lines[0], clean_lines[1])
    elif len(clean_lines) >= 3 and all(len(l) == 30 for l in clean_lines[:3]):
        return parse_td1_mrz(clean_lines[0], clean_lines[1], clean_lines[2])
    
    # Attempt fuzzy length matching if OCR dropped a trailing '<'
    td3_candidate_1 = clean_lines[0] if len(clean_lines) > 0 else ""
    td3_candidate_2 = clean_lines[1] if len(clean_lines) > 1 else ""

    if len(td3_candidate_1) >= 40 and len(td3_candidate_2) >= 40:
        line1 = td3_candidate_1.ljust(44, '<')[:44]
        line2 = td3_candidate_2.ljust(44, '<')[:44]
        return parse_td3_mrz(line1, line2)

    return {
        "present": False,
        "raw_lines": lines,
        "checksum_valid": False,
        "fields": {}
    }

def parse_td3_mrz(line1: str, line2: str) -> Dict[str, Any]:
    """Parses 2-line 44-character TD3 Passport MRZ."""
    doc_type = line1[0:2].replace('<', '')
    issuing_state = line1[2:5].replace('<', '')
    name_part = line1[5:44]
    
    name_split = name_part.split('<<')
    surname = name_split[0].replace('<', ' ').strip()
    given_names = name_split[1].replace('<', ' ').strip() if len(name_split) > 1 else ""
    full_name = f"{given_names} {surname}".strip()

    doc_num = line2[0:9]
    doc_num_chk = line2[9]
    nationality = line2[10:13].replace('<', '')
    dob = line2[13:19]
    dob_chk = line2[19]
    sex = line2[20]
    expiry = line2[21:27]
    expiry_chk = line2[27]

    # Check digit validations
    doc_num_valid = doc_num_chk.isdigit() and compute_mrz_check_digit(doc_num) == int(doc_num_chk)
    dob_valid = dob_chk.isdigit() and compute_mrz_check_digit(dob) == int(dob_chk)
    expiry_valid = expiry_chk.isdigit() and compute_mrz_check_digit(expiry) == int(expiry_chk)

    checksum_valid = doc_num_valid and dob_valid and expiry_valid

    # Format dates to YYYY-MM-DD (assume 19xx for high DOB years, 20xx for low)
    formatted_dob = format_mrz_date(dob, is_dob=True)
    formatted_expiry = format_mrz_date(expiry, is_dob=False)

    return {
        "present": True,
        "raw_lines": [line1, line2],
        "checksum_valid": checksum_valid,
        "fields": {
            "document_type": doc_type,
            "issuing_state": issuing_state,
            "name": full_name,
            "document_number": doc_num.replace('<', ''),
            "nationality": nationality,
            "dob": formatted_dob,
            "sex": sex,
            "expiry_date": formatted_expiry,
        },
        "check_digits": {
            "document_number_valid": doc_num_valid,
            "dob_valid": dob_valid,
            "expiry_valid": expiry_valid,
        }
    }

def parse_td1_mrz(line1: str, line2: str, line3: str) -> Dict[str, Any]:
    """Parses 3-line 30-character TD1 ID/Visa MRZ."""
    doc_num = line1[5:14]
    doc_num_chk = line1[14]
    
    dob = line2[0:6]
    dob_chk = line2[6]
    sex = line2[7]
    expiry = line2[8:14]
    expiry_chk = line2[14]

    doc_num_valid = doc_num_chk.isdigit() and compute_mrz_check_digit(doc_num) == int(doc_num_chk)
    dob_valid = dob_chk.isdigit() and compute_mrz_check_digit(dob) == int(dob_chk)
    expiry_valid = expiry_chk.isdigit() and compute_mrz_check_digit(expiry) == int(expiry_chk)

    checksum_valid = doc_num_valid and dob_valid and expiry_valid

    name_part = line3.replace('<', ' ').strip()

    return {
        "present": True,
        "raw_lines": [line1, line2, line3],
        "checksum_valid": checksum_valid,
        "fields": {
            "document_number": doc_num.replace('<', ''),
            "dob": format_mrz_date(dob, is_dob=True),
            "sex": sex,
            "expiry_date": format_mrz_date(expiry, is_dob=False),
            "name": name_part
        },
        "check_digits": {
            "document_number_valid": doc_num_valid,
            "dob_valid": dob_valid,
            "expiry_valid": expiry_valid,
        }
    }

def format_mrz_date(yymmdd: str, is_dob: bool = False) -> str:
    """Converts YYMMDD string to YYYY-MM-DD format."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return yymmdd
    yy = int(yymmdd[0:2])
    mm = yymmdd[2:4]
    dd = yymmdd[4:6]
    
    if is_dob:
        # If DOB, cutoff around 26: 00-26 -> 2000-2026, 27-99 -> 1927-1999
        century = "20" if yy <= 26 else "19"
    else:
        # If expiry date, cutoff around 70: 00-70 -> 2000-2070
        century = "20"
    
    return f"{century}{yy:02d}-{mm}-{dd}"
