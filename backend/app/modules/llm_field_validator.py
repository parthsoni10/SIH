import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from app.config import settings

logger = logging.getLogger(__name__)

HAS_GEMINI = False
try:
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

DOCUMENT_SCHEMAS: Dict[str, List[str]] = {
    "Aadhaar": ["full_name", "dob", "gender", "document_number", "address"],
    "PAN Card": ["full_name", "father_name", "dob", "document_number"],
    "Passport": ["surname", "given_names", "dob", "expiry_date", "document_number", "nationality", "place_of_birth"],
    "Driving License": ["full_name", "dob", "issue_date", "expiry_date", "document_number", "address"],
    "Visa": ["full_name", "passport_number", "visa_number", "issue_date", "expiry_date", "entries"],
    "Other": ["full_name", "dob", "document_number", "issue_date", "expiry_date"]
}

PREFERRED_GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]


def _call_gemini_once(api_key: str, prompt: str, timeout_sec: float = 30.0) -> Optional[str]:
    """
    Single internal helper that calls Gemini API with model fallback.
    This is the ONLY place in the entire project where an LLM call is made.
    """
    def _execute():
        try:
            import time as _time
            from google import genai
            client = genai.Client(api_key=api_key)
            for model_name in PREFERRED_GEMINI_MODELS:
                try:
                    t0 = _time.time()
                    logger.info(f"[Unified LLM] Trying model '{model_name}'...")
                    res = client.models.generate_content(model=model_name, contents=prompt)
                    elapsed = int((_time.time() - t0) * 1000)
                    if res and hasattr(res, "text") and res.text:
                        logger.info(f"[Unified LLM] Model '{model_name}' responded in {elapsed}ms")
                        return res.text.strip()
                    logger.warning(f"[Unified LLM] Model '{model_name}' returned empty after {elapsed}ms")
                except Exception as e:
                    err_msg = str(e)
                    logger.warning(f"[Unified LLM] Model '{model_name}' failed: {err_msg[:120]}")
                    if any(k in err_msg for k in ["API_KEY_INVALID", "API key not valid", "400", "403", "UNAUTHENTICATED"]):
                        logger.warning("[Unified LLM] Invalid/Unauthorized API key — skipping further model retries.")
                        break
                    continue
        except Exception as e:
            logger.warning(f"[Unified LLM] Client init error: {str(e)[:90]}")
        return None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute)
        return future.result(timeout=timeout_sec)


def run_unified_llm_analysis(
    raw_text_lines: List[str],
    document_type: str,
    risk_score: int = 0,
    prediction: str = "genuine",
    tampering_score: float = 0.0,
    face_match_score: Optional[float] = None,
    failed_rules: Optional[List[str]] = None,
    blacklist_hit: bool = False,
    preprocessing_warnings: Optional[List[str]] = None,
    tampering_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    THE SINGLE LLM CALL for the entire project.

    Takes raw OCR text lines, document type, and full pipeline context.
    Returns a unified result with:
      - extracted_fields
      - mismatches_or_anomalies
      - schema_matched
      - officer_explanation
    """
    empty_result = {
        "llm_available": False,
        "extracted_fields": {},
        "mismatches_or_anomalies": [],
        "schema_matched": False,
        "raw_llm_response": "",
        "officer_explanation": ""
    }

    if not raw_text_lines:
        return empty_result

    api_key = settings.GEMINI_API_KEY or settings.MISTRAL_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    if not api_key or not HAS_GEMINI:
        logger.warning("Gemini API key or google-genai library missing. Unified LLM analysis skipped.")
        return empty_result

    expected_schema = DOCUMENT_SCHEMAS.get(document_type, DOCUMENT_SCHEMAS["Other"])
    joined_ocr_text = "\n".join(raw_text_lines)
    face_str = f"{int(face_match_score * 100)}%" if face_match_score is not None else "N/A"

    prompt = f"""You are an expert AI document validation engine for border security and identity screening.
Analyze the following raw OCR text extracted from an identity document image.

Document Type: {document_type}
Expected Standard Fields: {expected_schema}

RAW OCR TEXT LINES:
---
{joined_ocr_text}
---

PIPELINE CONTEXT:
- Risk Score: {risk_score}/100 ({prediction.upper()})
- Tampering Score: {tampering_score}
- Face Match Similarity: {face_str}
- Failed Validation Rules: {failed_rules or []}
- Blacklist Match: {blacklist_hit}
- Preprocessing Warnings: {preprocessing_warnings or []}
- Tampering Flags: {tampering_flags or []}

TASKS:
1. Extract values for all expected fields from the raw OCR text.
2. Normalize date formats to DD/MM/YYYY where possible.
3. Clean whitespace and remove extraneous noise or prefixes (like 'DOB:', 'No:').
4. Verify if extracted values match expected document formatting rules:
   - Aadhaar: 12 digit number
   - PAN Card: 10 character alphanumeric (5 letters, 4 numbers, 1 letter)
   - Passport: 8 character alphanumeric (usually 1 letter + 7 digits)
   - Driving License: State code + numbers
5. Flag any suspicious patterns, typos, character substitutions (e.g. 'O' vs '0', 'I' vs '1'), or layout inconsistencies.
6. Based on the pipeline context, write EXACTLY ONE concise operational sentence (under 28 words) explaining why this document was flagged or verified, suitable for a border security officer.

Respond STRICTLY with a valid JSON object matching this schema:
{{
  "extracted_fields": {{
    "field_name": {{
      "value": "extracted string or null",
      "confidence": float_between_0_and_1,
      "status": "valid" | "suspicious" | "unclear" | "missing"
    }}
  }},
  "mismatches_or_anomalies": [
    "string description of any anomaly, typo, format failure, or inconsistency"
  ],
  "schema_matched": true_if_key_required_fields_present_and_valid,
  "officer_explanation": "One concise sentence for the screening officer."
}}

Return ONLY raw JSON, no markdown formatting tags.
"""

    try:
        logger.info("Running unified LLM analysis (single Gemini call)...")
        raw_text = _call_gemini_once(api_key, prompt, timeout_sec=30.0)
        if not raw_text:
            return empty_result

        # Clean markdown code block wrappers if present
        cleaned_text = re.sub(r'^```(?:json)?\s*', '', raw_text, flags=re.MULTILINE)
        cleaned_text = re.sub(r'\s*```$', '', cleaned_text, flags=re.MULTILINE).strip()

        parsed = json.loads(cleaned_text)
        logger.info("Unified LLM analysis completed successfully.")

        # Extract and truncate officer explanation if needed
        officer_explanation = parsed.get("officer_explanation", "")
        if officer_explanation:
            words = officer_explanation.split()
            if len(words) > 30:
                officer_explanation = " ".join(words[:28]) + "..."

        return {
            "llm_available": True,
            "extracted_fields": parsed.get("extracted_fields", {}),
            "mismatches_or_anomalies": parsed.get("mismatches_or_anomalies", []),
            "schema_matched": parsed.get("schema_matched", False),
            "raw_llm_response": raw_text,
            "officer_explanation": officer_explanation
        }

    except (FuturesTimeoutError, TimeoutError) as e:
        logger.error(f"Unified LLM analysis timed out: {str(e)}")
        return {
            "llm_available": False,
            "extracted_fields": {},
            "mismatches_or_anomalies": ["LLM analysis timed out — using regex extraction only."],
            "schema_matched": False,
            "raw_llm_response": "",
            "officer_explanation": ""
        }
    except Exception as e:
        logger.error(f"Unified LLM analysis error: {str(e)}")
        return {
            "llm_available": False,
            "extracted_fields": {},
            "mismatches_or_anomalies": [f"LLM processing error: {str(e)}"],
            "schema_matched": False,
            "raw_llm_response": "",
            "officer_explanation": ""
        }


def merge_llm_with_regex(regex_fields: Dict[str, Dict[str, Any]], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Combines regex-extracted fields with LLM validation results.
    LLM results refine or add missing fields while regex acts as anchor.
    MRZ values (if present upstream) take ultimate precedence.
    """
    combined = dict(regex_fields)

    if not llm_result or not llm_result.get("llm_available"):
        return combined

    llm_fields = llm_result.get("extracted_fields", {})
    for field_name, item in llm_fields.items():
        if isinstance(item, dict) and item.get("value"):
            val = str(item["value"]).strip()
            conf = float(item.get("confidence", 0.85))
            status = item.get("status", "valid")

            if field_name not in combined:
                combined[field_name] = {"value": val, "confidence": conf, "llm_status": status}
            else:
                existing_val = combined[field_name].get("value", "")
                if not existing_val or conf > combined[field_name].get("confidence", 0.0):
                    combined[field_name]["value"] = val
                    combined[field_name]["confidence"] = conf
                combined[field_name]["llm_status"] = status

    return combined
