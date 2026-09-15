# SENTINEL V3.3 — Global Document Coverage

Answers "does this work for every document type in the README" honestly:
**before this patch, no** — only Aadhaar and PAN Card were registered.
Everything else silently fell through `get_spec()` returning `None`, which
would have crashed or been mis-handled by anything downstream that assumes a
spec exists. This patch makes all six README types real, and makes the
system fail safely on anything not yet in the README too.

## What's now covered

| Document Type | Spec registered | Checksum (real, verified) | Anchors | QR |
|---|:---:|---|:---:|:---:|
| Aadhaar | ✅ | VERHOEFF | 6, fitted to 1 reference photo | mandatory (back) |
| PAN Card | ✅ | PAN_ENTITY_CHAR | 5, fitted to 1 reference photo | mandatory (front) |
| Passport | ✅ **new** | MRZ_TD3 (ICAO 9303) | 3, **unmeasured placeholders** | optional |
| Visa | ✅ **new** | MRZ_TD1 (ICAO 9303) | 2, **unmeasured placeholders** | optional |
| Driving License | ✅ **new** | DL_STATE_CODE (plausibility only) | 2, **unmeasured placeholders** | optional |
| Other | ✅ **new** | none (by design) | none (by design) | optional |
| *(anything else)* | ✅ **new** — safe fallback | none | none | optional |

Every spec ships `calibrated=False`. The V3.2 rule still applies uniformly:
uncalibrated anchor evidence is visible in the dashboard but cannot solely
gate a decision. Passport/Visa/DL get the *same* protection Aadhaar/PAN got
after your last regression — their anchor boxes are placeholders I estimated
with no reference photo, so they are *more* likely to misfire than Aadhaar's,
not less. Don't calibrate them without real samples.

## Three real gaps this closes, not just spec entries

**1. Checksums were never actually computed.** Every previous version passed
a `checksum_valid: bool` into `field_resolver` from outside — nothing in the
pipeline ever ran Verhoeff, let alone MRZ. `checksum.py` implements all four
algorithms and I validated `MRZ_TD3` against the **official ICAO 9303
worked example** (all four check digits: document number, DOB, expiry,
composite — all pass). Verhoeff and PAN entity-code were re-validated too.

**2. QR mandatoriness was hardcoded to two document types.** `QR_MANDATORY =
{"PAN Card", "Aadhaar"}` meant no other type — including ones you add later —
could ever be checked for a missing/fake QR. It's now read from
`spec.qr_expected_side`, so it's driven by the same declarative spec as
everything else and needs no code change to extend.

**3. A real bug I found while testing this: `mrz_line1`/`mrz_line2` were
declared `required=True` `FieldSpec`s but never populated through the normal
OCR path** — they're supplied via a separate `mrz_lines=` parameter for
checksum purposes only. Left as `required=True`, every single Passport/Visa
verification would have permanently reported them "missing" regardless of
whether a valid MRZ was present. Fixed to `required=False` before shipping.
Caught by actually running the Passport case end-to-end rather than trusting
the dataclass definition — same lesson as the anchor bugs I found earlier
testing genuine vs. altered Aadhaar.

## Verified end-to-end

```
get_spec() coverage:
  Aadhaar / PAN Card / Passport / Visa / Driving License / Other -> real specs
  TotallyUnknownType -> safe generic fallback, no crash

Passport, ICAO 9303 official worked example MRZ:
  checksum: MRZ_TD3  applicable=True  valid=True
  validity_score: 0.975   structurally_valid: True
  full_name -> "Eriksson"   dob -> "12/08/1974"

Other (no anchors, no checksum, by design):
  validity_score: 0.85   structurally_valid: True
  (fallback correctly does NOT get penalised for lacking structure
   it was never declared to have)

Blank/degenerate image against Passport's placeholder anchors:
  anchors report missing (expected -- it's a blank image)
  BUT spec.calibrated=False, so this cannot alone gate ALTERED
  (same protection Aadhaar/PAN already have)
```

## What you still have to do yourself

I want to be precise about what "global logic" does and doesn't mean here,
so you don't oversell this in the demo:

- **The architecture is now genuinely generic** — one spec format, one
  checksum registry, one decision engine, covering all six types with no
  per-type special-casing in the gating logic itself.
- **The Passport/Visa/DL anchor boxes are not measured.** I have no
  reference photos for them. They will behave exactly like Aadhaar/PAN did
  before V3.2's calibration gate — plausible-looking numbers fitted to
  nothing. They're protected from gating decisions by `calibrated=False`,
  but they are not yet *useful* signal. Run `calibrate_anchors.py` (from the
  V3.2 package) against real samples of each type before trusting their
  output for anything beyond dashboard curiosity.
- **DL_STATE_CODE is a plausibility check, not a checksum.** No
  cryptographically verifiable check digit exists nationally for Indian
  driving licenses. Say so if asked — claiming checksum-grade certainty here
  is the kind of overclaim that costs credibility with a panel.
- **Wire the router.** `resolve_structure` now needs `mrz_lines=` for
  Passport/Visa and `checksum_input=` (or a resolved `document_number`) for
  the rest — pass MRZ lines through from your OCR stage instead of only the
  parsed fields.
