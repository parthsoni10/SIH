# SENTINEL V3.1 — Alteration & Structure Fix Plan

Root-cause analysis of verification #207 (Aadhaar), which returned
`GENUINE · 70% confidence` on a card with an erased government logo, an
unread holder name, a misassigned date, and a missing back side.

---

## Part 1 — Why the altered card was not detected

### 1.1 Your pixel forensics were blind before they started

I measured the paper background on both cards:

```
near-white pixels with EXACTLY zero local variance : 86% (genuine)
                                                     86% (altered)
```

This scan went through a document-scanner app that clamps paper to pure
`255,255,255`. Then it went through WhatsApp, which stripped EXIF and
re-encoded the entire frame uniformly. Both files arrive as 1600×1073 JPEGs
with no EXIF and an identical ICC profile.

That kills four of your detectors at once:

| Detector | Precondition | Status on #207 |
|---|---|---|
| ELA | per-region compression history | destroyed by uniform WhatsApp re-encode |
| Noise inconsistency | a noise floor exists | background variance is exactly 0 |
| Provenance (Family A) | metadata survives | WhatsApp stripped both identically |
| White-point analysis | paper has a natural cast | clamped to pure white |

Your dashboard reported `ELA 2% · Overall tampering 4.3% · LOW TAMPERING`.
That 4.3% was not a measurement. It was four blind modules returning small
reassuring numbers. **This is the real bug**: a cue that cannot measure must
report `available: false`, not a low score. A low score means "I looked and
found nothing"; these modules never looked.

In `alteration_detector.py` all five cues self-disable when their
preconditions fail. On your Aadhaar, three of five disable themselves and say
so — and the verdict is still reached by the two that remain.

### 1.2 What actually works: presence assertions

The erased logo is trivially detectable, just not statistically. It is a
**presence assertion**: the UIDAI mark is mandatory at a fixed position on
every genuine Aadhaar. Measured ink coverage in that region:

```
genuine : 0.2390
altered : 0.0000      <- not "low". zero.
```

No reference image, no training, no threshold tuning. The region is either
inked or it is not. `document_spec.py` declares mandatory anchors per document
type; `anchor_verifier.py` checks them.

This generalises: blanking a photo, whiting out an address, covering a
security band all collapse ink coverage in a declared region.

### 1.3 Two bugs I found in my own cues while testing

Worth reading, because they are the same class of mistake as #207.

**The unwarped fallback padded anchor boxes outward.** That pulled in
neighbouring artwork and diluted the logo signal from 150× down to 3.4× —
enough to break the whole chain. Shrinking *inward* is correct: registration
error still lands inside the anchor core.

**`pasted_region_geometry` fired strongly on the GENUINE card.** It searched
the whole image for axis-aligned edges and clustered them into one box. But a
document is full of legitimate straight lines — the card border, the dashed
cut line, the red rule, and on Aadhaar specifically the genuine white rounded
panel behind the 12-digit number. The cue is now *targeted*: it only asks
whether an already-missing anchor has been covered by a uniform fill. It
explains how artwork disappeared rather than hunting for trouble.

Both bugs produced confident wrong answers from plausible-looking code. Test
every cue against a genuine document, not only against forgeries.

---

## Part 2 — Why the structure checks passed a broken card

#207 raised `Holder Name Missing or Unreadable in OCR`, extracted 2 fields,
and still scored `Document Validity 86% · STRUCTURALLY VALID`. Three separate
defects.

### 2.1 The date role bug

You extracted `dob = 03/01/2014` at 80% confidence. **That is the Issue Date**,
printed vertically down the left edge. The real DOB is 17/08/2005.

A bare `\d{2}/\d{2}/\d{4}` regex takes whichever date OCR emits first, and
vertical text often sorts first because it sits at low x. Age plausibility
cannot save you here — a 12-year-old Aadhaar holder is perfectly legal.

Fix: bind dates to roles by **label proximity using OCR bounding boxes**, with
a geometric rule that a vertically-oriented date is never a DOB. After the
fix both dates resolve correctly and independently.

### 2.2 The name bug

Aadhaar prints the holder name on an **unlabelled line** above the DOB.
Your extractor looked for a `Name:` token that Aadhaar never prints, so it
found nothing. Compounding it, PaddleOCR was running a Latin-only model
against a Gujarati/English card and returned only 8 lines.

Fix: positional fallback — Latin title-case lines that are not template
boilerplate, preferring the line directly above the DOB. Plus enable the
Gujarati/Devanagari OCR models.

### 2.3 The gate bug

The deepest one. A flagged anomaly that does not move the verdict is
decoration. Completeness is now a **multiplier**, not an addend, and missing
required fields hard-cap validity and force MANUAL_REVIEW.

### 2.4 Single-side submission

Aadhaar's address and QR live on the back. A front-only upload can never
satisfy the spec, but #207 reported no gap.

New status: **`INCOMPLETE_SUBMISSION`**, deliberately separate from both
GENUINE and SUSPICIOUS. A front-only card is not suspicious and does not need
an officer's judgement — it needs the back side. Conflating "you haven't shown
me everything" with "this looks forged" is what makes review queues unusable.

---

## Part 3 — Verified results

```
GENUINE Aadhaar
  aadhaar_logo ink=0.2390 (need 0.0225)   all 6 anchors present
  alteration prob=0.000  corroborated=False  cues_available=2/5
  dob -> 17/08/2005 [LABEL_PROXIMITY]   issue_date -> 03/01/2014
  full_name -> Dhruv Prajapati
  VERDICT: INCOMPLETE_SUBMISSION — upload back side

ALTERED Aadhaar (logo erased)
  aadhaar_logo ink=0.0000 (need 0.0225)   <-- MISSING
  alteration prob=0.900  corroborated=True
  strong cues: [anchor_removal, pasted_region_geometry]
  VERDICT: ALTERED [CONCLUSIVE] conf=0.90
  ANCHOR_MISSING_AADHAAR_LOGO
```

Note the genuine card scores **0.000**, not "low" — every cue that could
produce a false positive disabled itself and said why.

Also verified independently: **Verhoeff on 2778 3746 3380 is valid.** Your
checksum PASS was correct. That one was never a bug.

---

## Part 4 — Order of work

**Step 1 — anchors.** Drop in `document_spec.py` + `anchor_verifier.py`. This
alone catches the logo erasure. Measure anchor boxes on *your own* genuine
reference scans; mine are fitted to one 1600×1073 sample.

**Step 2 — deskew first.** Anchor boxes are normalised coordinates. Run
`layout_validator`'s 4-point warp before verification or every box is offset.
Unwarped still works (inset sampling) but at reduced confidence.

**Step 3 — replace tampering scoring.** Swap in `alteration_detector.py`.
Surface `available` per cue in `ForensicCard.jsx`. An officer seeing
"3 of 5 cues disabled: scanner preprocessing" makes a better decision than one
seeing "4.3% LOW TAMPERING".

**Step 4 — field resolver.** Needs OCR **bounding boxes**, not just text. If
your OCR wrapper discards boxes, restore them first — label-proximity binding
is impossible without them. Enable Gujarati + Devanagari models.

**Step 5 — decision gates.** `decision_engine_v31.py` adds the alteration,
completeness and side gates.

**Step 6 — expand the spec.** I wrote Aadhaar and PAN. Add Passport, Visa,
Driving License anchors from your own reference stock.

---

## Part 5 — Things to check on your side

**Duplicate audit record.** You pasted two different uploads that both returned
`Verification Result #207` with an identical timestamp of 10:18:37 AM. Either
you pasted the same result twice, or your endpoint is caching/deduplicating by
content hash and returning a stale record. If it is the latter, that is a
serious audit-integrity bug — two screenings must never share an ID. Check it
before the demo.

**Do not claim you detect all alteration.** You detect *erasure and
overwriting of mandatory artwork*, which is a specific and defensible claim.
An attacker who edits the DOB digits in place, leaving all anchors intact,
will not be caught by anchor verification — that needs the QR payload
cross-check, which on Aadhaar lives on the back side. This is exactly why
`sides_required` matters, and it is a good answer to give a panel.

**PII again.** This is a real Aadhaar with a live number, and you now have
three real identity documents in your test set. Aadhaar numbers are regulated
under the Aadhaar Act and UIDAI guidance discourages open storage or sharing.
Build your test fixtures from synthetic cards, keep these out of version
control, and do not put them in a public demo.
