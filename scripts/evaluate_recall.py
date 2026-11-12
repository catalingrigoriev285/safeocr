"""Run the full pipeline over every generated sample and report recall.

Recall = PII entities covered by black boxes / PII entities that exist (from ground truth).

    python scripts/evaluate_recall.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from safeocr.service import PipelineError, SafeOCRService  # noqa: E402


def main() -> int:
    svc = SafeOCRService()
    samples = sorted(svc.settings.samples_dir.glob("*.png"))
    if not samples:
        print("No samples found. Run: python scripts/generate_samples.py")
        return 1

    totals = defaultdict(lambda: {"covered": 0, "total": 0})
    passed = 0
    print(f"OCR backend: {svc.settings.ocr.backend}\n")
    print(f"{'document':<12}{'detected':>9}{'covered':>9}{'total':>7}{'recall':>8}  verify")
    for img in samples:
        try:
            doc = svc.ocr_document(str(img))["document_id"]
            det = svc.detect_pii(doc)
            res = svc.redact_image(doc)["result_id"]
            ver = svc.verify_redaction(res)
        except PipelineError as exc:
            print(f"{img.stem:<12} error: {exc}")
            continue
        rec = svc.recall(res)
        passed += ver["passed"]
        if rec:
            for t, s in rec["per_type"].items():
                totals[t]["covered"] += s["covered"]
                totals[t]["total"] += s["total"]
            print(f"{img.stem:<12}{det['total']:>9}{rec['covered']:>9}{rec['total']:>7}{rec['recall']:>8.2%}  "
                  f"{'PASS' if ver['passed'] else 'FAIL ' + str(ver['remaining_counts'])}")

    print("\nRecall per PII type")
    cov = tot = 0
    for t, s in sorted(totals.items()):
        cov, tot = cov + s["covered"], tot + s["total"]
        print(f"  {t:<10} {s['covered']:>3}/{s['total']:<3} {s['covered'] / s['total']:.2%}")
    if tot:
        print(f"\nOverall recall: {cov}/{tot} = {cov / tot:.2%}")
    print(f"Verification passed: {passed}/{len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
