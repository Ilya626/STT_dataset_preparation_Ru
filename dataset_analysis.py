import argparse
import json
from pathlib import Path

from jiwer import wer


def main():
    parser = argparse.ArgumentParser(description="Compute WER and SER for predictions")
    parser.add_argument("predictions", help="JSONL with fields: audio, ref, hyp")
    args = parser.parse_args()

    refs = []
    hyps = []
    total = 0
    wrong = 0
    with Path(args.predictions).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref = row.get("ref") or row.get("text") or ""
            hyp = row.get("hyp") or row.get("pred_text") or ""
            refs.append(ref)
            hyps.append(hyp)
            total += 1
            if ref.strip() != hyp.strip():
                wrong += 1
    w = wer(refs, hyps) if refs else 0.0
    ser = wrong / total if total else 0.0
    print(f"WER: {w:.4f}")
    print(f"SER: {ser:.4f}")


if __name__ == "__main__":
    main()
