"""Agreement between the LLM judge and human labels. Judge scores are only trusted after this check."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence

from contract_rag.schemas import EvalScores

# Landis & Koch (1977) bands for reading kappa
KAPPA_BANDS = [(0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]


def percent_agreement(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Agreement corrected for the agreement two raters would reach by chance, given their label rates."""
    n = len(a)
    observed = percent_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def kappa_band(kappa: float) -> str:
    band = "almost perfect"
    for upper, name in reversed(KAPPA_BANDS):
        if kappa <= upper:
            band = name
    return band


def judge_labels(scores: EvalScores) -> dict[str, str | None]:
    """Map judge scores onto the labels a human gives: faithful yes/no, correct yes/partial/no."""
    faithful = None if scores.faithfulness is None else ("yes" if scores.faithfulness == 1.0 else "no")
    correct = {1.0: "yes", 0.5: "partial", 0.0: "no"}.get(scores.answer_correctness)  # type: ignore[arg-type]
    return {"faithful": faithful, "correct": correct}


def agreement_report(
    human: dict[str, dict[str, str]],
    judged: dict[str, EvalScores],
    fields: Sequence[str] = ("faithful", "correct"),
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name in fields:
        pairs = [
            (human[qid][name], judge_labels(judged[qid])[name])
            for qid in sorted(human.keys() & judged.keys())
            if human[qid].get(name) and judge_labels(judged[qid])[name] is not None
        ]
        if not pairs:
            report[name] = {"n": 0}
            continue
        h, j = zip(*pairs, strict=True)
        kappa = cohen_kappa(h, j)
        report[name] = {
            "n": len(pairs),
            "agreement": round(percent_agreement(h, j), 3),
            "kappa": round(kappa, 3),
            "band": kappa_band(kappa),
            "confusion (human -> judge)": dict(Counter(f"{x} -> {y}" for x, y in pairs)),
        }
    return report
