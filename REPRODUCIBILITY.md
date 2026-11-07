# Reproducibility and method audit

## Scope of this audit

This document distinguishes what is implemented in the repository from what is **not yet independently reproduced or validated**. It is a transparency record, not a claim that all scientific defects have been resolved.

## Steps

1. Use Python 3.11 in a clean virtual environment and install `requirements.txt`.
2. Install spaCy's `en_core_web_sm` language model.
3. Run `python -m pytest -q` for offline record-processing tests.
4. Run `python src/link_prediction.py` to regenerate research outputs. This run is computationally heavier than the offline tests, and is **not currently exercised in CI**.
5. Compare new outputs with `results/` and `figures/`. Record the Python version, package versions, corpus checksum, random seed, execution command and any differences. Do not overwrite previously cited artefacts without preserving them.

## Methodological concerns requiring follow-up

**Temporal interpretation.** The pipeline divides extracted records at 2014 and compares with links observed later, but labels, years, candidate universes and observed nodes require independent verification. Approximate publication years are not clinical-trial completion times.

**Negative sampling and selection.** Test negatives are sampled from intervention–outcome pairs absent from the **entire supplied corpus**, whereas test positives are newly observed links among training-period nodes. This can introduce hindsight and selection bias: information from after the split is used to decide which negative examples are eligible. A strictly prospective benchmark should define its candidate universe and labels without privileged future information when sampling or ranking examples.

**Training examples.** The code trains on training-period graph edges, temporarily removes each positive edge to generate a feature vector, then restores it. The independently sampled training negatives use graph features of non-edges. A reviewer should check whether the resulting training protocol is comparable to the intended future-link-prediction task.

**Bootstrap confidence intervals.** The original `bci` implementation previously drew separate resamples for the class-presence condition and the metric calculation. This has been corrected in this audit branch to reuse a single index sample per replicate. Confidence intervals still do not account for uncertainty from corpus selection, NLP extraction or training-graph construction.

**Stochastic reproducibility.** Set-to-list iteration can vary with `PYTHONHASHSEED` and sample selection. Artefacts should be regenerated under a recorded environment. The audit branch sorts graph-edge and negative-candidate collections for stable sampling.

**NLP evaluation.** No verified manually annotated PICO extraction evaluation is included. Regex mention detection may mistake discussions, comparators and intervention/outcome roles.

**Statistical claims.** The previously reported AUC-ROC of 0.932 and any relative improvement figures must be treated as historical outputs pending an independent end-to-end rerun with reviewed evaluation design. Bootstrap confidence limits and top-ranked candidates are likewise provisional.

**Clinical interpretation.** A predicted missing link is not a verified literature gap and must not be translated into treatment guidance or research-priority recommendations without systematic evidence checking and subject-matter review.

## Evidence for public contribution

The public repository, source code, MIT licence, dated commits and reproducibility instructions substantiate **availability and work undertaken**. They do not establish independent adoption, contributions, downloads, peer review or community benefit. Preserve genuine outside issues, code reviews, contributor PRs, workshop records or documented reuse if they arise.
