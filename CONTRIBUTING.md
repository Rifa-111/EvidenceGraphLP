# Contributing to EvidenceGraphLP

EvidenceGraphLP is an experimental research repository. Contributions are welcome, particularly reproducibility checks, tests, clearer provenance, bug reports and independent method critiques.

## Getting started

```bash
git clone https://github.com/Rifa-111/EvidenceGraphLP.git
cd EvidenceGraphLP
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pytest -q
```

Please open an issue before substantial algorithmic changes. For pull requests, explain the behaviour changed, add focused tests where possible and distinguish reproduced numbers from historical values.

## Useful contribution areas

- Verify corpus provenance, PubMed IDs and year metadata against primary sources.
- Evaluate PICO extraction against manually reviewed annotations.
- Design strictly prospective negative sampling and out-of-time benchmarks.
- Audit edge-direction handling, calibration and confidence interval calculations.
- Improve offline unit tests and end-to-end reproducibility.

Do not upload identifiable patient information or restricted datasets. Code and output contributions must respect third-party text and image reuse rights. Communicate responsibly: model scores are **not clinical evidence recommendations**.

We welcome independent, critical feedback; no particular adoption or acceptance is presumed.
