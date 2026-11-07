# EvidenceGraphLP

**Exploratory NLP and bipartite-graph link prediction for mapping intervention–outcome evidence in sleep research**

[![MIT Licence](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

EvidenceGraphLP is a public **research prototype** for extracting intervention and outcome mentions from a sleep-related PubMed abstract corpus, constructing an intervention–outcome graph and exploring which previously unobserved links appear in a later period.

> **Research-only limitations:** The system is not a validated evidence-gap detector, medical recommendation engine or systematic-review substitute. A missing or low-ranked link in this corpus does **not** establish that an intervention–outcome relationship has never been studied. Predicting future research activity is not the same as judging clinical importance.

## Repository contents

- `data/sleep_corpus.json`: bundled, domain-specific abstract corpus; review provenance and filtering before reuse.
- `src/link_prediction.py`: research pipeline for extraction, temporal graph construction, model comparison and visualisations.
- `src/fetch_pubmed.py`: optional PubMed retrieval script requiring network access and user-configured NCBI contact information.
- `src/pico_extraction.py`: **legacy experimental extraction script**, not part of the primary reproducible pipeline.
- `tests/test_fetch_pubmed.py`: offline unit tests for PubMed record processing.
- `results/` and `figures/`: checked-in research artefacts; **not verified to have been generated from the present code and dependencies without rerunning the pipeline**.

## Install and verify

Python 3.11 is the documented reference interpreter. Install the spaCy English model separately:

```bash
git clone https://github.com/Rifa-111/EvidenceGraphLP.git
cd EvidenceGraphLP
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m pytest -q
```

Run the primary pipeline from the repository root:

```bash
python src/link_prediction.py
```

It reads `data/sleep_corpus.json` and writes generated files under `results/` and `figures/`. The pipeline is not yet tested end-to-end in continuous integration. Its outputs can differ from committed artefacts because of dependency versions, sampling choices and methodological limitations.

To inspect the saved outputs without rerunning, view the CSV files and images already in the repository.

### Docker

```bash
docker build -t evidencegraphlp .
docker run --rm evidencegraphlp
```

To **retain** outputs locally, mount writable `results/` and `figures/` folders; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Docker Compose is not configured in this repository.

## Research approach

The current script uses regex dictionaries and spaCy to identify intervention and outcome mentions, constructs a bipartite graph, uses a year threshold of **2014** and trains conventional link-prediction classifiers on graph-derived features. The project includes ablations, sampled-negative comparisons and descriptive plots.

### Reported experimental results — not independently verified here

An earlier repository README reported a gradient-boosting temporal AUC-ROC of **0.932**, a degree baseline AUC of **0.622** and a topology-only AUC of **0.876**. Those values are **historical reported figures**, not freshly reproduced results or independent clinical validation. The evaluation implementation requires further methodological review before claiming a robust prospective prediction benchmark. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

In particular, evaluation positives and negatives are selected under different constraints, the graph is constructed from heuristic text extraction, year metadata may be approximated and the original bootstrap confidence-interval implementation sampled labels and predictions inconsistently. These issues could affect headline metrics and uncertainty estimates.

## Data and scientific limits

- The corpus is a sleep-disorder sample, not a comprehensive register of all clinical evidence.
- The extraction is dictionary-based with heuristic dependency parsing; accuracy against manually annotated gold-standard PICO spans is **not established** here.
- The pipeline uses publication/approximate-year metadata, not independently verified trial completion timestamps.
- The output ranks candidate links by **learned research-co-occurrence patterns**, not the medical benefit, novelty or urgency of research.
- Retrieved PubMed content must be checked against its source terms, permitted reuse and provenance.
- No third-party adoption, external validation or community impact is claimed simply because the repository is public.

## Contributing and citing

Constructive reproducibility reports, documentation fixes and independent method reviews are welcome via [GitHub Issues](https://github.com/Rifa-111/EvidenceGraphLP/issues). See [CONTRIBUTING.md](CONTRIBUTING.md).

If citing this software, use the repository URL, version or commit SHA and access date. The code repository does **not** independently verify any journal submission, acceptance or publication. Cite a manuscript only with independently verified bibliographic status.

## Licence

Code is made available under the [MIT licence](LICENSE). The software licence does not automatically grant unrestricted reuse rights for third-party source abstracts or embedded figures.
