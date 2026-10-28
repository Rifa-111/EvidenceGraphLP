"""
Fetch PubMed abstracts about sleep disorder interventions.

Run with:
    python src/fetch_pubmed.py

The resulting corpus is written to sleep_corpus_300.json in the current
working directory.
"""

import json
import time
from collections import Counter

try:
    from Bio import Entrez, Medline
except ImportError:
    Entrez = None
    Medline = None


TARGET = 350
BATCH = 50
OUTPUT_FILE = "sleep_corpus_300.json"

QUERIES = [
    '("insomnia" OR "sleep disorder") AND "randomized" AND ("cognitive behavioral therapy" OR "CBT-I") AND ("sleep quality" OR "sleep onset" OR "insomnia severity")',
    '("insomnia" OR "sleep disorder") AND "randomized" AND "melatonin" AND ("sleep onset latency" OR "total sleep time" OR "sleep efficiency")',
    '("insomnia") AND "randomized controlled trial" AND ("digital" OR "internet" OR "app" OR "online") AND ("CBT" OR "cognitive behavioral")',
    '("insomnia" OR "sleep") AND "randomized" AND ("sleep restriction" OR "stimulus control" OR "sleep hygiene") AND "efficacy"',
    '("insomnia") AND "meta-analysis" AND ("CBT-I" OR "cognitive behavioral therapy for insomnia")',
    '("insomnia" OR "sleep disorder") AND "randomized" AND ("ramelteon" OR "suvorexant" OR "lemborexant" OR "zolpidem" OR "eszopiclone")',
    '("insomnia") AND "randomized" AND ("exercise" OR "physical activity") AND ("sleep quality" OR "sleep onset")',
    '("insomnia" OR "sleep") AND "randomized" AND ("light therapy" OR "bright light" OR "light exposure") AND "circadian"',
    '("insomnia") AND "meta-analysis" AND "melatonin" AND ("sleep" OR "insomnia")',
    '("sleep disorder" OR "insomnia") AND "randomized" AND ("mindfulness" OR "meditation" OR "relaxation" OR "yoga") AND "sleep"',
    '("insomnia") AND "randomized" AND ("older adults" OR "elderly" OR "geriatric") AND ("CBT" OR "sleep")',
    '("insomnia") AND "randomized" AND ("adolescent" OR "children" OR "pediatric") AND ("melatonin" OR "CBT")',
    '("insomnia") AND "randomized" AND ("acupuncture" OR "valerian" OR "herbal" OR "supplement") AND "sleep"',
    '("insomnia" OR "sleep disorder") AND "randomized" AND ("comorbid" OR "depression" OR "anxiety" OR "PTSD" OR "cancer") AND "sleep"',
]


def classify_study_type(title, abstract, publication_types):
    """Classify an article using PubMed publication types and its text."""
    types_text = " ".join(publication_types).lower()
    if "meta-analysis" in types_text:
        return "meta-analysis"
    if "systematic review" in types_text or "review" in types_text:
        return "review"
    if "randomized controlled trial" in types_text or "clinical trial" in types_text:
        return "RCT"
    if "observational" in types_text or "cohort" in types_text:
        return "cohort"

    text_lower = (title + " " + abstract).lower()
    if "randomized" in text_lower or "randomised" in text_lower:
        return "RCT"
    if "meta-analysis" in text_lower:
        return "meta-analysis"
    if "systematic review" in text_lower:
        return "review"
    return "other"


def parse_publication_year(year_text):
    """Return the leading four-digit year, or 2020 when it is unavailable."""
    try:
        return int(year_text[:4])
    except (TypeError, ValueError, IndexError):
        return 2020


def make_abstract_record(record):
    """Convert a parsed PubMed record, skipping missing or stub abstracts."""
    abstract = record.get("AB", "")
    if not abstract or len(abstract) < 100:
        return None

    title = record.get("TI", "")
    return {
        "pmid": record.get("PMID", ""),
        "title": title,
        "year": parse_publication_year(record.get("DP", "2000")),
        "abstract": abstract,
        "study_type": classify_study_type(title, abstract, record.get("PT", [])),
    }


def deduplicate_abstracts(abstracts):
    """Keep the first occurrence of each PMID while preserving input order."""
    seen = set()
    unique = []
    for abstract in abstracts:
        if abstract["pmid"] not in seen:
            seen.add(abstract["pmid"])
            unique.append(abstract)
    return unique


def search_pubmed(entrez_client=None, sleep_fn=time.sleep):
    """Search PubMed and return unique PMIDs from the configured queries."""
    client = entrez_client or Entrez
    if client is None:
        raise RuntimeError("Biopython is required. Install it with: pip install biopython")

    all_pmids = set()
    print(f"Searching PubMed with {len(QUERIES)} queries...")
    for index, query in enumerate(QUERIES, start=1):
        try:
            handle = client.esearch(
                db="pubmed", term=query, retmax=80, sort="relevance"
            )
            try:
                results = client.read(handle)
            finally:
                handle.close()
            pmids = results.get("IdList", [])
            all_pmids.update(pmids)
            print(
                f"  Query {index}/{len(QUERIES)}: {len(pmids)} hits "
                f"(total unique: {len(all_pmids)})"
            )
            sleep_fn(0.4)
        except Exception as error:
            print(f"  Query {index} failed: {error}")

    return list(all_pmids)[:TARGET]


def fetch_abstracts(
    pmids, entrez_client=None, medline_parser=None, batch_size=BATCH, sleep_fn=time.sleep
):
    """Fetch and parse abstracts for PMIDs in batches."""
    client = entrez_client or Entrez
    parser = medline_parser or Medline
    if client is None or parser is None:
        raise RuntimeError("Biopython is required. Install it with: pip install biopython")

    abstracts = []
    print(f"\nFetching {len(pmids)} abstracts in batches of {batch_size}...")
    for start in range(0, len(pmids), batch_size):
        batch = pmids[start : start + batch_size]
        try:
            handle = client.efetch(
                db="pubmed", id=batch, rettype="medline", retmode="text"
            )
            try:
                records = list(parser.parse(handle))
            finally:
                handle.close()

            for record in records:
                abstract = make_abstract_record(record)
                if abstract is not None:
                    abstracts.append(abstract)

            fetched = min(start + batch_size, len(pmids))
            print(
                f"  Fetched {fetched}/{len(pmids)} — "
                f"{len(abstracts)} with abstracts so far"
            )
            sleep_fn(0.5)
        except Exception as error:
            print(f"  Batch starting at {start} failed: {error}")

    return abstracts


def main(output_file=OUTPUT_FILE):
    """Fetch abstracts from PubMed and save the deduplicated corpus."""
    if Entrez is None or Medline is None:
        print("Install biopython first: pip install biopython")
        return 1

    Entrez.email = "your.email@example.com"
    pmids = search_pubmed()
    print(f"\nTotal unique PMIDs: {len(pmids)}")
    abstracts = deduplicate_abstracts(fetch_abstracts(pmids))

    with open(output_file, "w", encoding="utf-8") as output:
        json.dump(abstracts, output, indent=2)

    types = Counter(abstract["study_type"] for abstract in abstracts)
    years = [abstract["year"] for abstract in abstracts]
    year_range = f"{min(years)}-{max(years)}" if years else "unavailable"
    print(
        f"\n{'=' * 60}\n"
        "CORPUS COMPLETE\n"
        f"{'=' * 60}\n"
        f"  Abstracts saved: {len(abstracts)}\n"
        f"  Output file:     {output_file}\n"
        f"  Study types:     {dict(types)}\n"
        f"  Year range:      {year_range}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
