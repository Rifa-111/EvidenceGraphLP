"""
fetch_pubmed_sleep.py
=====================
Run this locally to fetch 300+ real PubMed abstracts on sleep disorder interventions.

Requirements:
    pip install biopython

Usage:
    python fetch_pubmed_sleep.py

Output:
    sleep_corpus_300.json  (feed this into sleep_pipeline.py)

Takes about 2-3 minutes. No API key needed for ≤3 requests/second.
"""

import json
import time
import sys

try:
    from Bio import Entrez, Medline
except ImportError:
    print("Install biopython first:  pip install biopython")
    sys.exit(1)

# ── CONFIG ──────────────────────────────────────────────────────
Entrez.email = "your.email@example.com"  # ← CHANGE THIS to your email
TARGET = 350  # fetch extra, will deduplicate
BATCH = 50

# Multiple queries to cover the breadth of sleep interventions
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
    '("insomnia") AND "randomized" AND ("comorbid" OR "depression" OR "anxiety" OR "PTSD" OR "cancer") AND "sleep"',
]

# ── FETCH ───────────────────────────────────────────────────────
all_pmids = set()
print(f"Searching PubMed with {len(QUERIES)} queries...")

for i, query in enumerate(QUERIES):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=80, sort="relevance")
        results = Entrez.read(handle)
        handle.close()
        pmids = results.get("IdList", [])
        all_pmids.update(pmids)
        print(f"  Query {i+1}/{len(QUERIES)}: {len(pmids)} hits (total unique: {len(all_pmids)})")
        time.sleep(0.4)  # respect rate limits
    except Exception as e:
        print(f"  Query {i+1} failed: {e}")

print(f"\nTotal unique PMIDs: {len(all_pmids)}")

# Limit to TARGET
pmid_list = list(all_pmids)[:TARGET]

# ── FETCH ABSTRACTS ─────────────────────────────────────────────
print(f"\nFetching {len(pmid_list)} abstracts in batches of {BATCH}...")
abstracts = []

for start in range(0, len(pmid_list), BATCH):
    batch = pmid_list[start:start + BATCH]
    try:
        handle = Entrez.efetch(db="pubmed", id=batch, rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()

        for rec in records:
            pmid = rec.get("PMID", "")
            title = rec.get("TI", "")
            abstract = rec.get("AB", "")
            year_str = rec.get("DP", "2000")  # e.g., "2023 Jan 15"
            pub_types = rec.get("PT", [])

            if not abstract or len(abstract) < 100:
                continue  # skip empty/stub abstracts

            # Determine study type from MeSH/pub types
            pt_lower = " ".join(pub_types).lower()
            if "meta-analysis" in pt_lower:
                study_type = "meta-analysis"
            elif "systematic review" in pt_lower or "review" in pt_lower:
                study_type = "review"
            elif "randomized controlled trial" in pt_lower or "clinical trial" in pt_lower:
                study_type = "RCT"
            elif "observational" in pt_lower or "cohort" in pt_lower:
                study_type = "cohort"
            else:
                # Infer from abstract text
                text_lower = (title + " " + abstract).lower()
                if "randomized" in text_lower or "randomised" in text_lower:
                    study_type = "RCT"
                elif "meta-analysis" in text_lower:
                    study_type = "meta-analysis"
                elif "systematic review" in text_lower:
                    study_type = "review"
                else:
                    study_type = "other"

            try:
                year = int(year_str[:4])
            except (ValueError, IndexError):
                year = 2020

            abstracts.append({
                "pmid": pmid,
                "title": title,
                "year": year,
                "abstract": abstract,
                "study_type": study_type,
            })

        fetched = min(start + BATCH, len(pmid_list))
        print(f"  Fetched {fetched}/{len(pmid_list)} — {len(abstracts)} with abstracts so far")
        time.sleep(0.5)

    except Exception as e:
        print(f"  Batch starting at {start} failed: {e}")

# ── DEDUPLICATE AND SAVE ────────────────────────────────────────
seen = set()
unique = []
for a in abstracts:
    if a["pmid"] not in seen:
        seen.add(a["pmid"])
        unique.append(a)

output_file = "sleep_corpus_300.json"
with open(output_file, "w") as f:
    json.dump(unique, f, indent=2)

# ── SUMMARY ─────────────────────────────────────────────────────
from collections import Counter
types = Counter(a["study_type"] for a in unique)
years = [a["year"] for a in unique]

print(f"""
{'='*60}
CORPUS COMPLETE
{'='*60}
  Abstracts saved: {len(unique)}
  Output file:     {output_file}
  Study types:     {dict(types)}
  Year range:      {min(years)}-{max(years)}
  
NEXT STEP:
  Copy {output_file} to the same folder as sleep_pipeline.py
  Edit sleep_pipeline.py line 5: change the filename to '{output_file}'
  Run:  python sleep_pipeline.py
""")
