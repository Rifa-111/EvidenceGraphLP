import json
from types import SimpleNamespace

from src import fetch_pubmed


def test_classify_study_type_prefers_pubmed_publication_types():
    assert (
        fetch_pubmed.classify_study_type(
            "A trial",
            "A randomized study",
            ["Randomized Controlled Trial"],
        )
        == "RCT"
    )


def test_classify_study_type_infers_from_article_text():
    assert fetch_pubmed.classify_study_type(
        "Sleep intervention", "A randomised clinical study", []
    ) == "RCT"
    assert fetch_pubmed.classify_study_type("Sleep", "No relevant terms", []) == "other"


def test_parse_publication_year_uses_fallback_for_invalid_values():
    assert fetch_pubmed.parse_publication_year("2023 Jan 15") == 2023
    assert fetch_pubmed.parse_publication_year("") == 2020
    assert fetch_pubmed.parse_publication_year(None) == 2020


def test_make_abstract_record_converts_pubmed_fields():
    record = fetch_pubmed.make_abstract_record(
        {
            "PMID": "123",
            "TI": "Randomized sleep trial",
            "DP": "2022 Mar",
            "AB": "x" * 100,
            "PT": ["Clinical Trial"],
        }
    )

    assert record == {
        "pmid": "123",
        "title": "Randomized sleep trial",
        "year": 2022,
        "abstract": "x" * 100,
        "study_type": "RCT",
    }


def test_make_abstract_record_skips_missing_or_short_abstracts():
    assert fetch_pubmed.make_abstract_record({"AB": ""}) is None
    assert fetch_pubmed.make_abstract_record({"AB": "x" * 99}) is None


def test_deduplicate_abstracts_preserves_first_record_order():
    abstracts = [
        {"pmid": "1", "title": "first"},
        {"pmid": "2", "title": "second"},
        {"pmid": "1", "title": "duplicate"},
    ]

    assert fetch_pubmed.deduplicate_abstracts(abstracts) == abstracts[:2]


class FakeHandle:
    def __init__(self, pmids):
        self.pmids = pmids
        self.closed = False

    def close(self):
        self.closed = True


class FakeEntrez:
    def __init__(self):
        self.handles = []

    def efetch(self, **kwargs):
        handle = FakeHandle(kwargs["id"])
        self.handles.append(handle)
        return handle


class FakeMedline:
    @staticmethod
    def parse(handle):
        return [
            {
                "PMID": pmid,
                "TI": f"Study {pmid}",
                "DP": "2021",
                "AB": "A sufficiently long abstract. " * 4,
                "PT": ["Clinical Trial"],
            }
            for pmid in handle.pmids
        ]


def test_fetch_abstracts_batches_requests_and_closes_handles():
    client = FakeEntrez()
    results = fetch_pubmed.fetch_abstracts(
        ["1", "2", "3"],
        entrez_client=client,
        medline_parser=FakeMedline,
        batch_size=2,
        sleep_fn=lambda _: None,
    )

    assert [handle.pmids for handle in client.handles] == [["1", "2"], ["3"]]
    assert all(handle.closed for handle in client.handles)
    assert [result["pmid"] for result in results] == ["1", "2", "3"]


def test_main_writes_results_without_pubmed_requests(tmp_path, monkeypatch):
    abstracts = [
        {
            "pmid": "1",
            "title": "Study",
            "year": 2022,
            "abstract": "abstract",
            "study_type": "RCT",
        }
    ]
    monkeypatch.setattr(fetch_pubmed, "Entrez", SimpleNamespace(email=None))
    monkeypatch.setattr(fetch_pubmed, "Medline", object())
    monkeypatch.setattr(fetch_pubmed, "search_pubmed", lambda: ["1"])
    monkeypatch.setattr(fetch_pubmed, "fetch_abstracts", lambda pmids: abstracts)
    output_file = tmp_path / "corpus.json"

    assert fetch_pubmed.main(output_file) == 0
    assert json.loads(output_file.read_text(encoding="utf-8")) == abstracts
