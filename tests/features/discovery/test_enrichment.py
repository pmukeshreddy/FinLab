from types import SimpleNamespace
from unittest.mock import MagicMock

from finpipe.discovery.models import PaperRecord
from finpipe.discovery.sources import crossref, semantic_scholar
from finpipe.discovery.sources.collector import PaperCollector


def test_collector_enriches_doi_before_semantic_metadata():
    collector = PaperCollector.__new__(PaperCollector)
    raw = {"p": PaperRecord(paper_id="p", scope_id="s", title="Title", abstract="a")}
    with_doi = dict(raw)
    enriched = dict(raw)
    collector.arxiv = SimpleNamespace(
        get_paper=lambda *_: raw["p"], scrape_scope=lambda *_, **__: raw
    )
    collector.crossref = SimpleNamespace(
        enrich_doi=lambda records: (
            with_doi if records is raw or records == raw else None
        )
    )
    collector.s2 = SimpleNamespace(
        enrich=lambda records: enriched if records is with_doi else None
    )
    assert collector.retrieve_paper("p", "s") is enriched["p"]
    assert collector.collect_scope(SimpleNamespace(), None, None, 5) is enriched


def test_crossref_backfills_without_overwriting_known_metadata_and_delays(monkeypatch):
    client = crossref.CrossrefClient.__new__(crossref.CrossrefClient)
    client.enabled, client.request_delay_secs = True, 0.25
    client._lookup_metadata = lambda *_: {
        "doi": "new",
        "journal_name": "new journal",
        "publication_type": "journal-article",
        "citation_count": 10,
    }
    record = PaperRecord(
        paper_id="p",
        scope_id="s",
        title="Title",
        abstract="a",
        authors=["Author"],
        doi="existing",
        citation_count=4,
    )
    pauses = []
    monkeypatch.setattr(crossref.time, "sleep", pauses.append)
    assert client.enrich_doi({"p": record})["p"] is record
    assert record.doi == "existing"
    assert record.citation_count == 4
    assert record.journal_name == "new journal"
    assert record.publication_types == ["journal-article"]
    assert record.peer_reviewed
    assert pauses == [0.25]


def test_crossref_selects_best_title_and_applies_similarity_cutoff():
    client = crossref.CrossrefClient.__new__(crossref.CrossrefClient)
    client.max_results, client.min_title_similarity = 5, 1.0
    client.client = SimpleNamespace(
        works=lambda **_: {
            "message": {
                "items": [
                    {"DOI": "irrelevant", "title": ["Other title"]},
                    {
                        "DOI": " ABC ",
                        "title": ["  MATCHING   TITLE "],
                        "container-title": [" Journal "],
                        "type": "journal-article",
                        "is-referenced-by-count": 8,
                    },
                ]
            }
        }
    )
    assert client._lookup_metadata("Matching Title", "Author") == {
        "doi": "abc",
        "journal_name": "Journal",
        "publication_type": "journal-article",
        "citation_count": 8,
    }
    assert client._lookup_metadata("Unmatched Title", "Author") is None


def test_semantic_scholar_batches_and_overwrites_fields_but_retains_peer_review(
    monkeypatch,
):
    client = semantic_scholar.SemanticScholarClient.__new__(
        semantic_scholar.SemanticScholarClient
    )
    client.enabled, client.batch_size, client.request_delay_secs = True, 2, 0.1
    client.client = MagicMock()

    def get_papers(ids, **_):
        return [
            SimpleNamespace(
                externalIds={"ArXiv": i.removeprefix("ArXiv:")},
                citationCount=7,
                influentialCitationCount=2,
                paperId="semantic",
                venue="venue",
                journal=None,
                publicationTypes=None,
            )
            for i in ids
        ]

    client.client.get_papers.side_effect = get_papers
    pauses = []
    monkeypatch.setattr(semantic_scholar.time, "sleep", pauses.append)
    records = {
        f"ArXiv:{i}": PaperRecord(
            paper_id=f"ArXiv:{i}",
            scope_id="s",
            title="Title",
            abstract="a",
            journal_name="old",
            peer_reviewed=True,
        )
        for i in range(3)
    }
    assert client.enrich(records) is records
    assert [call.args[0] for call in client.client.get_papers.call_args_list] == [
        ["ArXiv:0", "ArXiv:1"],
        ["ArXiv:2"],
    ]
    assert pauses == [0.1, 0.1]
    for record in records.values():
        assert record.citation_count == 7
        assert record.journal_name is None
        assert record.publication_types == []
        assert record.peer_reviewed is True
