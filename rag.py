"""
Lightweight live medical-knowledge retrieval, standing in for the paper's
MedRAG component. The real MedRAG toolkit (Xiong et al.) builds a vector
index over curated corpora (PubMed abstracts, StatPearls, textbooks) ahead
of time. That's explicitly out of scope here for cost/complexity reasons -
this instead hits two free, keyless public APIs at query time:
  - NCBI E-utilities, to search PubMed and pull back real abstracts
  - Wikipedia's public API, for general medical/anatomical background

No local corpus, no vector DB, no API key, no cost.
"""
import logging
import re
import time
import xml.etree.ElementTree as ET

import requests

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# NCBI's usage policy asks unauthenticated callers to self-identify via
# these params. A real deployment should swap in a real, monitored email.
NCBI_TOOL = "mdagents-demo"
NCBI_EMAIL = "mdagents@example.com"

# NCBI allows ~3 req/s without an API key. retrieve() makes at most two
# sequential E-utilities calls (esearch then efetch); this sleep just keeps
# back-to-back retrieve() calls (e.g. looping over several sub-questions)
# from bursting past that.
NCBI_THROTTLE_SECONDS = 0.34

SNIPPET_MAX_CHARS = 450

# Plain grammatical stopwords plus generic "question shape" words
# (mechanism/action/diagnostic/criteria/cause/common/...) that show up in
# almost every medical topic's text. Passing a raw natural-language question
# straight through to PubMed/Wikipedia search causes two concrete problems
# this stripping fixes: (1) PubMed's automatic term mapping occasionally
# misparses a stopword pair like "is the" as an author-name filter (e.g.
# "The, Is[Full Author Name]"), silently zeroing out results; (2) both
# PubMed's title-AND matching and Wikipedia's relevance ranking get diluted
# by generic words that are common across unrelated articles, burying the
# actually-relevant one (e.g. "diagnostic criteria" pulled up Lupus/ALS
# ahead of the Myasthenia gravis article itself during testing).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "whom", "how", "why", "when", "where", "does", "do", "did", "of", "for",
    "in", "on", "to", "and", "or", "with", "that", "this", "these", "those",
    "be", "been", "being", "it", "its", "as", "by", "at",
    "most", "common", "cause", "causes", "mechanism", "action", "diagnostic",
    "criteria", "sign", "signs", "symptom", "symptoms",
}

# (query, k) -> list[str]. Only successful (non-empty) results are cached,
# so a transient outage doesn't get "stuck" for the rest of the process.
_cache = {}


def _keywords(query):
    words = re.findall(r"[A-Za-z0-9\-]+", query)
    kept = [w for w in words if w.lower() not in _STOPWORDS]
    return " ".join(kept) if kept else query


def _truncate(text, max_chars=SNIPPET_MAX_CHARS):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def _pubmed_esearch(query, k, timeout, restrict_title):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": k,
        "retmode": "json",
        "sort": "relevance",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
    }
    if restrict_title:
        params["field"] = "title"
    resp = requests.get(EUTILS_BASE + "esearch.fcgi", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["esearchresult"].get("idlist", [])


def _pubmed_efetch(pmids, timeout):
    resp = requests.get(
        EUTILS_BASE + "efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": NCBI_TOOL,
            "email": NCBI_EMAIL,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    results = []
    for article in root.findall(".//PubmedArticle"):
        # Read the PMID back out of each <PubmedArticle> itself rather than
        # zipping against the input pmids list - articles can fail to parse
        # or be skipped, which would silently misalign a zip.
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        abstract_parts = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            text = "".join(ab.itertext())
            abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)
        display = f"{title} {abstract}".strip() if abstract else title
        if display:
            results.append({
                "source": "pubmed",
                "id": pmid,
                "title": title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "snippet": _truncate(abstract) if abstract else "",
                "display": _truncate(display),
            })
    return results


def _pubmed_snippets(query, k, timeout):
    """PubMed's automatic term-mapping expands plain keywords into a broad
    OR-of-synonyms MeSH query, which for short factual questions surfaces
    tangentially related papers rather than on-topic ones. Restricting the
    search to the title field is far more precise (all query words must
    appear in the title) and is tried first; if it doesn't fill k results
    (common - not every topic has k+ papers with all those words in the
    title), the remainder is topped up from an unrestricted search."""
    term = _keywords(query)
    try:
        pmids = _pubmed_esearch(term, k, timeout, restrict_title=True)
        if len(pmids) < k:
            more = _pubmed_esearch(term, k, timeout, restrict_title=False)
            pmids = pmids + [p for p in more if p not in pmids]
        pmids = pmids[:k]
        if not pmids:
            return []

        time.sleep(NCBI_THROTTLE_SECONDS)
        return _pubmed_efetch(pmids, timeout)
    except (requests.RequestException, ValueError, KeyError, ET.ParseError) as e:
        logger.warning("PubMed retrieval failed for query %r: %s", query, e)
        return []


def _wikipedia_snippets(query, k, timeout):
    """Wikipedia's default relevance ranking is plain text-match density,
    which for multi-word medical questions often ranks big, generic-word-
    heavy articles (e.g. "Lupus", "ALS", both of which discuss "diagnostic
    criteria" at length) above the one article that's actually about the
    query's subject. Two mitigations: search on the same stopword-stripped
    keywords used for PubMed (fewer generic words to over-match on), and
    then keep only hits whose title shares one of those keywords - dropping
    the rest rather than padding the result with off-topic filler."""
    term = _keywords(query)
    # Wikipedia's CirrusSearch hard-rejects queries over 300 chars
    # ("cirrussearch-query-too-long") - a long clinical vignette easily
    # produces a 400+ char keyword string even after stopword-stripping,
    # which silently zeroed out every Wikipedia result for such questions
    # (the error response has no "query" key, so it looked like "no
    # results" rather than a rejected request). Capping to the first 15
    # keywords fixes the crash and, per the same over-matching concern
    # above, also improves relevance - a shorter query has less chance to
    # get diluted by tangential terms.
    words = term.split()[:15]
    term = " ".join(words)
    if len(term) > 300:
        term = term[:300].rsplit(" ", 1)[0]
    keywords = {w.lower() for w in term.split()}
    try:
        resp = requests.get(
            WIKI_SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srlimit": max(k * 3, 5),
                "format": "json",
            },
            headers={"User-Agent": f"{NCBI_TOOL}/1.0 ({NCBI_EMAIL})"},
            timeout=timeout,
        )
        resp.raise_for_status()
        hits = resp.json()["query"]["search"]
        relevant = [
            h["title"] for h in hits
            if keywords & {w.lower() for w in re.findall(r"[A-Za-z0-9\-]+", h["title"])}
        ]
        titles = relevant[:k]
        if not titles:
            return []

        results = []
        for title in titles:
            try:
                summary_resp = requests.get(
                    WIKI_SUMMARY_URL + requests.utils.quote(title),
                    headers={"User-Agent": f"{NCBI_TOOL}/1.0 ({NCBI_EMAIL})"},
                    timeout=timeout,
                )
                summary_resp.raise_for_status()
                extract = summary_resp.json().get("extract", "")
                if extract:
                    results.append({
                        "source": "wikipedia",
                        "id": title,
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{requests.utils.quote(title.replace(' ', '_'))}",
                        "snippet": _truncate(extract),
                        "display": _truncate(f"{title}: {extract}"),
                    })
            except (requests.RequestException, ValueError) as e:
                logger.warning("Wikipedia summary fetch failed for %r: %s", title, e)
        return results
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.warning("Wikipedia retrieval failed for query %r: %s", query, e)
        return []


def retrieve_structured(query: str, k: int = 3, timeout: int = 10) -> list[dict]:
    """Return up to k structured evidence items relevant to `query`, each
    {source, id, title, url, snippet, display}. This is the single source of
    truth - retrieve() below just extracts `display` from each item, so
    citation metadata (for e.g. "according to PubMed, PMID ...") is
    available without a second round-trip to these APIs.

    Tries PubMed first (real clinical literature, higher quality when it
    hits) and tops up with Wikipedia (broader coverage, good for basic
    anatomy/physiology/definitions PubMed abstracts often assume as known)
    if PubMed didn't return enough. Never raises: any failure is logged as
    a warning and treated as zero results from that source, so a down or
    slow API degrades the result list instead of crashing the caller.
    """
    cache_key = (query, k)
    if cache_key in _cache:
        return _cache[cache_key]

    if not query or not query.strip():
        return []

    items = []
    try:
        items.extend(_pubmed_snippets(query, k, timeout))
    except Exception as e:  # belt-and-suspenders: this function must never raise
        logger.warning("Unexpected error during PubMed retrieval for %r: %s", query, e)

    if len(items) < k:
        try:
            remaining = k - len(items)
            items.extend(_wikipedia_snippets(query, remaining, timeout))
        except Exception as e:
            logger.warning("Unexpected error during Wikipedia retrieval for %r: %s", query, e)

    items = items[:k]
    if items:
        _cache[cache_key] = items
    return items


def retrieve(query: str, k: int = 3, timeout: int = 10) -> list[str]:
    """Backwards-compatible plain-text interface: same retrieval as
    retrieve_structured(), just the display strings. Existing callers that
    only need text for prompt injection (not citation metadata) can keep
    using this unchanged."""
    return [item["display"] for item in retrieve_structured(query, k, timeout)]


def format_evidence(snippets: list[str]) -> str:
    """Format snippets for prompt injection. Returns "" for an empty list
    so callers can always do `prompt += format_evidence(snippets)` without
    an extra conditional."""
    if not snippets:
        return ""
    lines = [f"{i}. {s}" for i, s in enumerate(snippets, 1)]
    return "Relevant medical literature:\n" + "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    questions = [
        "What is the most common cause of hemorrhagic brain metastasis?",
        "What is the mechanism of action of proton pump inhibitors?",
        "What are the diagnostic criteria for myasthenia gravis?",
    ]

    for q in questions:
        print("=" * 80)
        print("QUERY:", q)
        results = retrieve(q, k=3)
        print(format_evidence(results) or "(no results)")
        print()
