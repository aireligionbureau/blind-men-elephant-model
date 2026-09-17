from __future__ import annotations

import html
import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from html.parser import HTMLParser
from typing import Any, Protocol

from .model import DigitalPerson


@dataclass(frozen=True)
class SearchResult:
    id: str
    query: str
    title: str
    url: str
    snippet: str
    source_name: str
    source_type: str
    evidence_type: str
    retrieval_layer: str = "external_search"
    verification_status: str = "external_unverified"
    retrieval_note: str = ""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        ...


@dataclass(frozen=True)
class SearchBatch:
    results: list[SearchResult]
    providers_attempted: list[str]
    provider_errors: list[str]
    providers_succeeded: list[str] | None = None
    providers_skipped: list[str] | None = None
    cache_hit: bool = False
    duration_seconds: float = 0.0


class MockSearchProvider:
    """Deterministic provider used for local development and tests.

    It does not access the network. The goal is to test the filtered retrieval
    and evidence-ledger pipeline before a real search API is selected.
    """

    name = "mock"

    _SOURCE_TEMPLATES = [
        ("Empirical Fixture", "研究材料", "直接测量", "可观察结果、测量边界和方法限制"),
        ("Mechanism Fixture", "机制材料", "因果机制", "中间机制、必要条件和竞争解释"),
        ("Counterexample Fixture", "反证材料", "反例与失败条件", "反例、负结果和可证伪条件"),
        ("Baseline Fixture", "基准材料", "基准率与对照", "基准率、对照组和替代参照"),
        ("Affected-Party Fixture", "一手经验", "受影响者直接经验", "直接经验、摩擦、伤害和意外后果"),
        ("Implementation Fixture", "执行记录", "制度执行证据", "执行过程、责任边界和落地偏差"),
        ("Boundary Fixture", "边界案例", "适用边界", "尺度、时间范围和适用条件"),
        ("Primary-Source Fixture", "一手资料", "原始记录", "原始记录、出处和可复核线索"),
    ]

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        results: list[SearchResult] = []
        for index, template in enumerate(self._SOURCE_TEMPLATES[:limit], start=1):
            source_name, source_type, evidence_type, angle = template
            seed = hashlib.sha256(f"{query}|{source_name}".encode("utf-8")).hexdigest()[:10]
            results.append(
                SearchResult(
                    id=f"mock_{seed}",
                    query=query,
                    title=f"{source_name}: {query}",
                    url=f"mock://{seed}",
                    snippet=f"围绕当前查询“{query}”提供{angle}。这是隔离的 mock 测试材料，不包含任何其他问题的事实。",
                    source_name=source_name,
                    source_type=source_type,
                    evidence_type=evidence_type,
                    retrieval_layer="mock_search",
                    verification_status="synthetic_fixture",
                    retrieval_note="本地模拟搜索结果，只用于开发和测试。",
                )
            )
        return results


def build_model_prior_results(question: str, person: DigitalPerson, *, limit: int = 5) -> list[SearchResult]:
    """Build the digital person's pre-search memory/assumption layer.

    These are not external facts. They make the person's prior commitments
    visible before web or domain-source search begins.
    """

    candidates: list[tuple[str, str, str]] = []
    preferred = person.information_filter.preferred_evidence
    ignored = person.information_filter.ignored_evidence
    top_values = ", ".join(f"{key}:{value}" for key, value in sorted(person.values.items(), key=lambda item: item[1], reverse=True)[:3])

    for evidence_type in preferred[:3]:
        candidates.append(
            (
                evidence_type,
                f"先验：{person.name}会先想到{evidence_type}",
                (
                    f"在看到外部材料之前，{person.name}会从“{person.role_summary}”出发，"
                    f"优先寻找能支持{evidence_type}的线索。价值权重：{top_values}；"
                    f"风险姿态：{person.risk_attitude}。这是一条模型先验，不是外部证据。"
                ),
            )
        )
    for evidence_type in ignored[:2]:
        candidates.append(
            (
                evidence_type,
                f"先验阴影：{person.name}容易低估{evidence_type}",
                (
                    f"这个数字人的信息过滤器写明容易忽略“{evidence_type}”。"
                    f"因此在判断“{question}”时，它可能不会主动寻找这类材料，"
                    "即使这类材料可能改变结论。这是一条模型先验，不是外部证据。"
                ),
            )
        )

    results: list[SearchResult] = []
    for index, (evidence_type, title, snippet) in enumerate(candidates[:limit], start=1):
        seed = hashlib.sha256(f"model-prior|{question}|{person.id}|{index}|{title}".encode("utf-8")).hexdigest()[:12]
        results.append(
            SearchResult(
                id=f"model_prior_{seed}",
                query="model-prior",
                title=title,
                url=f"model-prior://{person.id}/{seed}",
                snippet=snippet,
                source_name="digital-person-prior",
                source_type="模型先验",
                evidence_type=evidence_type,
                retrieval_layer="model_prior",
                verification_status="unverified_model_prior",
                retrieval_note="数字人搜索前会自然带入的记忆、类比、价值排序和注意力偏向。",
            )
        )
    return results


class GoogleCustomSearchProvider:
    """Google Programmable Search / Custom Search JSON API provider."""

    name = "google"

    def __init__(
        self,
        api_key: str | None = None,
        cse_id: str | None = None,
        endpoint: str = "https://www.googleapis.com/customsearch/v1",
        language: str | None = None,
        country: str | None = None,
        safe: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")
        self.endpoint = endpoint
        self.language = language or os.getenv("GOOGLE_SEARCH_LANGUAGE")
        self.country = country or os.getenv("GOOGLE_SEARCH_COUNTRY")
        self.safe = safe or os.getenv("GOOGLE_SEARCH_SAFE")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")
        if not self.cse_id:
            raise RuntimeError("GOOGLE_CSE_ID is not set.")

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": str(min(max(limit, 1), 10)),
        }
        if self.language:
            params["lr"] = self.language
        if self.country:
            params["gl"] = self.country
        if self.safe:
            params["safe"] = self.safe
        payload = _request_json(f"{self.endpoint}?{urllib.parse.urlencode(params)}", timeout=self.timeout_seconds)
        items = payload.get("items") or []
        results: list[SearchResult] = []
        for item in items[:limit]:
            title = str(item.get("title") or "")
            url = str(item.get("link") or "")
            snippet = str(item.get("snippet") or "")
            source_name = _source_name_from_url(url)
            source_type = _infer_source_type(url, title, snippet)
            evidence_type = _infer_evidence_type(url, title, snippet, source_type)
            results.append(
                SearchResult(
                    id=_result_id("google", query, url, title),
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_name=source_name,
                    source_type=source_type,
                    evidence_type=evidence_type,
                )
            )
        return results


class BaiduHtmlSearchProvider:
    """Experimental Baidu result-page provider.

    This is a pragmatic bridge before a stable Baidu JSON/SERP API is selected.
    It may break when Baidu changes its result page. Prefer BaiduJsonSearchProvider
    for production use when a proper endpoint is available.
    """

    name = "baidu-html"

    def __init__(
        self,
        endpoint: str = "https://www.baidu.com/s",
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        params = {"wd": query, "rn": str(min(max(limit, 1), 10))}
        url = f"{self.endpoint}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; BlindMenElephantModel/0.1)",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Baidu HTML search failed: {exc}") from exc
        if any(marker in body for marker in ["百度安全验证", "wappass.baidu.com", "请输入验证码"]):
            raise RuntimeError("Baidu HTML search was blocked by a verification challenge.")
        results = _parse_baidu_html(query, body, limit=limit)
        if not results:
            raise RuntimeError("Baidu HTML search returned no parseable results.")
        return results


class BingHtmlSearchProvider:
    """Keyless Bing result-page provider used as a pragmatic live source."""

    name = "bing-html"

    def __init__(self, endpoint: str = "https://www.bing.com/search", timeout_seconds: int = 30) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        params = {
            "q": query,
            "count": str(min(max(limit, 1), 10)),
            "setlang": "zh-cn" if _contains_cjk(query) else "en-us",
        }
        body = _request_html(f"{self.endpoint}?{urllib.parse.urlencode(params)}", timeout=self.timeout_seconds)
        results = _parse_bing_html(query, body, limit=limit)
        if not results:
            raise RuntimeError("Bing HTML search returned no parseable results.")
        return results


class BingRssSearchProvider:
    """Keyless Bing RSS endpoint with stable structured result fields."""

    name = "bing-rss"

    def __init__(
        self,
        endpoint: str = "https://www.bing.com/search",
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        params = {
            "q": query,
            # Ask for a wider pool so irrelevant top hits can be rejected
            # without immediately exhausting the provider.
            "count": str(min(max(limit * 3, 8), 20)),
            "format": "rss",
            "setlang": "zh-cn" if _contains_cjk(query) else "en-us",
        }
        body = _request_html(
            f"{self.endpoint}?{urllib.parse.urlencode(params)}",
            timeout=self.timeout_seconds,
        )
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise RuntimeError(f"Bing RSS returned invalid XML: {exc}") from exc
        ranked_results: list[tuple[int, SearchResult]] = []
        for item in root.findall(".//item"):
            title = _clean_html(item.findtext("title") or "")
            url = html.unescape((item.findtext("link") or "").strip())
            snippet = _clean_html(item.findtext("description") or "")[:500]
            if not title or not url.startswith(("http://", "https://")):
                continue
            candidate_text = f"{title} {snippet}"
            if not _is_query_result_relevant(query, candidate_text):
                continue
            source_name = _source_name_from_url(url)
            source_type = _infer_source_type(url, title, snippet)
            evidence_type = _infer_evidence_type(
                url, title, snippet, source_type
            )
            ranked_results.append(
                (
                    _query_relevance_score(query, candidate_text),
                    SearchResult(
                    id=_result_id(self.name, query, url, title),
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    source_name=source_name,
                    source_type=source_type,
                    evidence_type=evidence_type,
                    retrieval_note=(
                        "通过 Bing RSS 实时检索；内容仍需来源核验。"
                    ),
                    ),
                )
            )
        ranked_results.sort(key=lambda item: item[0], reverse=True)
        results = [result for _, result in ranked_results[:limit]]
        if not results:
            raise RuntimeError("Bing RSS search returned no relevant results.")
        return results


class GoogleNewsRssSearchProvider:
    """Keyless Google News RSS results with publisher and date provenance."""

    name = "google-news-rss"

    def __init__(
        self,
        endpoint: str = "https://news.google.com/rss/search",
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if _contains_cjk(query):
            locale = {"hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"}
        else:
            locale = {"hl": "en-US", "gl": "US", "ceid": "US:en"}
        body = _request_html(
            f"{self.endpoint}?{urllib.parse.urlencode({'q': query, **locale})}",
            timeout=self.timeout_seconds,
        )
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise RuntimeError(f"Google News RSS returned invalid XML: {exc}") from exc

        ranked_results: list[tuple[int, SearchResult]] = []
        for item in root.findall(".//item"):
            title = _clean_html(item.findtext("title") or "")
            url = html.unescape((item.findtext("link") or "").strip())
            description = _clean_html(item.findtext("description") or "")[:500]
            publisher_node = item.find("source")
            publisher = (
                _clean_html(publisher_node.text or "")
                if publisher_node is not None
                else ""
            )
            publisher_url = (
                html.unescape((publisher_node.attrib.get("url") or "").strip())
                if publisher_node is not None
                else ""
            )
            published_at = (item.findtext("pubDate") or "").strip()
            candidate_text = f"{title} {description} {publisher}"
            if (
                not title
                or not url.startswith(("http://", "https://"))
                or not _is_query_result_relevant(query, candidate_text)
            ):
                continue
            source_name = publisher or _source_name_from_url(publisher_url or url)
            note_parts = ["通过 Google News RSS 实时检索；文章内容仍需来源核验。"]
            if published_at:
                note_parts.append(f"聚合源标注发布时间：{published_at}。")
            if publisher_url:
                note_parts.append(f"原发布方入口：{publisher_url}。")
            ranked_results.append(
                (
                    _query_relevance_score(query, candidate_text),
                    SearchResult(
                        id=_result_id(self.name, query, url, title),
                        query=query,
                        title=title,
                        url=url,
                        snippet=description or title,
                        source_name=source_name,
                        source_type="媒体报道",
                        evidence_type="近期事件",
                        retrieval_layer="external_search",
                        verification_status="external_unverified",
                        retrieval_note=" ".join(note_parts),
                    ),
                )
            )
        ranked_results.sort(key=lambda item: item[0], reverse=True)
        results = [result for _, result in ranked_results[:limit]]
        if not results:
            raise RuntimeError("Google News RSS search returned no relevant results.")
        return results


class DuckDuckGoHtmlSearchProvider:
    """Keyless DuckDuckGo HTML provider for global web retrieval."""

    name = "duckduckgo-html"

    def __init__(self, endpoint: str = "https://html.duckduckgo.com/html/", timeout_seconds: int = 30) -> None:
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        body = _request_html(
            f"{self.endpoint}?{urllib.parse.urlencode({'q': query})}",
            timeout=self.timeout_seconds,
        )
        results = _parse_duckduckgo_html(query, body, limit=limit)
        if not results:
            raise RuntimeError("DuckDuckGo HTML search returned no parseable results.")
        return results


class FederatedHtmlSearchProvider:
    """Concurrent live results from independent keyless web sources.

    Exact-query single-flight caching shares network transport only. Every
    digital person still applies its own source-selection filter afterwards.
    """

    name = "federated-html"

    def __init__(
        self,
        providers: list[SearchProvider] | None = None,
        *,
        circuit_failure_threshold: int = 2,
        cache_size: int = 256,
        single_flight_probe_names: set[str] | None = None,
    ) -> None:
        using_defaults = providers is None
        self.providers = providers or [
            GoogleNewsRssSearchProvider(timeout_seconds=8),
            BingRssSearchProvider(timeout_seconds=8),
            BaiduHtmlSearchProvider(timeout_seconds=6),
            BingHtmlSearchProvider(timeout_seconds=8),
            DuckDuckGoHtmlSearchProvider(timeout_seconds=8),
        ]
        self.circuit_failure_threshold = max(1, circuit_failure_threshold)
        self.cache_size = max(1, cache_size)
        self._lock = threading.Lock()
        self._cache: dict[str, SearchBatch] = {}
        self._inflight: dict[str, Future[SearchBatch]] = {}
        self._provider_failures: dict[str, int] = {}
        self._open_circuits: dict[str, str] = {}
        self._single_flight_probe_names = (
            set(single_flight_probe_names)
            if single_flight_probe_names is not None
            else (
                {"baidu-html", "bing-html", "duckduckgo-html"}
                if using_defaults
                else set()
            )
        )
        self._provider_probe_futures: dict[str, Future[bool]] = {}
        self._provider_proven: set[str] = set()

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        return self.search_with_trace(query, limit=limit).results

    def search_with_trace(self, query: str, *, limit: int = 5) -> SearchBatch:
        cache_key = " ".join(query.casefold().split()) + f"|{max(1, int(limit))}"
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return _cached_search_batch(cached, query)
            pending = self._inflight.get(cache_key)
            leader = pending is None
            if leader:
                pending = Future()
                self._inflight[cache_key] = pending
        assert pending is not None
        if not leader:
            return _cached_search_batch(pending.result(), query)

        try:
            batch = self._search_uncached(query, limit=limit)
        except Exception as exc:
            pending.set_exception(exc)
            raise
        else:
            with self._lock:
                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = batch
            pending.set_result(batch)
            return batch
        finally:
            with self._lock:
                self._inflight.pop(cache_key, None)

    def _search_uncached(self, query: str, *, limit: int) -> SearchBatch:
        started = time.perf_counter()
        with self._lock:
            eligible = [
                provider
                for provider in self.providers
                if provider.name not in self._open_circuits
            ]
            skipped = [
                f"{provider.name}: circuit_open"
                for provider in self.providers
                if provider.name in self._open_circuits
            ]
            if not eligible and self.providers:
                eligible = [
                    min(
                        self.providers,
                        key=lambda item: self._provider_failures.get(
                            item.name, 0
                        ),
                    )
                ]
        per_provider_limit = min(max(limit, 2), 10)
        attempted: list[str] = []
        errors_by_name: dict[str, str] = {}
        buckets_by_name: dict[str, list[SearchResult]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(eligible))) as executor:
            futures = {
                executor.submit(
                    self._search_provider,
                    provider,
                    query,
                    per_provider_limit,
                ): provider
                for provider in eligible
            }
            for future in as_completed(futures):
                provider = futures[future]
                try:
                    results, error, skip_reason, called = future.result()
                except Exception as exc:  # noqa: BLE001 - failures remain evidence-ledger trace.
                    error = str(exc)
                    results = []
                    skip_reason = ""
                    called = True
                    self._record_provider_failure(provider.name, error)
                if called:
                    attempted.append(provider.name)
                if skip_reason:
                    skipped.append(f"{provider.name}: {skip_reason}")
                if error:
                    errors_by_name[provider.name] = error
                    buckets_by_name[provider.name] = []
                    continue
                buckets_by_name[provider.name] = results

        attempted = [
            provider.name
            for provider in eligible
            if provider.name in set(attempted)
        ]

        errors = [
            f"{name}: {errors_by_name[name]}"
            for name in attempted
            if name in errors_by_name
        ]
        succeeded = [name for name in attempted if buckets_by_name.get(name)]

        buckets = [buckets_by_name.get(name, []) for name in attempted]
        merged: list[SearchResult] = []
        seen: set[str] = set()
        cursor = 0
        while len(merged) < limit and any(
            cursor < len(bucket) for bucket in buckets
        ):
            for bucket in buckets:
                if cursor >= len(bucket):
                    continue
                result = bucket[cursor]
                key = _canonical_result_key(result)
                if key not in seen:
                    seen.add(key)
                    merged.append(result)
                    if len(merged) >= limit:
                        break
            cursor += 1
        if not merged:
            detail = "; ".join(errors) or "all providers returned zero results"
            raise RuntimeError(f"Federated HTML search failed: {detail}")
        return SearchBatch(
            results=merged,
            providers_attempted=attempted,
            provider_errors=errors,
            providers_succeeded=succeeded,
            providers_skipped=skipped,
            duration_seconds=round(time.perf_counter() - started, 3),
        )

    def _search_provider(
        self,
        provider: SearchProvider,
        query: str,
        limit: int,
    ) -> tuple[list[SearchResult], str, str, bool]:
        name = provider.name
        if name not in self._single_flight_probe_names:
            return self._search_provider_direct(provider, query, limit)

        with self._lock:
            if name in self._open_circuits:
                return [], "", "circuit_open", False
            if name in self._provider_proven:
                pending = None
                leader = True
            else:
                pending = self._provider_probe_futures.get(name)
                leader = pending is None
                if leader:
                    pending = Future()
                    self._provider_probe_futures[name] = pending
        if name in self._provider_proven:
            return self._search_provider_direct(provider, query, limit)
        assert pending is not None
        if not leader:
            healthy = pending.result()
            if not healthy:
                return [], "", "cold_probe_failed", False
            return self._search_provider_direct(provider, query, limit)

        try:
            result = self._search_provider_direct(provider, query, limit)
            healthy = bool(result[0]) or _is_query_specific_search_miss(
                result[1]
            )
            pending.set_result(healthy)
            if healthy:
                with self._lock:
                    self._provider_proven.add(name)
            return result
        finally:
            with self._lock:
                self._provider_probe_futures.pop(name, None)

    def _search_provider_direct(
        self,
        provider: SearchProvider,
        query: str,
        limit: int,
    ) -> tuple[list[SearchResult], str, str, bool]:
        try:
            results = provider.search(query, limit=limit)
        except Exception as exc:  # noqa: BLE001 - retained in the search trace.
            message = str(exc)
            self._record_provider_failure(provider.name, message)
            return [], message, "", True
        if results:
            self._record_provider_success(provider.name)
            with self._lock:
                self._provider_proven.add(provider.name)
        return results, "", "", True

    def _record_provider_failure(self, name: str, message: str) -> None:
        if _is_query_specific_search_miss(message):
            return
        terminal_markers = (
            "verification challenge",
            "captcha",
            "is not set",
            "http 401",
            "http 403",
        )
        with self._lock:
            failures = self._provider_failures.get(name, 0) + 1
            self._provider_failures[name] = failures
            if failures >= self.circuit_failure_threshold or any(
                marker in message.lower() for marker in terminal_markers
            ):
                self._open_circuits[name] = message[:240]

    def _record_provider_success(self, name: str) -> None:
        with self._lock:
            self._provider_failures[name] = 0
            self._open_circuits.pop(name, None)


def _is_query_specific_search_miss(message: str) -> bool:
    normalized = " ".join(str(message or "").casefold().split())
    return any(
        marker in normalized
        for marker in (
            "no relevant results",
            "no parseable results",
            "returned zero results",
            "all providers returned zero results",
        )
    )


def _cached_search_batch(batch: SearchBatch, query: str) -> SearchBatch:
    return SearchBatch(
        results=[replace(item, query=query) for item in batch.results],
        providers_attempted=list(batch.providers_attempted),
        provider_errors=list(batch.provider_errors),
        providers_succeeded=list(batch.providers_succeeded or []),
        providers_skipped=list(batch.providers_skipped or []),
        cache_hit=True,
        duration_seconds=0.0,
    )


class BaiduJsonSearchProvider:
    """Configurable JSON provider for Baidu-compatible SERP APIs.

    Required env:
    - BAIDU_SEARCH_ENDPOINT

    Optional env:
    - BAIDU_SEARCH_API_KEY
    - BAIDU_SEARCH_QUERY_PARAM, default: q
    - BAIDU_SEARCH_LIMIT_PARAM, default: limit
    - BAIDU_SEARCH_API_KEY_PARAM, send key as query param instead of header
    - BAIDU_SEARCH_AUTH_HEADER, default: Authorization
    - BAIDU_SEARCH_AUTH_PREFIX, default: Bearer
    - BAIDU_SEARCH_RESULTS_PATH, dot path to result list
    - BAIDU_SEARCH_TITLE_FIELD, default: title,name
    - BAIDU_SEARCH_URL_FIELD, default: url,link
    - BAIDU_SEARCH_SNIPPET_FIELD, default: snippet,summary,description
    """

    name = "baidu-json"

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        query_param: str | None = None,
        limit_param: str | None = None,
        api_key_param: str | None = None,
        auth_header: str | None = None,
        auth_prefix: str | None = None,
        results_path: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.endpoint = endpoint or os.getenv("BAIDU_SEARCH_ENDPOINT")
        self.api_key = api_key or os.getenv("BAIDU_SEARCH_API_KEY")
        self.query_param = query_param or os.getenv("BAIDU_SEARCH_QUERY_PARAM", "q")
        self.limit_param = limit_param or os.getenv("BAIDU_SEARCH_LIMIT_PARAM", "limit")
        self.api_key_param = api_key_param or os.getenv("BAIDU_SEARCH_API_KEY_PARAM")
        self.auth_header = auth_header or os.getenv("BAIDU_SEARCH_AUTH_HEADER", "Authorization")
        self.auth_prefix = auth_prefix or os.getenv("BAIDU_SEARCH_AUTH_PREFIX", "Bearer")
        self.results_path = results_path or os.getenv("BAIDU_SEARCH_RESULTS_PATH")
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        if not self.endpoint:
            raise RuntimeError("BAIDU_SEARCH_ENDPOINT is not set.")

        separator = "&" if "?" in self.endpoint else "?"
        query_params = {self.query_param: query, self.limit_param: str(limit)}
        if self.api_key and self.api_key_param:
            query_params[self.api_key_param] = self.api_key
        params = urllib.parse.urlencode(query_params)
        headers = {"Accept": "application/json"}
        if self.api_key and not self.api_key_param:
            prefix = f"{self.auth_prefix} " if self.auth_prefix else ""
            headers[self.auth_header] = f"{prefix}{self.api_key}"
        request = urllib.request.Request(f"{self.endpoint}{separator}{params}", headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Baidu JSON search failed: {exc}") from exc

        return _parse_generic_json_results(
            "baidu-json",
            query,
            payload,
            limit=limit,
            results_path=self.results_path,
        )


def choose_regional_provider(region: str | None = None) -> SearchProvider:
    selected = (region or os.getenv("BME_SEARCH_REGION") or "global").lower()
    allow_mock_fallback = os.getenv("BME_SEARCH_ALLOW_MOCK_FALLBACK", "0").lower() in {"1", "true", "yes"}
    if selected in {"cn", "china", "zh"}:
        if os.getenv("BAIDU_SEARCH_ENDPOINT"):
            return BaiduJsonSearchProvider()
        return FederatedHtmlSearchProvider(
            [
                GoogleNewsRssSearchProvider(timeout_seconds=8),
                BingRssSearchProvider(timeout_seconds=8),
                BaiduHtmlSearchProvider(timeout_seconds=6),
            ]
        )
    if selected in {"global", "overseas", "google", "intl", "international"}:
        if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
            return GoogleCustomSearchProvider()
        if allow_mock_fallback:
            return MockSearchProvider()
        return FederatedHtmlSearchProvider(
            [
                GoogleNewsRssSearchProvider(timeout_seconds=8),
                BingRssSearchProvider(timeout_seconds=8),
                BingHtmlSearchProvider(timeout_seconds=8),
                DuckDuckGoHtmlSearchProvider(timeout_seconds=8),
            ]
        )
    raise ValueError(f"Unknown search region {selected!r}. Use cn or global.")


def build_query_plan(question: str, person: DigitalPerson) -> dict[str, object]:
    plan = person.search_plan(question)
    plan["query_intent"] = _infer_query_intent(person)
    intents = plan["query_intent"]
    plan["search_tool_requests"] = [
        {
            "tool": "web_search",
            "query": query,
            "why_this_person_searches_it": intents[index % len(intents)] if intents else "寻找与自身视角相关的证据",
            "filter_bias": {
                "trusted_sources": person.information_filter.trusted_sources,
                "distrusted_sources": person.information_filter.distrusted_sources,
                "preferred_evidence": person.information_filter.preferred_evidence,
                "ignored_evidence": person.information_filter.ignored_evidence,
            },
        }
        for index, query in enumerate(plan["preferred_queries"])
    ]
    plan["prior_layer_note"] = "搜索前先显影该数字人的模型先验；先验不是外部事实，但会影响它搜什么、信什么、忽略什么。"
    return plan


def collect_candidates(
    question: str,
    person: DigitalPerson,
    provider: SearchProvider,
    *,
    results_per_query: int = 5,
    attempt_trace: list[dict[str, object]] | None = None,
) -> list[SearchResult]:
    plan = build_query_plan(question, person)
    candidates: list[SearchResult] = []
    seen: set[str] = set()
    errors: list[str] = []
    preferred_queries = [str(query) for query in plan["preferred_queries"]]

    def search_one(query: str, *, planning_source: str) -> None:
        try:
            results = provider.search(query, limit=results_per_query)
        except Exception as exc:  # noqa: BLE001 - another biased query may still work.
            errors.append(f"{query}: {exc}")
            if attempt_trace is not None:
                attempt_trace.append(
                    {
                        "query": query,
                        "planning_source": planning_source,
                        "status": "failed",
                        "result_count": 0,
                        "error": str(exc),
                    }
                )
            return
        if attempt_trace is not None:
            attempt_trace.append(
                {
                    "query": query,
                    "planning_source": planning_source,
                    "status": "executed",
                    "result_count": len(results),
                }
            )
        for result in results:
            if result.id in seen:
                continue
            seen.add(result.id)
            candidates.append(result)

    for query in preferred_queries:
        search_one(query, planning_source="persona_filtered_fallback")

    normalized_preferred = {" ".join(query.casefold().split()) for query in preferred_queries}
    normalized_question = " ".join(question.casefold().split())
    if not candidates and normalized_question not in normalized_preferred:
        search_one(question, planning_source="question_centered_recovery")

    if not candidates and errors:
        raise RuntimeError("All persona-filtered search queries failed: " + "; ".join(errors))
    return candidates


def serialize_results(results: list[SearchResult]) -> list[dict[str, object]]:
    return [asdict(result) for result in results]


def _request_json(url: str, *, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Search HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Search request failed: {exc}") from exc


def _request_html(url: str, *, timeout: int) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Search HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Search request failed: {exc}") from exc


def _parse_bing_html(query: str, body: str, *, limit: int) -> list[SearchResult]:
    parser = _BingResultParser()
    parser.feed(body)
    return _html_entries_to_results("bing-html", query, parser.entries, limit=limit)


def _parse_duckduckgo_html(query: str, body: str, *, limit: int) -> list[SearchResult]:
    parser = _DuckDuckGoResultParser()
    parser.feed(body)
    entries = [
        {**entry, "url": _normalize_duckduckgo_url(str(entry.get("url", "")))}
        for entry in parser.entries
    ]
    return _html_entries_to_results("duckduckgo-html", query, entries, limit=limit)


class _BingResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[dict[str, str]] = []
        self.current: dict[str, Any] | None = None
        self.in_h2 = False
        self.capture_title = False
        self.capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set(str(attributes.get("class") or "").split())
        if tag == "li" and "b_algo" in classes:
            self.current = {"url": "", "title_parts": [], "snippet_parts": []}
        elif self.current is not None and tag == "h2":
            self.in_h2 = True
        elif self.current is not None and self.in_h2 and tag == "a" and attributes.get("href"):
            self.current["url"] = str(attributes["href"])
            self.capture_title = True
        elif self.current is not None and tag == "p":
            self.capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.capture_title = False
        elif tag == "h2":
            self.in_h2 = False
        elif tag == "p":
            self.capture_snippet = False
        elif tag == "li" and self.current is not None:
            self.entries.append(
                {
                    "url": str(self.current.get("url", "")),
                    "title": " ".join(self.current.get("title_parts", [])).strip(),
                    "snippet": " ".join(self.current.get("snippet_parts", [])).strip(),
                }
            )
            self.current = None
            self.in_h2 = False
            self.capture_title = False
            self.capture_snippet = False

    def handle_data(self, data: str) -> None:
        if self.current is None or not data.strip():
            return
        if self.capture_title:
            self.current["title_parts"].append(data.strip())
        elif self.capture_snippet:
            self.current["snippet_parts"].append(data.strip())


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.entries: list[dict[str, str]] = []
        self.current_title: dict[str, Any] | None = None
        self.capture_title = False
        self.capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set(str(attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes and attributes.get("href"):
            self.current_title = {
                "url": str(attributes["href"]),
                "title_parts": [],
                "snippet": "",
            }
            self.capture_title = True
        elif "result__snippet" in classes and self.entries:
            self.capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.capture_title and self.current_title is not None:
            self.entries.append(
                {
                    "url": str(self.current_title.get("url", "")),
                    "title": " ".join(self.current_title.get("title_parts", [])).strip(),
                    "snippet": "",
                }
            )
            self.current_title = None
            self.capture_title = False
        elif self.capture_snippet and tag in {"a", "div", "span"}:
            self.capture_snippet = False

    def handle_data(self, data: str) -> None:
        if not data.strip():
            return
        if self.capture_title and self.current_title is not None:
            self.current_title["title_parts"].append(data.strip())
        elif self.capture_snippet and self.entries:
            existing = self.entries[-1].get("snippet", "")
            self.entries[-1]["snippet"] = f"{existing} {data.strip()}".strip()


def _html_entries_to_results(
    provider_name: str,
    query: str,
    entries: list[dict[str, str]],
    *,
    limit: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for entry in entries:
        url = html.unescape(str(entry.get("url", "")).strip())
        title = re.sub(r"\s+", " ", str(entry.get("title", ""))).strip()
        snippet = re.sub(r"\s+", " ", str(entry.get("snippet", ""))).strip()[:500]
        if not url or not title or not url.startswith(("http://", "https://")):
            continue
        if not _is_query_result_relevant(query, f"{title} {snippet}"):
            continue
        source_name = _source_name_from_url(url)
        source_type = _infer_source_type(url, title, snippet)
        evidence_type = _infer_evidence_type(url, title, snippet, source_type)
        results.append(
            SearchResult(
                id=_result_id(provider_name, query, url, title),
                query=query,
                title=title,
                url=url,
                snippet=snippet,
                source_name=source_name,
                source_type=source_type,
                evidence_type=evidence_type,
                retrieval_layer="external_search",
                verification_status="external_unverified",
                retrieval_note=f"通过 {provider_name} 实时检索；内容仍需来源核验。",
            )
        )
        if len(results) >= limit:
            break
    return results


def _normalize_duckduckgo_url(url: str) -> str:
    absolute = urllib.parse.urljoin("https://duckduckgo.com", html.unescape(url))
    parsed = urllib.parse.urlparse(absolute)
    if parsed.netloc.endswith("duckduckgo.com"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [])
        if target:
            return urllib.parse.unquote(target[0])
    return absolute


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= character <= "\u9fff" for character in text)


def _canonical_result_key(result: SearchResult) -> str:
    try:
        parsed = urllib.parse.urlparse(result.url)
        return f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}"
    except ValueError:
        return result.url.lower()


def _query_relevance_score(query: str, candidate: str) -> int:
    ascii_terms, cjk_terms = _query_relevance_terms(query)
    lowered_candidate = candidate.lower()
    return sum(1 for term in ascii_terms if term in lowered_candidate) + sum(
        2 for term in cjk_terms if term in candidate
    )


def _query_relevance_terms(query: str) -> tuple[set[str], set[str]]:
    lowered_query = query.lower()
    ascii_stop = {
        "about",
        "could",
        "evidence",
        "from",
        "have",
        "research",
        "should",
        "study",
        "that",
        "theories",
        "what",
        "whether",
        "will",
        "with",
        "would",
    }
    ascii_terms = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", lowered_query)
        if token not in ascii_stop
    }
    cjk_stop = {"是否", "应该", "如何", "什么", "问题", "研究", "证据", "可能", "会不", "不会"}
    cjk_terms: set[str] = set()
    for chunk in re.findall(r"[\u4e00-\u9fff]+", query):
        if len(chunk) == 1:
            cjk_terms.add(chunk)
        else:
            cjk_terms.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    cjk_terms -= cjk_stop
    return ascii_terms, cjk_terms


def _is_query_result_relevant(query: str, candidate: str) -> bool:
    """Reject broad location/topic pages that only echo one weak query term."""

    ascii_terms, cjk_terms = _query_relevance_terms(query)
    lowered_candidate = candidate.lower()
    ascii_matches = {term for term in ascii_terms if term in lowered_candidate}
    cjk_matches = {term for term in cjk_terms if term in candidate}
    total_terms = len(ascii_terms) + len(cjk_terms)
    total_matches = len(ascii_matches) + len(cjk_matches)
    if total_matches == 0:
        return False
    if total_terms <= 1 or total_matches >= 2:
        return True

    generic_ascii = {
        "analysis",
        "crash",
        "evidence",
        "market",
        "price",
        "risk",
        "share",
        "stock",
        "valuation",
    }
    if any(len(term) >= 5 and term not in generic_ascii for term in ascii_matches):
        return True

    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{4,}", query)
    return any(chunk in candidate for chunk in cjk_chunks)


def _parse_baidu_html(query: str, body: str, *, limit: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    # Baidu result markup changes often. This catches common h3 result anchors
    # and keeps snippets conservative.
    pattern = re.compile(
        r"<h3[^>]*>.*?<a[^>]+href=[\"'](?P<url>[^\"']+)[\"'][^>]*>(?P<title>.*?)</a>.*?</h3>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(body):
        raw_url = html.unescape(match.group("url"))
        title = _clean_html(match.group("title"))
        if not title or not raw_url:
            continue
        snippet = _nearby_text(body, match.end(), max_chars=220)
        if not _is_query_result_relevant(query, f"{title} {snippet}"):
            continue
        source_name = _source_name_from_url(raw_url)
        source_type = _infer_source_type(raw_url, title, snippet)
        evidence_type = _infer_evidence_type(raw_url, title, snippet, source_type)
        results.append(
            SearchResult(
                id=_result_id("baidu-html", query, raw_url, title),
                query=query,
                title=title,
                url=raw_url,
                snippet=snippet,
                source_name=source_name,
                source_type=source_type,
                evidence_type=evidence_type,
            )
        )
        if len(results) >= limit:
            break
    return results


def _parse_generic_json_results(
    provider_name: str,
    query: str,
    payload: dict[str, Any],
    *,
    limit: int,
    results_path: str | None = None,
) -> list[SearchResult]:
    items = _items_from_payload(payload, results_path)
    if isinstance(items, dict):
        items = items.get("items") or items.get("results") or []

    results: list[SearchResult] = []
    for item in list(items)[:limit]:
        if not isinstance(item, dict):
            continue
        title = str(_first_field(item, os.getenv("BAIDU_SEARCH_TITLE_FIELD"), ["title", "name", "headline"]) or "")
        url = str(_first_field(item, os.getenv("BAIDU_SEARCH_URL_FIELD"), ["url", "link", "target"]) or "")
        snippet = str(
            _first_field(
                item,
                os.getenv("BAIDU_SEARCH_SNIPPET_FIELD"),
                ["snippet", "summary", "description", "abstract"],
            )
            or ""
        )
        if not title and not url:
            continue
        source_name = str(item.get("source") or _source_name_from_url(url))
        source_type = _infer_source_type(url, title, snippet)
        evidence_type = _infer_evidence_type(url, title, snippet, source_type)
        results.append(
            SearchResult(
                id=_result_id(provider_name, query, url, title),
                query=query,
                title=title,
                url=url,
                snippet=snippet,
                source_name=source_name,
                source_type=source_type,
                evidence_type=evidence_type,
            )
        )
    return results


def _items_from_payload(payload: dict[str, Any], results_path: str | None) -> Any:
    if results_path:
        selected = _nested_get(payload, results_path)
        if selected is not None:
            return selected
    return payload.get("items") or payload.get("results") or payload.get("data") or []


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
        if current is None:
            return None
    return current


def _first_field(item: dict[str, Any], configured: str | None, defaults: list[str]) -> Any:
    fields = []
    if configured:
        fields.extend(part.strip() for part in configured.split(",") if part.strip())
    fields.extend(defaults)
    for field in fields:
        value = _nested_get(item, field) if "." in field else item.get(field)
        if value:
            return value
    return None


def _result_id(provider_name: str, query: str, url: str, title: str) -> str:
    seed = hashlib.sha256(f"{provider_name}|{query}|{url}|{title}".encode("utf-8")).hexdigest()[:12]
    return f"{provider_name}_{seed}"


def _source_name_from_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError:
        return "unknown"
    return parsed.netloc or parsed.path.split("/")[0] or "unknown"


def _infer_source_type(url: str, title: str, snippet: str) -> str:
    text = f"{url} {title} {snippet}".lower()
    if any(token in text for token in ["gov", "政府", "监管", "policy", "regulation"]):
        return "监管文件"
    if any(token in text for token in ["edu", "ac.", "journal", "论文", "学术", "research", "study"]):
        return "学术论文"
    if any(token in text for token in ["事故", "incident", "failure", "risk", "安全公告"]):
        return "事故调查报告"
    if any(token in text for token in ["whitepaper", "白皮书", "industry", "company", "企业"]):
        return "企业白皮书"
    if any(token in text for token in ["worker", "工人", "访谈", "forum", "社区"]):
        return "工人访谈"
    if any(token in text for token in ["market", "finance", "成本", "收益", "价格"]):
        return "市场数据"
    if any(token in text for token in ["ethics", "伦理", "rights", "权利"]):
        return "伦理论文"
    if any(token in text for token in ["local", "地方", "news", "媒体"]):
        return "地方媒体"
    return "公开网页"


def _infer_evidence_type(url: str, title: str, snippet: str, source_type: str) -> str:
    text = f"{url} {title} {snippet}".lower()
    if source_type == "事故调查报告" or any(token in text for token in ["事故", "failure", "incident"]):
        return "事故案例"
    if source_type == "学术论文" or any(token in text for token in ["study", "research", "论文"]):
        return "同行评审论文"
    if source_type == "监管文件":
        return "监管文件"
    if source_type == "企业白皮书":
        return "企业白皮书"
    if source_type == "工人访谈" or any(token in text for token in ["访谈", "interview", "story"]):
        return "质性叙事"
    if source_type == "市场数据":
        return "市场数据"
    if source_type == "伦理论文":
        return "原则冲突"
    if source_type == "地方媒体":
        return "地方案例"
    return "公开信息"


def _clean_html(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return html.unescape(re.sub(r"\s+", " ", without_tags)).strip()


def _nearby_text(body: str, start: int, *, max_chars: int) -> str:
    window = body[start : start + 1200]
    cleaned = _clean_html(window)
    cut_points = [
        cleaned.find(marker)
        for marker in ('"}],', '"isSingleLine"', '"rightGrid"', 'cos-imag')
        if marker in cleaned
    ]
    if cut_points:
        cleaned = cleaned[: min(cut_points)].strip(' ,;:{[\\"')
    if cleaned.startswith("<"):
        return ""
    return cleaned[:max_chars]


def _infer_query_intent(person: DigitalPerson) -> list[str]:
    intents: list[str] = []
    if any(value in person.values for value in ["safety", "stability"]):
        intents.append("寻找风险、失败模式和稳定性证据")
    if any(value in person.values for value in ["efficiency", "innovation"]):
        intents.append("寻找效率、成本、采用率和创新扩散证据")
    if any(value in person.values for value in ["fairness", "dignity"]):
        intents.append("寻找分配影响、受影响者叙事和权利证据")
    if "系统思维" in person.cognitive_frames:
        intents.append("寻找反馈回路、二阶后果和系统外部性")
    if not intents:
        intents.append("寻找与自身专业域最相关的证据")
    return intents
