from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .agentic_search import run_agentic_search
from .cohort import CohortBundle, build_fixture_cohort
from .config import DEFAULT_MODEL, RUN_PROFILES
from .context import validate_run_isolation
from .evidence import build_evidence_ledger
from .json_utils import parse_json_content
from .meta_model import diagnose_cognitive_shadows
from .model import build_run_plan, select_personas
from .prompts import render_digital_person_messages, render_meta_model_messages
from .providers import DeepSeekAPIError, DeepSeekClient, retry_delay_seconds
from .puzzle import build_shadow_puzzle
from .runtime import atomic_write_json, read_json
from .search import (
    BaiduHtmlSearchProvider,
    BaiduJsonSearchProvider,
    BingHtmlSearchProvider,
    BingRssSearchProvider,
    DuckDuckGoHtmlSearchProvider,
    FederatedHtmlSearchProvider,
    GoogleCustomSearchProvider,
    GoogleNewsRssSearchProvider,
    MockSearchProvider,
    SearchResult,
    build_query_plan,
    build_model_prior_results,
    choose_regional_provider,
    collect_candidates,
    serialize_results,
)


RETRIEVAL_IMPLEMENTATION_REVISION = "2026-07-28-query-miss-circuit-v6"


@dataclass(frozen=True)
class RetrievalStack:
    name: str
    external_provider: Any | None = None
    include_model_prior: bool = False
    tolerate_external_errors: bool = False
    evidence_mode: str = "live_external"
    agentic_tool_calls: bool = False


class RetrievalBatchError(RuntimeError):
    """Raised after all independent retrieval jobs have had a chance to finish."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = failures
        self.retryable = all(item.get("retryable", True) for item in failures)
        person_ids = ", ".join(str(item.get("person_id")) for item in failures)
        super().__init__(f"Retrieval failed for {person_ids}")


def build_profile_plan(question: str, profile_name: str) -> dict[str, Any]:
    profile = RUN_PROFILES[profile_name]
    plan = build_run_plan(question, person_count=profile.person_count)
    plan["profile"] = asdict(profile)
    return plan


def run_filtered_retrieval(
    question: str,
    *,
    profile_name: str | None = None,
    person_count: int | None = None,
    provider_name: str = "mock",
    results_per_query: int = 5,
    cohort: CohortBundle | None = None,
    model: str | None = DEFAULT_MODEL,
    tool_retries: int = 2,
    concurrency: int = 1,
    checkpoint_dir: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if profile_name is not None:
        profile = RUN_PROFILES[profile_name]
        count = profile.person_count
        profile_payload: dict[str, Any] | None = asdict(profile)
    else:
        count = person_count or 4
        profile_payload = None

    stack = _build_retrieval_stack(provider_name)
    active_cohort = cohort or build_fixture_cohort(question, count)
    if active_cohort.question != question:
        raise ValueError("cohort question does not match the current question")
    if len(active_cohort.personas) != count:
        raise ValueError(f"cohort has {len(active_cohort.personas)} people; expected {count}")
    personas = active_cohort.personas

    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    reused_count = 0
    pending = []
    for person in personas:
        cached = _load_retrieval_checkpoint(
            checkpoint_path / f"{person.id}.json" if checkpoint_path else None,
            question=question,
            cohort_id=active_cohort.cohort_id,
            person_id=person.id,
            provider_name=provider_name,
            results_per_query=results_per_query,
            model=model,
        )
        if cached:
            completed[person.id] = cached
            reused_count += 1
        else:
            pending.append(person)

    failures: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback(
            {
                "completed": len(completed),
                "failed": 0,
                "reused": reused_count,
                "total": len(personas),
            }
        )
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_one_retrieval_item,
                question,
                person,
                active_cohort,
                provider_name,
                results_per_query,
                model,
                tool_retries,
                stack=stack,
            ): person
            for person in pending
        }
        for future in as_completed(futures):
            person = futures[future]
            try:
                item = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve completed peers and the failure record.
                item = {
                    "schema": "bme.retrieval-item.v2",
                    "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
                    "status": "failed",
                    "question": question,
                    "cohort_id": active_cohort.cohort_id,
                    "person_id": person.id,
                    "provider_name": provider_name,
                    "results_per_query": results_per_query,
                    "model": model,
                    "error": str(exc),
                    "retryable": bool(getattr(exc, "retryable", True)),
                }
                failures.append(item)
            if checkpoint_path is not None:
                atomic_write_json(checkpoint_path / f"{person.id}.json", item)
            if item.get("status") == "succeeded":
                completed[person.id] = item
            elif item not in failures:
                failures.append(item)
            if progress_callback:
                progress_callback(
                    {
                        "completed": len(completed),
                        "failed": len(failures),
                        "reused": reused_count,
                        "total": len(personas),
                    }
                )

    if failures:
        raise RetrievalBatchError(failures)

    ordered_items = [completed[person.id] for person in personas]
    ledgers = [item["evidence_ledger"] for item in ordered_items]
    retrieval_records = [item["retrieval_record"] for item in ordered_items]
    tool_traces = [
        item["tool_trace"]
        for item in ordered_items
        if isinstance(item.get("tool_trace"), dict) and item["tool_trace"]
    ]

    result = {
        "question": question,
        "cohort_id": active_cohort.cohort_id,
        "question_frame": active_cohort.question_frame,
        "cohort_generation": active_cohort.generation,
        "profile": profile_payload,
        "search_provider": stack.name,
        "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
        "evidence_mode": stack.evidence_mode,
        "retrieval_stack": {
            "name": stack.name,
            "evidence_mode": stack.evidence_mode,
            "agentic_tool_calls": stack.agentic_tool_calls,
            "model_prior_enabled": stack.include_model_prior,
            "external_provider": getattr(stack.external_provider, "name", None),
            "tolerate_external_errors": stack.tolerate_external_errors,
        },
        "results_per_query": results_per_query,
        "tool_retries": tool_retries,
        "person_count": count,
        "personas": [asdict(person) for person in personas],
        "retrieval_records": retrieval_records,
        "evidence_ledgers": ledgers,
        "search_tool_traces": tool_traces,
        "search_tool_usage": _summarize_tool_usage(tool_traces, model=model),
        "meta_model_ready": True,
        "next_step": "将 evidence_ledgers 与数字人分析输出一起交给元模型做阴影显影。",
    }
    validate_run_isolation(
        question,
        active_cohort.cohort_id,
        result["personas"],
        question_frame=result["question_frame"],
        cohort_generation=result["cohort_generation"],
        retrieval_records=result["retrieval_records"],
        evidence_ledgers=result["evidence_ledgers"],
    )
    return result


def assemble_retrieval_result(
    question: str,
    cohort: CohortBundle,
    *,
    provider_name: str,
    profile_payload: dict[str, Any] | None,
    results_per_query: int,
    tool_retries: int,
    model: str | None,
    stack: RetrievalStack,
    ordered_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble validated batch artifacts from independently saved items."""

    ledgers = [item["evidence_ledger"] for item in ordered_items]
    retrieval_records = [item["retrieval_record"] for item in ordered_items]
    tool_traces = [
        item["tool_trace"]
        for item in ordered_items
        if isinstance(item.get("tool_trace"), dict) and item["tool_trace"]
    ]
    result = {
        "question": question,
        "cohort_id": cohort.cohort_id,
        "question_frame": cohort.question_frame,
        "cohort_generation": cohort.generation,
        "profile": profile_payload,
        "search_provider": stack.name,
        "requested_provider": provider_name,
        "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
        "evidence_mode": stack.evidence_mode,
        "retrieval_stack": {
            "name": stack.name,
            "evidence_mode": stack.evidence_mode,
            "agentic_tool_calls": stack.agentic_tool_calls,
            "model_prior_enabled": stack.include_model_prior,
            "external_provider": getattr(stack.external_provider, "name", None),
            "tolerate_external_errors": stack.tolerate_external_errors,
        },
        "results_per_query": results_per_query,
        "tool_retries": tool_retries,
        "person_count": len(cohort.personas),
        "personas": [asdict(person) for person in cohort.personas],
        "retrieval_records": retrieval_records,
        "evidence_ledgers": ledgers,
        "search_tool_traces": tool_traces,
        "search_tool_usage": _summarize_tool_usage(tool_traces, model=model),
        "meta_model_ready": True,
        "next_step": "Review complete person chains with the meta-model.",
    }
    validate_run_isolation(
        question,
        cohort.cohort_id,
        result["personas"],
        question_frame=result["question_frame"],
        cohort_generation=result["cohort_generation"],
        retrieval_records=result["retrieval_records"],
        evidence_ledgers=result["evidence_ledgers"],
    )
    return result


def _run_one_retrieval_item(
    question: str,
    person: Any,
    cohort: CohortBundle,
    provider_name: str,
    results_per_query: int,
    model: str | None,
    tool_retries: int,
    *,
    stack: RetrievalStack | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    active_stack = stack or _build_retrieval_stack(provider_name)
    search_strategy = build_query_plan(question, person)
    candidates: list[SearchResult] = []
    semantic_source_decisions: list[dict[str, Any]] | None = None
    external_errors: list[str] = []
    tool_trace: dict[str, Any] = {}
    if active_stack.include_model_prior:
        candidates.extend(
            build_model_prior_results(
                question,
                person,
                limit=max(3, min(results_per_query, 5)),
            )
        )
    if active_stack.external_provider is not None:
        try:
            if active_stack.agentic_tool_calls:
                if not model:
                    raise RuntimeError("A model is required for agentic search tool calls.")
                last_tool_error: Exception | None = None
                trace = None
                for tool_attempt in range(tool_retries + 1):
                    try:
                        trace = run_agentic_search(
                            question,
                            person,
                            model=model,
                            provider=active_stack.external_provider,
                            client=(
                                client_factory()
                                if client_factory is not None
                                else None
                            ),
                            default_limit=results_per_query,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - retry transient model/search failures.
                        last_tool_error = exc
                        if isinstance(exc, DeepSeekAPIError) and not exc.retryable:
                            break
                        if _is_external_result_failure(exc):
                            break
                        if tool_attempt < tool_retries:
                            time.sleep(
                                retry_delay_seconds(exc, tool_attempt, cap=8)
                                if isinstance(exc, DeepSeekAPIError)
                                else min(2**tool_attempt, 4)
                            )
                if trace is None:
                    if isinstance(last_tool_error, DeepSeekAPIError) and not last_tool_error.retryable:
                        raise last_tool_error
                    fallback_attempts: list[dict[str, object]] = []
                    fallback_results = collect_candidates(
                        question,
                        person,
                        active_stack.external_provider,
                        results_per_query=results_per_query,
                        attempt_trace=fallback_attempts,
                    )
                    if not fallback_results:
                        raise RuntimeError(
                            f"Agentic search failed after retries: {last_tool_error}"
                        ) from last_tool_error
                    candidates.extend(fallback_results)
                    for attempt in fallback_attempts:
                        if attempt.get("planning_source") != "question_centered_recovery":
                            continue
                        search_strategy["search_tool_requests"].append(
                            {
                                "tool": "web_search",
                                "query": attempt.get("query"),
                                "why_this_person_searches_it": (
                                    "角色化窄查询没有返回材料，因此只放宽关键词到原问题；"
                                    "后续采信标准仍由本认知身份证决定。"
                                ),
                                "evidence_sought": "与原问题直接相关、可供本数字人筛选的外部材料",
                                "requested_limit": results_per_query,
                                "planning_source": "question_centered_recovery",
                                "status": attempt.get("status"),
                                "result_count": attempt.get("result_count", 0),
                            }
                        )
                    search_strategy["search_tool_mode"] = (
                        "persona_filtered_query_fallback"
                    )
                    search_strategy["selection_summary"] = (
                        "工具调用未取得可用外部结果；保留失败轨迹，并按同一认知身份证的预设查询补取外部材料。"
                    )
                    tool_trace = {
                        "person_id": person.id,
                        "mode": "persona_filtered_query_fallback",
                        "model": model,
                        "requests": search_strategy.get("search_tool_requests", []),
                        "result_count": len(fallback_results),
                        "source_decisions": [],
                        "selection_summary": search_strategy["selection_summary"],
                        "provider_errors": [str(last_tool_error)],
                        "fallback_attempts": fallback_attempts,
                        "usage": {},
                    }
                    external_errors.append(str(last_tool_error))
                else:
                    candidates.extend(trace.results)
                    semantic_source_decisions = trace.source_decisions
                    search_strategy["search_tool_requests"] = trace.requests
                    search_strategy["search_tool_mode"] = trace.mode
                    search_strategy["selection_summary"] = trace.selection_summary
                    tool_trace = {"person_id": person.id, **trace.to_dict()}
                    external_errors.extend(trace.provider_errors)
            else:
                last_external_error: Exception | None = None
                external_results: list[SearchResult] | None = None
                for external_attempt in range(tool_retries + 1):
                    try:
                        external_results = collect_candidates(
                            question,
                            person,
                            active_stack.external_provider,
                            results_per_query=results_per_query,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001 - retry transient search failures.
                        last_external_error = exc
                        if external_attempt < tool_retries:
                            time.sleep(min(2**external_attempt, 4))
                if external_results is None:
                    raise RuntimeError(
                        f"External retrieval failed after retries: {last_external_error}"
                    ) from last_external_error
                candidates.extend(external_results)
        except Exception as exc:  # noqa: BLE001 - record an explicit evidence gap when sources are empty.
            recoverable_empty_search = (
                active_stack.include_model_prior and _is_external_result_failure(exc)
            )
            if not active_stack.tolerate_external_errors and not recoverable_empty_search:
                raise
            external_errors.append(str(exc))
            if recoverable_empty_search:
                search_strategy["external_search_status"] = (
                    "unavailable_after_question_centered_recovery"
                )
                search_strategy["selection_summary"] = (
                    "角色化查询与原问题放宽查询均未取得外部结果。保留搜索失败轨迹，"
                    "本数字人只能使用搜索前先验继续思考；这项证据缺口必须进入元模型诊断。"
                )
                tool_trace = {
                    "person_id": person.id,
                    "mode": "explicit_external_evidence_gap",
                    "model": model,
                    "requests": search_strategy.get("search_tool_requests", []),
                    "result_count": 0,
                    "source_decisions": [],
                    "selection_summary": search_strategy["selection_summary"],
                    "provider_errors": [str(exc)],
                    "usage": {},
                    "external_evidence_gap": True,
                }

    ledger = build_evidence_ledger(
        question,
        person,
        search_strategy,
        candidates,
        semantic_source_decisions=semantic_source_decisions,
    )
    retrieval_record = {
        "person_id": person.id,
        "person_name": person.name,
        "search_strategy": search_strategy,
        "retrieval_stack": {
            "name": active_stack.name,
            "evidence_mode": active_stack.evidence_mode,
            "agentic_tool_calls": active_stack.agentic_tool_calls,
            "model_prior_enabled": active_stack.include_model_prior,
            "external_provider": getattr(active_stack.external_provider, "name", None),
            "external_errors": external_errors,
        },
        "external_search_status": search_strategy.get(
            "external_search_status",
            "available" if any(
                item.retrieval_layer == "external_search" for item in candidates
            ) else "not_applicable",
        ),
        "question_context_id": cohort.cohort_id,
        "question_frame": cohort.question_frame,
        "candidate_sources": serialize_results(candidates),
        "ledger_summary": {
            "accepted": len(ledger.accepted_sources),
            "rejected": len(ledger.rejected_sources),
            "ignored": len(ledger.ignored_sources),
            "information_shadow_hints": ledger.information_shadow_hints,
        },
    }
    return {
        "schema": "bme.retrieval-item.v2",
        "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
        "status": "succeeded",
        "question": question,
        "cohort_id": cohort.cohort_id,
        "person_id": person.id,
        "provider_name": provider_name,
        "results_per_query": results_per_query,
        "model": model,
        "retrieval_record": retrieval_record,
        "evidence_ledger": asdict(ledger),
        "tool_trace": tool_trace,
    }


def _load_retrieval_checkpoint(
    path: Path | None,
    *,
    question: str,
    cohort_id: str,
    person_id: str,
    provider_name: str,
    results_per_query: int,
    model: str | None,
) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        item = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(item, dict) or item.get("status") != "succeeded":
        return {}
    expected = {
        "schema": "bme.retrieval-item.v2",
        "retrieval_implementation_revision": RETRIEVAL_IMPLEMENTATION_REVISION,
        "question": question,
        "cohort_id": cohort_id,
        "person_id": person_id,
        "provider_name": provider_name,
        "results_per_query": results_per_query,
        "model": model,
    }
    if any(item.get(key) != value for key, value in expected.items()):
        return {}
    record = item.get("retrieval_record") or {}
    ledger = item.get("evidence_ledger") or {}
    if (
        record.get("person_id") != person_id
        or record.get("question_context_id") != cohort_id
        or (record.get("question_frame") or {}).get("exact_question") != question
        or ledger.get("person_id") != person_id
        or ledger.get("question") != question
    ):
        return {}
    return item


def run_shadow_diagnosis(
    question: str,
    *,
    profile_name: str | None = None,
    person_count: int | None = None,
    provider_name: str = "mock",
    results_per_query: int = 5,
) -> dict[str, Any]:
    retrieval = run_filtered_retrieval(
        question,
        profile_name=profile_name,
        person_count=person_count,
        provider_name=provider_name,
        results_per_query=results_per_query,
    )
    diagnosis = diagnose_cognitive_shadows(
        question,
        retrieval["personas"],
        retrieval["evidence_ledgers"],
    )
    puzzle = build_shadow_puzzle(question, diagnosis)
    return {
        "question": question,
        "cohort_id": retrieval["cohort_id"],
        "profile": retrieval["profile"],
        "search_provider": retrieval["search_provider"],
        "person_count": retrieval["person_count"],
        "retrieval": {
            "results_per_query": retrieval["results_per_query"],
            "records": retrieval["retrieval_records"],
            "evidence_ledgers": retrieval["evidence_ledgers"],
        },
        "meta_model": diagnosis,
        "shadow_puzzle": puzzle,
    }


def run_one_person_live(
    question: str,
    person_id: str,
    *,
    model: str,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    personas = select_personas(question, count=50)
    person = next(
        (
            candidate
            for candidate in personas
            if candidate.id == person_id or candidate.id.endswith(f"_{person_id}")
        ),
        None,
    )
    if person is None:
        known = ", ".join(candidate.id for candidate in personas)
        raise ValueError(f"Unknown person_id {person_id!r}. Known: {known}")

    client = DeepSeekClient(model=model)
    response = client.chat(
        render_digital_person_messages(question, person),
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    parsed = parse_json_content(response.content)
    return {
        "person": asdict(person),
        "output": parsed,
        "usage": response.usage,
    }


def run_meta_model_live(
    question: str,
    person_outputs: list[dict[str, Any]],
    *,
    model: str,
    max_tokens: int = 4096,
    diagnostic_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = DeepSeekClient(model=model)
    response = client.chat(
        render_meta_model_messages(question, person_outputs, diagnostic_report),
        temperature=0.3,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return {
        "output": parse_json_content(response.content),
        "usage": response.usage,
    }


def _is_external_result_failure(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "at least one external result",
            "returned no parseable results",
            "all providers returned zero results",
            "federated html search failed",
            "no external results",
            "returned zero results",
        )
    )


def _get_search_provider(provider_name: str):
    if provider_name == "mock":
        return MockSearchProvider()
    if provider_name == "google":
        return GoogleCustomSearchProvider()
    if provider_name == "google-news-rss":
        return GoogleNewsRssSearchProvider()
    if provider_name == "baidu-html":
        return BaiduHtmlSearchProvider()
    if provider_name == "baidu-json":
        return BaiduJsonSearchProvider()
    if provider_name == "bing-html":
        return BingHtmlSearchProvider()
    if provider_name == "bing-rss":
        return BingRssSearchProvider()
    if provider_name == "duckduckgo-html":
        return DuckDuckGoHtmlSearchProvider()
    if provider_name == "federated-html":
        return FederatedHtmlSearchProvider()
    if provider_name == "auto-cn":
        return choose_regional_provider("cn")
    if provider_name == "auto-global":
        return choose_regional_provider("global")
    raise ValueError(
        f"Unknown search provider {provider_name!r}. "
        "Available: mock, google, google-news-rss, baidu-html, baidu-json, bing-html, bing-rss, duckduckgo-html, "
        "federated-html, auto-cn, auto-global"
    )


def _build_retrieval_stack(provider_name: str) -> RetrievalStack:
    if provider_name in {"prior", "model-prior", "deepseek-prior"}:
        return RetrievalStack(name="model-prior", include_model_prior=True, evidence_mode="prior_only")
    if provider_name in {"hybrid", "hybrid-global"}:
        provider = choose_regional_provider("global")
        return RetrievalStack(
            name=f"model-prior+{provider.name}",
            external_provider=provider,
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="hybrid_live",
        )
    if provider_name == "hybrid-cn":
        provider = choose_regional_provider("cn")
        return RetrievalStack(
            name=f"model-prior+{provider.name}",
            external_provider=provider,
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="hybrid_live",
        )
    if provider_name == "hybrid-mock":
        return RetrievalStack(
            name="model-prior+mock",
            external_provider=MockSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="test_fixture",
        )
    if provider_name == "hybrid-google":
        return RetrievalStack(
            name="model-prior+google",
            external_provider=GoogleCustomSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="hybrid_live",
        )
    if provider_name == "hybrid-baidu-json":
        return RetrievalStack(
            name="model-prior+baidu-json",
            external_provider=BaiduJsonSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="hybrid_live",
        )
    if provider_name == "hybrid-baidu-html":
        return RetrievalStack(
            name="model-prior+baidu-html",
            external_provider=BaiduHtmlSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=True,
            evidence_mode="hybrid_live",
        )
    if provider_name in {"three-layer", "agentic-federated"}:
        return RetrievalStack(
            name="model-prior+deepseek-tools+federated-html",
            external_provider=FederatedHtmlSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=False,
            evidence_mode="three_layer_live",
            agentic_tool_calls=True,
        )
    if provider_name == "hybrid-federated":
        return RetrievalStack(
            name="model-prior+federated-html",
            external_provider=FederatedHtmlSearchProvider(),
            include_model_prior=True,
            tolerate_external_errors=False,
            evidence_mode="hybrid_live",
        )
    provider = _get_search_provider(provider_name)
    mode = "test_fixture" if isinstance(provider, MockSearchProvider) else "live_external"
    return RetrievalStack(name=provider.name, external_provider=provider, evidence_mode=mode)


def _summarize_tool_usage(traces: list[dict[str, Any]], *, model: str | None) -> dict[str, Any]:
    prompt_tokens = sum(int((trace.get("usage") or {}).get("prompt_tokens") or 0) for trace in traces)
    completion_tokens = sum(int((trace.get("usage") or {}).get("completion_tokens") or 0) for trace in traces)
    total_tokens = sum(int((trace.get("usage") or {}).get("total_tokens") or 0) for trace in traces)
    return {
        "model": model,
        "sessions": len(traces),
        "calls": sum(int((trace.get("usage") or {}).get("api_calls") or 1) for trace in traces),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
        "executed_search_requests": sum(len(trace.get("requests") or []) for trace in traces),
        "external_result_count": sum(int(trace.get("result_count") or 0) for trace in traces),
        "provider_error_count": sum(len(trace.get("provider_errors") or []) for trace in traces),
        "provider_skip_count": sum(
            len(trace.get("provider_skips") or []) for trace in traces
        ),
        "timing": {
            "planning_seconds_sum": round(
                sum(
                    float(trace.get("planning_duration_seconds") or 0)
                    for trace in traces
                ),
                3,
            ),
            "external_search_seconds_sum": round(
                sum(
                    float(trace.get("external_search_duration_seconds") or 0)
                    for trace in traces
                ),
                3,
            ),
            "source_filter_seconds_sum": round(
                sum(
                    float(trace.get("source_filter_duration_seconds") or 0)
                    for trace in traces
                ),
                3,
            ),
            "slowest_session_seconds": round(
                max(
                    (
                        float(trace.get("total_duration_seconds") or 0)
                        for trace in traces
                    ),
                    default=0.0,
                ),
                3,
            ),
        },
    }
