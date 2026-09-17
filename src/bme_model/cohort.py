from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .json_utils import parse_json_content
from .model import DigitalPerson, InformationFilter, select_personas
from .providers import (
    ChatClient,
    DeepSeekAPIError,
    DeepSeekClient,
    bounded_request_timeout_seconds,
    retry_delay_seconds,
)
from .question_frame import build_reasoning_grammar, merge_evidence_dimensions


COHORT_SYSTEM = """你是盲人摸象模型的“问题隔离与数字人群生成器”。

你只知道本次输入的 CURRENT_QUESTION。不得假设存在上一道题，不得复用任何其他问题的事实、案例、行业对象、结论、时间窗口或专用术语。

先为 CURRENT_QUESTION 建立问题框架，再生成全新的 AI 数字人群。数字人不是答案生成器，而是带有局限、盲区、价值权重和信息过滤器的观察者。固定的只有九维认知参数空间；职业身份、专业知识、证据偏好和检索语言必须针对本题重新生成。

九个维度：角色与专业边界、认知框架、价值权重、时间尺度、风险态度、分析层级、隐藏偏见、信息过滤器、气质。

硬约束：
1. 不回答 CURRENT_QUESTION，不预先分配结论。
2. 至少三分之一的人拥有直接相关的领域或机制知识；这些人必须把 relevance_class 明确标为 direct_domain 或 direct_mechanism。
   relevance_class 每人只能填写一个值，允许值恰好为：direct_domain、direct_mechanism、adjacent、affected、executor、counter、institutional。不得复制整串选项，不得使用斜杠组合值。
3. 同时包含相邻领域、受影响者、执行者、反证者和制度视角，但不得拿无关职业凑数。
4. 每个人的信息来源与证据类型必须与 CURRENT_QUESTION 有可解释关系；偏食可以存在，无关材料不能伪装成盲区。
5. 不得出现“上一题”“延续此前”“同前”等跨题引用。
6. 输出严格 JSON，不输出 Markdown。
7. JSON 根节点只能有 question_frame 和 personas；不要回显 CURRENT_QUESTION、person_count、minimum_direct_domain_experts 或 required_output 外壳。
8. 一次生成就要完成认知覆盖：先让 question_frame 写清本题需要检查的证据维度，再让整个人群共同覆盖这些维度、定义分支、时间尺度、反证入口和受影响者位置；不得生成九维身份证近似复制的人物。
9. 不预设事后追加数字人。人群必须在本次生成中同时包含主流机制、反向机制、边界条件和容易被共同漏看的位置。
"""

COHORT_DIRECT_REPAIR_SYSTEM = """你是盲人摸象模型的“数字人群定点补全器”。

你只修复当前问题中缺少的直接领域或直接机制观察者，不回答问题，不改变其他数字人，也不把相邻角色改标签冒充专家。

硬约束：
1. 每个补充角色的 relevance_class 只能是 direct_domain 或 direct_mechanism。
2. 角色必须真正具备分析 CURRENT_QUESTION 所需的直接领域知识或直接因果机制知识。
3. 补充角色之间、与现有人群之间不得身份重复；优先补足现有人群缺失的直接机制。
4. 每个角色必须完整填写九维认知身份证和信息过滤器，同时保留真实的专业边界、盲区和偏见。
5. 不预先分配答案，不输出对 CURRENT_QUESTION 的结论。
6. 只输出严格 JSON：根节点只有 replacements，数量必须与 replacement_count 完全一致。
"""

COHORT_SHORTAGE_REPAIR_SYSTEM = """你是盲人摸象模型的“数字人群缺席补位器”。

你只为 CURRENT_QUESTION 补齐缺失席位，不回答问题，不改写已生成人物，也不重复现有身份。

硬约束：
1. 只输出新增人物；根节点只能有 additions，数量必须与 addition_count 完全一致。
2. 至少生成 minimum_direct_additions 个 direct_domain 或 direct_mechanism 人物，并优先补足现有人群缺失的直接机制。
3. 其余新增人物应补足缺失的相邻领域、受影响者、执行者、反证者或制度视角，不得拿无关职业凑数。
4. 每个人必须完整填写九维认知身份证和信息过滤器，并保留真实的专业边界、盲区和偏见。
5. 不预先分配答案，不输出对 CURRENT_QUESTION 的结论。
6. relevance_class 每人只能填写一个允许值，不得复制选项串或使用斜杠组合。
"""


class DirectCoverageError(ValueError):
    def __init__(self, found: int, required: int) -> None:
        self.found = found
        self.required = required
        super().__init__(
            "too few directly relevant domain or mechanism personas: "
            f"found {found}, required {required}"
        )


class PersonaCountError(ValueError):
    def __init__(self, found: int, required: int) -> None:
        self.found = found
        self.required = required
        super().__init__(f"expected {required} personas, got {found}")


@dataclass(frozen=True)
class CohortBundle:
    cohort_id: str
    question: str
    question_frame: dict[str, Any]
    personas: list[DigitalPerson]
    generation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cohort_id": self.cohort_id,
            "question": self.question,
            "question_frame": self.question_frame,
            "personas": [asdict(person) for person in self.personas],
            "generation": self.generation,
        }


def generate_live_cohort(
    question: str,
    count: int,
    *,
    model: str,
    retries: int = 2,
    max_tokens: int | None = None,
    client_factory: Callable[[], ChatClient] | None = None,
    deadline_epoch: float | None = None,
) -> CohortBundle:
    if count < 1:
        raise ValueError("count must be positive")
    request = _cohort_request(question, count)
    token_limit = max_tokens or (16384 if count <= 24 else 32768)
    last_error: Exception | None = None
    started = time.perf_counter()
    usage_records: list[dict[str, Any]] = []
    attempt_errors: list[str] = []
    factory = client_factory or (
        lambda: DeepSeekClient(
            model=model,
            timeout_seconds=bounded_request_timeout_seconds(
                deadline_epoch, 180
            ),
        )
    )
    base_messages = [
        {"role": "system", "content": COHORT_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                request, ensure_ascii=False, separators=(",", ":")
            ),
        },
    ]
    messages = list(base_messages)

    for attempt in range(1, retries + 2):
        response = None
        try:
            response = factory().chat(
                messages,
                temperature=0.85,
                max_tokens=token_limit,
                response_format={"type": "json_object"},
                thinking={"type": "disabled"},
            )
            usage_records.append(response.usage)
            parsed = parse_json_content(response.content)
            cohort_id = f"{_question_fingerprint(question)}-{uuid.uuid4().hex[:8]}"
            final_payload = parsed
            targeted_repairs: list[dict[str, Any]] = []
            while True:
                try:
                    frame, personas, selection = _normalize_generated_payload(
                        final_payload, question, count, cohort_id
                    )
                    break
                except PersonaCountError as count_error:
                    if any(
                        item.get("policy") == "targeted_persona_shortage_completion_v1"
                        for item in targeted_repairs
                    ):
                        raise
                    (
                        final_payload,
                        repair_response,
                        repair_metadata,
                    ) = _repair_persona_shortage(
                        final_payload,
                        question,
                        count,
                        token_limit=token_limit,
                        count_error=count_error,
                        client_factory=factory,
                    )
                except DirectCoverageError as coverage_error:
                    if any(
                        item.get("policy") == "targeted_direct_coverage_repair_v1"
                        for item in targeted_repairs
                    ):
                        raise
                    (
                        final_payload,
                        repair_response,
                        repair_metadata,
                    ) = _repair_direct_coverage(
                        final_payload,
                        question,
                        count,
                        token_limit=token_limit,
                        coverage_error=coverage_error,
                        client_factory=factory,
                    )
                usage_records.append(repair_response.usage)
                targeted_repairs.append(repair_metadata)
            return CohortBundle(
                cohort_id=cohort_id,
                question=question,
                question_frame=frame,
                personas=personas,
                generation={
                    "mode": (
                        "llm_question_native_with_targeted_repair"
                        if targeted_repairs
                        else "llm_question_native"
                    ),
                    "model": model,
                    "attempts": attempt,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "usage": _merge_usage_records(usage_records),
                    "attempt_errors": attempt_errors,
                    "context_isolation": "only_current_question",
                    "payload_hash": hashlib.sha256(
                        json.dumps(
                            final_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:16],
                    "candidate_selection": selection,
                    "targeted_repairs": targeted_repairs,
                    "cohort_geometry": _cohort_geometry(personas, frame),
                    "epistemic_passes": 1,
                },
            )
        except DeepSeekAPIError as exc:
            last_error = exc
            attempt_errors.append(str(exc))
            if not exc.retryable:
                break
            if attempt <= retries:
                time.sleep(retry_delay_seconds(exc, attempt - 1))
        except Exception as exc:  # noqa: BLE001 - validation failures are retryable model failures.
            last_error = exc
            attempt_errors.append(str(exc))
            response_content = response.content if response is not None else ""
            if response_content:
                finish_reason = str(
                    (((response.raw.get("choices") or [{}])[0]) or {}).get("finish_reason")
                    or ""
                )
                if finish_reason == "length":
                    token_limit = min(max(token_limit + 4096, int(token_limit * 1.5)), 49152)
                messages = list(base_messages) + [
                    {"role": "assistant", "content": response_content},
                    {
                        "role": "user",
                        "content": (
                            "上一版数字人群没有通过完整性和问题隔离校验。"
                            "请完全重写严格 JSON，并修复：" + str(exc)
                        ),
                    },
                ]
            if attempt <= retries:
                time.sleep(min(2 ** (attempt - 1), 4))

    if isinstance(last_error, DeepSeekAPIError):
        raise last_error
    raise RuntimeError(f"Question-native cohort generation failed: {last_error}") from last_error


def _merge_usage_records(records: list[dict[str, Any]]) -> dict[str, int]:
    prompt = sum(int(item.get("prompt_tokens") or item.get("input_tokens") or 0) for item in records)
    completion = sum(
        int(item.get("completion_tokens") or item.get("output_tokens") or 0)
        for item in records
    )
    total = sum(int(item.get("total_tokens") or 0) for item in records)
    return {
        "api_calls": len(records),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
    }


def build_fixture_cohort(question: str, count: int) -> CohortBundle:
    """Build a network-free cohort for protocol tests, never for disguised live use."""
    frame = build_fixture_question_frame(question)
    personas = select_personas(question, count=count)
    return CohortBundle(
        cohort_id=_question_fingerprint(question),
        question=question,
        question_frame=frame,
        personas=personas,
        generation={
            "mode": "deterministic_question_isolated_fixture",
            "model": None,
            "attempts": 0,
            "duration_seconds": 0.0,
            "usage": {},
            "context_isolation": "only_current_question",
            "cohort_geometry": _cohort_geometry(personas, frame),
            "epistemic_passes": 1,
        },
    )


def build_fixture_question_frame(question: str) -> dict[str, Any]:
    scope = {
        "population": "待界定",
        "geography": "待界定",
        "time_horizon": "待界定",
    }
    grammar = build_reasoning_grammar(
        question,
        _question_type(question),
        disputed_definitions=[],
        scope=scope,
    )
    dimensions = merge_evidence_dimensions(
        [
            "直接测量",
            "因果机制",
            "反例与失败条件",
            "基准率与对照",
            "受影响者直接经验",
            "制度执行证据",
        ],
        grammar,
    )
    return {
        "context_id": _question_fingerprint(question),
        "exact_question": question,
        "normalized_question": question.strip(),
        "judgment_target": question,
        "question_type": _question_type(question),
        "question_types": grammar["question_types"],
        "primary_question_type": grammar["primary_question_type"],
        "key_terms": [question],
        "disputed_definitions": [],
        "scope": scope,
        "relevant_knowledge_domains": ["由本题专属数字人进一步界定"],
        "relevant_evidence_dimensions": dimensions,
        "misleading_substitutions": [],
        "reasoning_grammar": grammar,
        "one_pass_coverage": {
            "coverage_targets": dimensions,
            "automatic_second_round": False,
            "remaining_gaps_become_explicit_unknowns": True,
        },
        "carryover_policy": "禁止继承任何其他问题的事实、案例、结论和专用措辞。",
    }


def _cohort_request(question: str, count: int) -> dict[str, Any]:
    minimum_experts = max(1, math.ceil(count / 3))
    return {
        "CURRENT_QUESTION": question,
        "person_count": count,
        "minimum_direct_domain_experts": minimum_experts,
        "required_output": {
            "question_frame": {
                "judgment_target": "本题究竟要判断什么",
                "question_type": "预测/解释/规范/决策/混合",
                "key_terms": [],
                "disputed_definitions": [],
                "scope": {"population": "", "geography": "", "time_horizon": ""},
                "relevant_knowledge_domains": [],
                "relevant_evidence_dimensions": [],
                "misleading_substitutions": [],
            },
            "personas": [
                {
                    "name": "",
                    "role_summary": "",
                    "relevance_class": "direct_domain",
                    "cognitive_frames": [],
                    "expertise_strong": [],
                    "expertise_weak": [],
                    "values": {"value_name": 0.0},
                    "time_horizon": "",
                    "risk_attitude": "",
                    "analysis_levels": [],
                    "hidden_biases": [],
                    "temperament": "",
                    "information_filter": {
                        "trusted_sources": [],
                        "distrusted_sources": [],
                        "preferred_evidence": [],
                        "ignored_evidence": [],
                        "query_style": [],
                    },
                    "topic_relevance": "为什么这个人对本题有直接、相邻或必要的反向价值",
                }
            ],
        },
    }


def _normalize_generated_payload(
    payload: dict[str, Any],
    question: str,
    count: int,
    cohort_id: str,
) -> tuple[dict[str, Any], list[DigitalPerson], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("cohort output is not an object")
    declared_question = str(payload.get("CURRENT_QUESTION", "")).strip()
    if declared_question and declared_question != question:
        raise ValueError("cohort output declares a different question")
    payload = _unwrap_cohort_payload(payload)
    raw_frame = payload.get("question_frame")
    raw_personas = payload.get("personas")
    if not isinstance(raw_frame, dict):
        raise ValueError("question_frame is missing")
    if not isinstance(raw_personas, list):
        raise ValueError("personas is missing or is not an array")
    if len(raw_personas) < count:
        raise PersonaCountError(len(raw_personas), count)

    raw_personas, selection = _select_persona_payloads(raw_personas, count)

    frame = _normalize_frame(raw_frame, question, cohort_id)
    personas: list[DigitalPerson] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_personas, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"persona {index} is not an object")
        person = _normalize_person(raw, question, cohort_id, index)
        if person.name in seen_names:
            raise ValueError(f"duplicate persona name: {person.name}")
        seen_names.add(person.name)
        personas.append(person)

    direct_count = sum(1 for raw in raw_personas if _is_directly_relevant(raw))
    required_direct = max(1, math.ceil(count / 3))
    if direct_count < required_direct:
        raise DirectCoverageError(direct_count, required_direct)
    return frame, personas, selection


def _repair_persona_shortage(
    payload: dict[str, Any],
    question: str,
    count: int,
    *,
    token_limit: int,
    count_error: PersonaCountError,
    client_factory: Callable[[], ChatClient],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    unwrapped = _unwrap_cohort_payload(payload)
    raw_frame = unwrapped.get("question_frame")
    existing = unwrapped.get("personas")
    if not isinstance(raw_frame, dict) or not isinstance(existing, list):
        raise ValueError("persona-shortage repair requires a valid cohort payload")
    shortage = count_error.required - count_error.found
    maximum_targeted_shortage = max(2, math.ceil(count * 0.1))
    if shortage <= 0 or shortage > maximum_targeted_shortage:
        raise ValueError(
            "persona shortage is too large for targeted completion: "
            f"missing {shortage}, limit {maximum_targeted_shortage}"
        )
    if any(not isinstance(raw, dict) for raw in existing):
        raise ValueError("existing cohort contains a damaged persona object")

    direct_before = sum(_is_directly_relevant(raw) for raw in existing)
    required_direct = max(1, math.ceil(count / 3))
    minimum_direct_additions = min(
        shortage,
        max(0, required_direct - direct_before),
    )
    present_classes = {
        _canonical_relevance_class(raw)
        for raw in existing
        if isinstance(raw, dict)
    }
    perspective_classes = {
        "adjacent",
        "affected",
        "executor",
        "counter",
        "institutional",
    }
    summaries = [
        {
            "slot": index + 1,
            "name": str(raw.get("name", "")).strip(),
            "role_summary": str(raw.get("role_summary", "")).strip(),
            "relevance_class": _canonical_relevance_class(raw),
            "expertise_strong": _strings(raw.get("expertise_strong")),
        }
        for index, raw in enumerate(existing)
    ]
    request = {
        "CURRENT_QUESTION": question,
        "question_frame": raw_frame,
        "addition_count": shortage,
        "minimum_direct_additions": minimum_direct_additions,
        "missing_perspective_classes": sorted(perspective_classes - present_classes),
        "new_slots": list(range(count_error.found + 1, count_error.required + 1)),
        "current_persona_summaries": summaries,
        "required_persona_schema": _cohort_request(question, count)[
            "required_output"
        ]["personas"][0],
    }
    response = client_factory().chat(
        [
            {"role": "system", "content": COHORT_SHORTAGE_REPAIR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    request, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        temperature=0.65,
        max_tokens=min(max(4096, shortage * 3072), token_limit, 16384),
        response_format={"type": "json_object"},
        thinking={"type": "disabled"},
    )
    repaired = parse_json_content(response.content)
    additions = _unwrap_additions(repaired)
    if len(additions) != shortage:
        raise ValueError(
            "targeted persona-shortage completion returned "
            f"{len(additions)} additions; expected {shortage}"
        )
    if any(not isinstance(item, dict) for item in additions):
        raise ValueError("targeted persona-shortage completion returned an invalid persona")
    direct_additions = sum(_is_directly_relevant(item) for item in additions)
    if direct_additions < minimum_direct_additions:
        raise ValueError(
            "targeted persona-shortage completion did not supply enough direct experts: "
            f"found {direct_additions}, required {minimum_direct_additions}"
        )
    existing_names = {str(raw.get("name", "")).strip() for raw in existing}
    addition_names = [str(raw.get("name", "")).strip() for raw in additions]
    if any(not name for name in addition_names):
        raise ValueError("targeted persona-shortage completion returned an unnamed persona")
    if len(set(addition_names)) != len(addition_names) or existing_names.intersection(
        addition_names
    ):
        raise ValueError("targeted persona-shortage completion duplicated an identity")

    merged = deepcopy(existing) + deepcopy(additions)
    return (
        {"question_frame": deepcopy(raw_frame), "personas": merged},
        response,
        {
            "policy": "targeted_persona_shortage_completion_v1",
            "found_before": count_error.found,
            "required": count_error.required,
            "addition_count": shortage,
            "direct_count_before": direct_before,
            "minimum_direct_additions": minimum_direct_additions,
            "direct_additions": direct_additions,
            "preserved_person_count": count_error.found,
            "additions": [
                {
                    "slot": count_error.found + index + 1,
                    "added_name": str(addition.get("name", "")).strip(),
                    "added_relevance_class": _canonical_relevance_class(addition),
                }
                for index, addition in enumerate(additions)
            ],
        },
    )


def _repair_direct_coverage(
    payload: dict[str, Any],
    question: str,
    count: int,
    *,
    token_limit: int,
    coverage_error: DirectCoverageError,
    client_factory: Callable[[], ChatClient],
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    unwrapped = _unwrap_cohort_payload(payload)
    raw_frame = unwrapped.get("question_frame")
    candidates = unwrapped.get("personas")
    if not isinstance(raw_frame, dict) or not isinstance(candidates, list):
        raise ValueError("direct-coverage repair requires a valid cohort payload")
    selected, original_selection = _select_persona_payloads(candidates, count)
    shortage = coverage_error.required - coverage_error.found
    targets = _choose_direct_repair_targets(selected, shortage)
    summaries = [
        {
            "slot": index + 1,
            "name": str(raw.get("name", "")).strip(),
            "role_summary": str(raw.get("role_summary", "")).strip(),
            "relevance_class": _canonical_relevance_class(raw),
            "expertise_strong": _strings(raw.get("expertise_strong")),
        }
        for index, raw in enumerate(selected)
        if isinstance(raw, dict)
    ]
    request = {
        "CURRENT_QUESTION": question,
        "question_frame": raw_frame,
        "replacement_count": shortage,
        "replace_slots": [index + 1 for index in targets],
        "current_persona_summaries": summaries,
        "required_persona_schema": _cohort_request(question, count)[
            "required_output"
        ]["personas"][0],
    }
    response = client_factory().chat(
        [
            {"role": "system", "content": COHORT_DIRECT_REPAIR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    request, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ],
        temperature=0.65,
        max_tokens=min(max(4096, shortage * 3072), token_limit, 12288),
        response_format={"type": "json_object"},
        thinking={"type": "disabled"},
    )
    repaired = parse_json_content(response.content)
    replacements = _unwrap_replacements(repaired)
    if len(replacements) != shortage:
        raise ValueError(
            "targeted direct-coverage repair returned "
            f"{len(replacements)} replacements; expected {shortage}"
        )
    if any(not isinstance(item, dict) or not _is_directly_relevant(item) for item in replacements):
        raise ValueError("targeted repair returned a non-direct replacement")

    merged = deepcopy(selected)
    replaced = []
    for target, replacement in zip(targets, replacements):
        previous = merged[target]
        merged[target] = replacement
        replaced.append(
            {
                "slot": target + 1,
                "removed_name": (
                    str(previous.get("name", "")).strip()
                    if isinstance(previous, dict)
                    else ""
                ),
                "added_name": str(replacement.get("name", "")).strip(),
                "added_relevance_class": _canonical_relevance_class(replacement),
            }
        )
    return (
        {"question_frame": deepcopy(raw_frame), "personas": merged},
        response,
        {
            "policy": "targeted_direct_coverage_repair_v1",
            "found_before": coverage_error.found,
            "required": coverage_error.required,
            "replacement_count": shortage,
            "replacements": replaced,
            "preserved_person_count": count - shortage,
            "original_candidate_selection": original_selection,
        },
    )


def _choose_direct_repair_targets(
    personas: list[Any], shortage: int
) -> list[int]:
    if shortage <= 0:
        return []
    classes = [
        _canonical_relevance_class(raw) if isinstance(raw, dict) else "unknown"
        for raw in personas
    ]
    counts = Counter(classes)
    remaining = [
        index
        for index, raw in enumerate(personas)
        if isinstance(raw, dict) and not _is_directly_relevant(raw)
    ]
    targets: list[int] = []
    for _ in range(shortage):
        if not remaining:
            raise ValueError("no non-direct persona is available for targeted repair")

        def replacement_cost(index: int) -> tuple[int, int, int]:
            relevance = classes[index]
            preserves_class = relevance == "unknown" or counts[relevance] > 1
            completeness = _raw_persona_completeness(personas[index])
            return (0 if preserves_class else 1, completeness, -index)

        target = min(remaining, key=replacement_cost)
        remaining.remove(target)
        targets.append(target)
        counts[classes[target]] -= 1
    return sorted(targets)


def _unwrap_replacements(payload: dict[str, Any]) -> list[Any]:
    if isinstance(payload.get("replacements"), list):
        return payload["replacements"]
    for key in ("required_output", "output", "result", "data"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and isinstance(candidate.get("replacements"), list):
            return candidate["replacements"]
    return []


def _unwrap_additions(payload: dict[str, Any]) -> list[Any]:
    if isinstance(payload.get("additions"), list):
        return payload["additions"]
    for key in ("required_output", "output", "result", "data"):
        candidate = payload.get(key)
        if isinstance(candidate, dict) and isinstance(candidate.get("additions"), list):
            return candidate["additions"]
    return []


def _select_persona_payloads(
    candidates: list[Any],
    count: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Choose a compliant cohort when the model returns a small surplus.

    Models occasionally return 25 candidates for a 24-person request. Rejecting
    that otherwise useful response costs minutes and another API call. We keep
    the exact output contract by selecting the strongest group, while treating
    direct expertise, perspective coverage, unique identities, and field
    completeness as constraints rather than truncating by position.
    """
    candidate_count = len(candidates)
    selected = list(enumerate(candidates))
    while len(selected) > count:
        alternatives: list[tuple[tuple[Any, ...], int, list[tuple[int, Any]]]] = []
        for removal_position in range(len(selected)):
            remaining = selected[:removal_position] + selected[removal_position + 1 :]
            score = _cohort_group_score([raw for _, raw in remaining], count)
            # On a complete tie, discard the later surplus candidate. This keeps
            # selection deterministic without making array order the quality rule.
            removed_original_index = selected[removal_position][0]
            alternatives.append((score + (removed_original_index,), removal_position, remaining))
        _, _, selected = max(alternatives, key=lambda item: item[0])

    chosen_indexes = {original_index for original_index, _ in selected}
    excluded = [
        {
            "candidate_index": index + 1,
            "name": str(raw.get("name", "")).strip() if isinstance(raw, dict) else "",
            "relevance_class": (
                str(raw.get("relevance_class", "")).strip() if isinstance(raw, dict) else ""
            ),
        }
        for index, raw in enumerate(candidates)
        if index not in chosen_indexes
    ]
    return [raw for _, raw in selected], {
        "policy": "constraint_preserving_surplus_selection_v1",
        "candidate_count": candidate_count,
        "selected_count": count,
        "surplus_count": candidate_count - count,
        "excluded_candidates": excluded,
    }


def _cohort_group_score(group: list[Any], count: int) -> tuple[Any, ...]:
    valid_objects = [raw for raw in group if isinstance(raw, dict)]
    names = [str(raw.get("name", "")).strip() for raw in valid_objects]
    nonempty_names = [name for name in names if name]
    unique_names = len(set(nonempty_names))
    no_invalid_objects = len(valid_objects) == len(group)
    no_duplicate_or_empty_names = len(nonempty_names) == len(group) and unique_names == len(group)

    classes = [_canonical_relevance_class(raw) for raw in valid_objects]
    direct_count = sum(1 for raw in valid_objects if _is_directly_relevant(raw))
    required_direct = max(1, math.ceil(count / 3))
    perspective_classes = {"adjacent", "affected", "executor", "counter", "institutional"}
    perspective_coverage = len(perspective_classes.intersection(classes))
    known_classes = sum(1 for value in classes if value != "unknown")
    identity_signatures = {
        _raw_identity_signature(raw) for raw in valid_objects
    }
    frame_diversity = len(
        {
            str(value).strip().casefold()
            for raw in valid_objects
            for value in raw.get("cognitive_frames", []) or []
            if str(value).strip()
        }
    )
    filter_diversity = len(
        {
            str(value).strip().casefold()
            for raw in valid_objects
            for value in (raw.get("information_filter") or {}).get(
                "preferred_evidence", []
            )
            or []
            if str(value).strip()
        }
    )
    horizon_diversity = len(
        {
            str(raw.get("time_horizon") or "").strip().casefold()
            for raw in valid_objects
            if str(raw.get("time_horizon") or "").strip()
        }
    )
    risk_diversity = len(
        {
            str(raw.get("risk_attitude") or "").strip().casefold()
            for raw in valid_objects
            if str(raw.get("risk_attitude") or "").strip()
        }
    )
    completeness_scores = [
        _raw_persona_completeness(raw) for raw in valid_objects
    ]
    complete_identity_count = sum(score == 17 for score in completeness_scores)
    completeness = sum(completeness_scores)

    return (
        int(no_invalid_objects),
        int(no_duplicate_or_empty_names),
        complete_identity_count,
        int(direct_count >= required_direct),
        perspective_coverage,
        len(set(classes) - {"unknown"}),
        len(identity_signatures),
        frame_diversity,
        filter_diversity,
        horizon_diversity,
        risk_diversity,
        known_classes,
        direct_count,
        completeness,
    )


def _canonical_relevance_class(raw: dict[str, Any]) -> str:
    declared = str(raw.get("relevance_class", "")).strip().lower()
    normalized = declared.replace("-", "_").replace(" ", "_")
    aliases = {
        "直接领域": "direct_domain",
        "直接领域专家": "direct_domain",
        "直接机制": "direct_mechanism",
        "直接机制专家": "direct_mechanism",
        "相邻领域": "adjacent",
        "受影响者": "affected",
        "执行者": "executor",
        "反证者": "counter",
        "制度视角": "institutional",
    }
    normalized = aliases.get(normalized, normalized)
    known = {
        "direct_domain",
        "direct_mechanism",
        "adjacent",
        "affected",
        "executor",
        "counter",
        "institutional",
    }
    if normalized in known:
        return normalized
    if _is_directly_relevant(raw):
        return "direct_unspecified"
    return "unknown"


def _raw_persona_completeness(raw: dict[str, Any]) -> int:
    scalar_fields = (
        "name",
        "role_summary",
        "topic_relevance",
        "time_horizon",
        "risk_attitude",
        "temperament",
    )
    list_fields = (
        "cognitive_frames",
        "expertise_strong",
        "expertise_weak",
        "analysis_levels",
        "hidden_biases",
    )
    score = sum(1 for key in scalar_fields if str(raw.get(key, "")).strip())
    score += sum(1 for key in list_fields if _strings(raw.get(key)))
    score += int(isinstance(raw.get("values"), dict) and bool(raw.get("values")))
    info_filter = raw.get("information_filter")
    if isinstance(info_filter, dict):
        score += sum(
            1
            for key in (
                "trusted_sources",
                "distrusted_sources",
                "preferred_evidence",
                "ignored_evidence",
                "query_style",
            )
            if _strings(info_filter.get(key))
        )
    return score


def _is_directly_relevant(raw: dict[str, Any]) -> bool:
    """Read the explicit contract first, with a compatibility path for old payloads."""
    declared = str(raw.get("relevance_class", "")).strip().lower()
    normalized = declared.replace("-", "_").replace(" ", "_")
    direct_classes = {
            "direct",
            "direct_domain",
            "direct_mechanism",
            "直接",
            "直接领域",
            "直接机制",
            "直接领域专家",
            "直接机制专家",
    }
    if normalized in direct_classes:
        return True
    known_non_direct = {
        "adjacent",
        "affected",
        "executor",
        "counter",
        "institutional",
        "相邻领域",
        "受影响者",
        "执行者",
        "反证者",
        "制度视角",
    }
    if normalized in known_non_direct:
        return False

    # Invalid or novel labels must not suppress the content-based compatibility
    # contract. Some models copy the slash-delimited schema hint verbatim.
    legacy_text = " ".join(
        [
            str(raw.get("role_summary", "")),
            " ".join(_strings(raw.get("expertise_strong"))),
            str(raw.get("topic_relevance", "")),
        ]
    ).lower()
    return any(
        marker in legacy_text
        for marker in (
            "直接相关",
            "直接研究",
            "核心机制",
            "领域专家",
            "机制专家",
            "direct domain",
            "direct mechanism",
            "directly relevant",
        )
    )


def _unwrap_cohort_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("question_frame"), dict) and isinstance(payload.get("personas"), list):
        return payload
    for key in ("required_output", "output", "result", "data"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            if isinstance(candidate.get("question_frame"), dict) and isinstance(candidate.get("personas"), list):
                return candidate
    return payload


def _normalize_frame(raw: dict[str, Any], question: str, cohort_id: str) -> dict[str, Any]:
    declared_dimensions = _required_strings(
        raw, "relevant_evidence_dimensions", minimum=4
    )
    domains = _required_strings(raw, "relevant_knowledge_domains", minimum=2)
    key_terms = _required_strings(raw, "key_terms", minimum=1)
    disputed_definitions = _strings(raw.get("disputed_definitions"))
    scope = raw.get("scope") if isinstance(raw.get("scope"), dict) else {}
    declared_type = _required_text(raw, "question_type")
    grammar = build_reasoning_grammar(
        question,
        declared_type,
        disputed_definitions=disputed_definitions,
        scope=scope,
    )
    evidence_dimensions = merge_evidence_dimensions(
        declared_dimensions,
        grammar,
    )
    return {
        "context_id": cohort_id,
        "exact_question": question,
        "normalized_question": question.strip(),
        "judgment_target": _required_text(raw, "judgment_target"),
        "question_type": declared_type,
        "question_types": grammar["question_types"],
        "primary_question_type": grammar["primary_question_type"],
        "key_terms": key_terms,
        "disputed_definitions": disputed_definitions,
        "scope": scope,
        "relevant_knowledge_domains": domains,
        "relevant_evidence_dimensions": evidence_dimensions,
        "misleading_substitutions": _strings(raw.get("misleading_substitutions")),
        "reasoning_grammar": grammar,
        "one_pass_coverage": {
            "coverage_targets": evidence_dimensions,
            "automatic_second_round": False,
            "remaining_gaps_become_explicit_unknowns": True,
        },
        "carryover_policy": "禁止继承任何其他问题的事实、案例、结论和专用措辞。",
    }


def _normalize_person(raw: dict[str, Any], question: str, cohort_id: str, index: int) -> DigitalPerson:
    name = _required_text(raw, "name")
    role_summary = _required_text(raw, "role_summary")
    topic_relevance = _required_text(raw, "topic_relevance")
    information_filter = raw.get("information_filter")
    if not isinstance(information_filter, dict):
        raise ValueError(f"persona {index} information_filter is missing")
    values = _normalized_values(raw.get("values"))
    trusted_sources = _required_strings(information_filter, "trusted_sources", minimum=1)
    distrusted_sources = _required_strings(information_filter, "distrusted_sources", minimum=1)
    preferred_evidence = _required_strings(information_filter, "preferred_evidence", minimum=1)
    ignored_evidence = _required_strings(information_filter, "ignored_evidence", minimum=1)
    query_style = _strings(information_filter.get("query_style"))
    if not query_style:
        query_style = [
            f"优先检索{trusted_sources[0]}中的{preferred_evidence[0]}，"
            f"并降低只提供{ignored_evidence[0]}的材料权重"
        ]
    return DigitalPerson(
        id=f"{cohort_id}_p{index:02d}",
        name=f"{name}#{index:02d}",
        role_summary=f"{role_summary} 本题关联：{topic_relevance}",
        cognitive_frames=_required_strings(raw, "cognitive_frames", minimum=1),
        expertise_strong=_required_strings(raw, "expertise_strong", minimum=2),
        expertise_weak=_required_strings(raw, "expertise_weak", minimum=1),
        values=values,
        time_horizon=_required_text(raw, "time_horizon"),
        risk_attitude=_required_text(raw, "risk_attitude"),
        analysis_levels=_required_strings(raw, "analysis_levels", minimum=1),
        hidden_biases=_required_strings(raw, "hidden_biases", minimum=1),
        information_filter=InformationFilter(
            trusted_sources=trusted_sources,
            distrusted_sources=distrusted_sources,
            preferred_evidence=preferred_evidence,
            ignored_evidence=ignored_evidence,
            query_style=query_style,
        ),
        temperament=_required_text(raw, "temperament"),
    )


def _normalized_values(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("persona values are missing")
    parsed: dict[str, float] = {}
    for key, raw in value.items():
        try:
            number = max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
        if number > 0:
            parsed[str(key)] = number
    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("persona values are invalid")
    return {key: round(number / total, 4) for key, number in parsed.items()}


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"required text is missing: {key}")
    return value


def _required_strings(payload: dict[str, Any], key: str, *, minimum: int) -> list[str]:
    values = _strings(payload.get(key))
    if len(values) < minimum:
        raise ValueError(f"{key} needs at least {minimum} values")
    return values


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _raw_identity_signature(raw: dict[str, Any]) -> tuple[Any, ...]:
    information_filter = raw.get("information_filter") or {}
    return (
        tuple(sorted(_strings(raw.get("cognitive_frames")))),
        tuple(sorted(_strings(raw.get("expertise_strong")))),
        tuple(sorted(str(key).strip() for key in (raw.get("values") or {}) if str(key).strip())),
        str(raw.get("time_horizon") or "").strip().casefold(),
        str(raw.get("risk_attitude") or "").strip().casefold(),
        tuple(sorted(_strings(raw.get("analysis_levels")))),
        tuple(sorted(_strings(raw.get("hidden_biases")))),
        tuple(sorted(_strings(information_filter.get("preferred_evidence")))),
        tuple(sorted(_strings(information_filter.get("ignored_evidence")))),
    )


def _cohort_geometry(
    personas: list[DigitalPerson], frame: dict[str, Any]
) -> dict[str, Any]:
    """Measure declared cognitive spread locally; never add people or calls."""

    count = len(personas)
    signatures = {
        (
            tuple(sorted(person.cognitive_frames)),
            tuple(sorted(person.expertise_strong)),
            tuple(sorted(person.values)),
            person.time_horizon,
            person.risk_attitude,
            tuple(sorted(person.analysis_levels)),
            tuple(sorted(person.hidden_biases)),
            tuple(sorted(person.information_filter.preferred_evidence)),
            tuple(sorted(person.information_filter.ignored_evidence)),
        )
        for person in personas
    }
    dimensions = {
        "cognitive_frames": {
            value
            for person in personas
            for value in person.cognitive_frames
        },
        "time_horizons": {person.time_horizon for person in personas},
        "risk_attitudes": {person.risk_attitude for person in personas},
        "analysis_levels": {
            value
            for person in personas
            for value in person.analysis_levels
        },
        "value_priorities": {
            value for person in personas for value in person.values
        },
        "preferred_evidence": {
            value
            for person in personas
            for value in person.information_filter.preferred_evidence
        },
        "ignored_evidence": {
            value
            for person in personas
            for value in person.information_filter.ignored_evidence
        },
    }
    expected_spread = {
        "cognitive_frames": 6,
        "time_horizons": 4,
        "risk_attitudes": 4,
        "analysis_levels": 4,
        "value_priorities": 6,
        "preferred_evidence": 8,
        "ignored_evidence": 8,
    }
    ratios = {
        key: round(
            min(1.0, len(values) / max(1, min(count, expected_spread[key]))),
            4,
        )
        for key, values in dimensions.items()
    }
    diversity_score = (
        round(sum(ratios.values()) / len(ratios), 4) if ratios else 0.0
    )
    duplicate_count = max(0, count - len(signatures))
    return {
        "schema_version": "bme.cohort-geometry.v1",
        "person_count": count,
        "effective_declared_identity_count": len(signatures),
        "exact_duplicate_identity_count": duplicate_count,
        "dimension_cardinality": {
            key: len(values) for key, values in dimensions.items()
        },
        "dimension_spread": ratios,
        "declared_diversity_score": diversity_score,
        "coverage_target_count": len(
            frame.get("relevant_evidence_dimensions", []) or []
        ),
        "status": (
            "passed"
            if duplicate_count == 0 and diversity_score >= 0.7
            else "review"
        ),
        "adds_model_calls": False,
        "automatic_second_round": False,
    }


def _question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:8]


def _question_type(question: str) -> str:
    if any(token in question for token in ["是否", "会不会", "能否", "该不该", "要不要"]):
        return "判断或决策"
    if any(token in question for token in ["为什么", "为何", "原因"]):
        return "解释"
    if any(token in question for token in ["如何", "怎么"]):
        return "方案"
    return "开放问题"
