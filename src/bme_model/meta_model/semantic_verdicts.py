from __future__ import annotations

import json
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from ..json_utils import parse_json_content
from ..providers import (
    DeepSeekAPIError,
    DeepSeekClient,
    DeepSeekTimeoutError,
    retry_delay_seconds,
)
from ..runtime import atomic_write_json
from .forensics import DETECTOR_IDS


SEMANTIC_VERDICT_SYSTEM = """你是盲人摸象模型中的“认知链判读层”。

你一次只能阅读一个数字人的完整思考链。十个检测器已经完成候选发现；你不得增加新检测器，也不得研究其他数字人，更不得宣布问题的标准答案。

你的任务是把重复候选合并成一个最能改写结论的根问题，并用大白话准确说明：这个人凭什么具体材料，从哪一句推到了哪一句，中间缺少哪座桥，因此结论的方向、范围、时间或信心应该怎样改变，哪些局部仍然成立。只有确有第二个互不重复、且同样能改写结论的根问题时，才输出第二项；最多两项。

硬规则：
1. 每项判读必须引用输入中真实存在的 evidence_refs，必须包含结论锚点 C1 和至少一个非结论锚点。
2. issue、from_claim 和 to_conclusion 必须使用这个人的具体原话或忠实短述。issue 必须同时点名推理起点里的具体对象和实际结论；只写职业身份或“推理跳跃”“证据不足”“存在偏见”等通用模板不算具体。
3. 多个检测器若指向同一根问题，只写一次；最多输出两个真正独立的根问题，按“修正后最可能改写结论”的程度排序。
4. 必须给出竞争解释和“推翻元模型这条批评”的条件。falsification_test 必须以“若……，这条批评应撤销或降级。”收尾；不得把“证明批评正确”的条件冒充推翻条件。没有足够证据就放进 unresolved，不得硬判。
5. source_detector_ids 只能从输入给出的检测器中选择。检测器名称只用于后台，不得出现在 issue 中。
6. 不得添加输入中不存在的事实、数字、人物、机制或外部知识。每个数字、概率、年份和时间范围都必须出现在本条 evidence_refs 实际指向的锚点中；价值权重引用 I3，自报置信度引用 C2，来源数量引用 S1。不得凭空给出修正后的精确数值。
7. 输入案例只是只读材料，绝对不得在输出中复述。输出 JSON 顶层只能有 person_id、headline、verdicts、unresolved 四个字段；不得输出 input_case、single_person_case、required_output、output_schema 或 final_check。

输出必须是 JSON，不要输出 Markdown。
"""


def build_semantic_verdict_case(
    question: str,
    persona: dict[str, Any],
    ledger: dict[str, Any],
    output: dict[str, Any],
    chain_scan: dict[str, Any],
) -> dict[str, Any]:
    """Build an evidence-addressable case for one person only."""
    anchors: list[dict[str, str]] = []

    def add(anchor_id: str, stage: str, text: Any) -> None:
        cleaned = _compact(text, 380)
        if cleaned:
            anchors.append({"id": anchor_id, "stage": stage, "text": cleaned})

    add(
        "I1",
        "identity",
        "角色：" + str(persona.get("role_summary", ""))
        + "；认知框架：" + "、".join(_strings(persona.get("cognitive_frames"))[:6]),
    )
    add(
        "I2",
        "identity",
        "强项：" + "、".join(_strings(persona.get("expertise_strong"))[:6])
        + "；弱项：" + "、".join(_strings(persona.get("expertise_weak"))[:6]),
    )
    add("I3", "identity", "价值权重：" + json.dumps(persona.get("values", {}), ensure_ascii=False))
    add("I4", "identity", "风险态度：" + str(persona.get("risk_attitude", "")))
    search_strategy = ledger.get("search_strategy", {})
    for index, item in enumerate(_strings(search_strategy.get("preferred_queries"))[:2], start=1):
        add(f"Q{index}", "retrieval", item)
    information_filter = persona.get("information_filter", {})
    add(
        "F1",
        "information_filter",
        "偏好证据："
        + "、".join(_strings(information_filter.get("preferred_evidence"))[:5])
        + "；容易忽略："
        + "、".join(_strings(information_filter.get("ignored_evidence"))[:5]),
    )
    for index, item in enumerate(_strings(output.get("core_assumptions"))[:6], start=1):
        add(f"A{index}", "assumption", item)
    for index, item in enumerate(_strings(output.get("reasoning_path"))[:8], start=1):
        add(f"R{index}", "reasoning", item)
    for index, item in enumerate(_strings(output.get("value_judgements"))[:4], start=1):
        add(f"V{index}", "value_evaluation", item)
    add("C1", "conclusion", output.get("conclusion"))
    if output.get("confidence") is not None:
        add("C2", "conclusion_calibration", f"自报置信度：{output.get('confidence')}")
    for index, item in enumerate(_strings(output.get("what_i_underweighted"))[:4], start=1):
        add(f"U{index}", "self_correction", item)
    for index, item in enumerate(_strings(output.get("what_would_change_my_mind"))[:4], start=1):
        add(f"M{index}", "self_correction", item)

    candidates = []
    evidence_index = 1
    detected_ids: list[str] = []
    detector_statuses = []
    for finding in chain_scan.get("detector_results", []):
        detector_id = str(finding.get("detector_id", ""))
        status = str(finding.get("status", ""))
        detector_statuses.append({"detector_id": detector_id, "status": status})
        if status != "detected":
            continue
        detected_ids.append(detector_id)
        evidence_refs = []
        for observation in finding.get("observed_evidence", [])[:3]:
            if not isinstance(observation, dict):
                continue
            anchor_id = f"E{evidence_index}"
            evidence_index += 1
            fact = str(observation.get("fact", "")).strip()
            excerpt = str(observation.get("excerpt", "")).strip()
            text = fact
            if excerpt:
                text = f"{fact} 原话：{excerpt}" if fact else excerpt
            add(anchor_id, str(observation.get("stage", "unknown")), text)
            evidence_refs.append(anchor_id)
        candidates.append(
            {
                "detector_id": detector_id,
                "candidate_problem": finding.get("plain_language_diagnosis", ""),
                "mechanism": finding.get("mechanism", ""),
                "possible_effect": finding.get("consequence_for_conclusion", ""),
                "competing_explanations": finding.get("competing_explanations", [])[:2],
                "questions_to_distinguish": finding.get("questions_to_distinguish", [])[:2],
                "falsification_test": finding.get("falsification_test", ""),
                "preserved_valid_part": finding.get("preserved_valid_part", ""),
                "diagnostic_confidence": finding.get("diagnostic_confidence", 0),
                "evidence_refs": evidence_refs,
            }
        )

    decisions = ledger.get("source_decisions", [])
    verified_external = sum(
        1
        for item in ledger.get("entries", [])
        if item.get("retrieval_layer") not in {None, "model_prior", "mock_search"}
        and item.get("verification_status") not in {None, "unverified_model_prior", "synthetic_fixture"}
    )
    add(
        "S1",
        "source_context",
        f"候选来源 {len(decisions)} 条；采信 {len(ledger.get('accepted_sources', []))} 条；"
        f"拒绝 {len(ledger.get('rejected_sources', []))} 条；忽略 {len(ledger.get('ignored_sources', []))} 条；"
        f"外部核验 {verified_external} 条。",
    )
    return {
        "question": question,
        "person_id": persona.get("id", chain_scan.get("person_id", "")),
        "person_name": persona.get("name", chain_scan.get("person_name", "")),
        "scope_rule": "只判读这个人的认知链；不得引用其他数字人的材料。",
        "identity_context": {
            "role_summary": persona.get("role_summary", ""),
            "cognitive_frames": persona.get("cognitive_frames", []),
            "expertise_strong": persona.get("expertise_strong", []),
            "expertise_weak": persona.get("expertise_weak", []),
            "values": persona.get("values", {}),
            "risk_attitude": persona.get("risk_attitude", ""),
        },
        "source_context": {
            "candidate_count": len(decisions),
            "accepted_count": len(ledger.get("accepted_sources", [])),
            "rejected_count": len(ledger.get("rejected_sources", [])),
            "ignored_count": len(ledger.get("ignored_sources", [])),
            "verified_external_count": verified_external,
        },
        "chain_coverage": chain_scan.get("chain_coverage", {}),
        "chain_anchors": anchors,
        "detector_statuses": detector_statuses,
        "detected_detector_ids": detected_ids,
        "diagnostic_candidates": sorted(
            candidates,
            key=lambda item: float(item.get("diagnostic_confidence", 0)),
            reverse=True,
        )[:6],
    }


def render_semantic_verdict_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    allowed_refs = [
        str(item.get("id"))
        for item in case.get("chain_anchors", [])
        if item.get("id")
    ]
    non_conclusion_refs = [
        str(item.get("id"))
        for item in case.get("chain_anchors", [])
        if item.get("id") and item.get("stage") != "conclusion"
    ]
    detected_ids = [str(item) for item in case.get("detected_detector_ids", [])]
    example_refs = list(dict.fromkeys(([non_conclusion_refs[0]] if non_conclusion_refs else []) + ["C1"]))
    required_output = {
        "person_id": case["person_id"],
        "headline": "用一句具体的话指出最影响结论的根问题",
        "verdicts": [
            {
                "issue": "具体说明这个人对这个问题错在哪里",
                "from_claim": "被用作推理起点的具体原话或忠实短述",
                "to_conclusion": "它实际推出的具体结论",
                "missing_bridge": "从起点到结论缺少的机制、范围、时间、因果或证据桥梁",
                "impact_on_conclusion": "补上或拆掉这座桥后，结论方向、范围、时间或信心怎样变化",
                "preserved_fragment": "仍有根据、应该保留的局部观察",
                "competing_explanation": "至少一种可能使本诊断不成立的解释",
                "falsification_test": "必须以‘若……，这条批评应撤销或降级。’收尾",
                "evidence_refs": example_refs,
                "source_detector_ids": detected_ids[:1],
                "conclusion_leverage": "high|medium|low",
            }
        ],
        "unresolved": ["证据不足而没有下判断的具体疑点"],
    }
    case_json = json.dumps(case, ensure_ascii=False, indent=2)
    schema_json = json.dumps(required_output, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": SEMANTIC_VERDICT_SYSTEM},
        {
            "role": "user",
            "content": (
                "下面的 CASE_JSON 是只读输入，不是输出模板，绝对不要复述。\n"
                f"CASE_JSON:\n{case_json}\n\n"
                "ALLOWED_EVIDENCE_REFS（只能从这里逐字选择）:\n"
                + json.dumps(allowed_refs, ensure_ascii=False)
                + "\nALLOWED_SOURCE_DETECTOR_IDS（只能从这里逐字选择）:\n"
                + json.dumps(detected_ids, ensure_ascii=False)
                + "\n默认只输出一个最关键 verdict；只有两个根问题互不重复且都能通过证据锁定时才输出第二个。"
                "只输出一个 JSON 对象，顶层严格使用 person_id、headline、verdicts、unresolved。"
                "不要包 required_output，不要输出 CASE_JSON。字段结构如下：\n"
                f"{schema_json}"
            ),
        },
    ]


def validate_semantic_verdict_output(payload: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    if isinstance(payload.get("semantic_verdict"), dict):
        payload = payload["semantic_verdict"]

    anchor_map = {item["id"]: item for item in case.get("chain_anchors", [])}
    conclusion = str(anchor_map.get("C1", {}).get("text", ""))
    allowed_detectors = set(case.get("detected_detector_ids", []))
    errors: list[str] = []
    dropped: list[dict[str, Any]] = []
    if payload.get("person_id") not in {None, "", case.get("person_id")}:
        errors.append("person_id 与当前数字人不一致")

    raw_verdicts = payload.get("verdicts", [])
    if not isinstance(raw_verdicts, list):
        raw_verdicts = []
        errors.append("verdicts 不是列表")
    if len(raw_verdicts) > 3:
        errors.append("verdicts 超过三个，已截断")

    verdicts: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_verdicts[:3], start=1):
        if not isinstance(raw, dict):
            dropped.append({"index": index, "errors": ["不是对象"]})
            continue
        verdict, verdict_errors = _validate_one_verdict(raw, case, anchor_map, conclusion, allowed_detectors)
        if verdict_errors:
            dropped.append({"index": index, "errors": verdict_errors, "raw": raw})
            continue
        if _duplicates_existing(verdict, verdicts):
            dropped.append({"index": index, "errors": ["与更高优先级问题重复"], "raw": raw})
            continue
        if len(verdicts) >= 2:
            dropped.append({"index": index, "errors": ["优先级低于前两个根问题"], "raw": raw})
            continue
        verdicts.append(verdict)

    unresolved = _strings(payload.get("unresolved"))[:6]
    if raw_verdicts and not verdicts:
        errors.append("没有任何 verdict 通过证据锁定校验")
    if not raw_verdicts and not unresolved:
        errors.append("既没有具体问题，也没有说明为何证据不足")
    substantive_drops = [
        item
        for item in dropped
        if any("重复" not in reason and "优先级低于" not in reason for reason in item.get("errors", []))
    ]
    if substantive_drops and verdicts:
        errors.append("存在未通过证据、具体性或证伪校验的判读，不能用合格项遮盖")

    summary = _summary_from_verdicts(verdicts, unresolved)
    valid = not errors and (bool(verdicts) or bool(unresolved))
    return {
        "person_id": case.get("person_id", ""),
        "person_name": case.get("person_name", ""),
        "headline": summary["headline"],
        "verdicts": verdicts,
        "unresolved": unresolved,
        "summary": summary,
        "validation": {
            "valid": valid,
            "errors": errors,
            "dropped_verdicts": dropped,
            "accepted_verdict_count": len(verdicts),
            "all_evidence_refs_valid": all(item.get("evidence_validation", {}).get("valid") for item in verdicts),
            "minimum_specificity": min(
                (item.get("evidence_validation", {}).get("specificity_score", 0) for item in verdicts),
                default=1.0 if unresolved else 0.0,
            ),
        },
    }


def build_fallback_semantic_verdict(case: dict[str, Any]) -> dict[str, Any]:
    anchor_map = {item["id"]: item for item in case.get("chain_anchors", [])}
    conclusion = str(anchor_map.get("C1", {}).get("text", ""))
    raw_verdicts = []
    used_groups: set[str] = set()
    candidates = sorted(
        case.get("diagnostic_candidates", []),
        key=lambda item: float(item.get("diagnostic_confidence", 0)),
        reverse=True,
    )
    for candidate in candidates:
        group = _root_group(str(candidate.get("detector_id", "")))
        if group in used_groups:
            continue
        refs = [ref for ref in candidate.get("evidence_refs", []) if ref in anchor_map]
        source_ref = next(
            (
                ref
                for ref in refs
                if ref != "C1" and anchor_map.get(ref, {}).get("stage") != "conclusion"
            ),
            None,
        )
        if source_ref is None or not conclusion:
            continue
        source_text = anchor_map[source_ref]["text"]
        from_claim = _evidence_excerpt(source_text)
        questions = _strings(candidate.get("questions_to_distinguish"))
        missing = questions[0] if questions else str(candidate.get("mechanism", ""))
        evidence_refs = list(dict.fromkeys([source_ref, "C1"] + refs[:2]))
        raw_verdicts.append(
            {
                "issue": (
                    f"它根据“{_compact(from_claim, 82)}”推出“{_compact(conclusion, 86)}”，"
                    f"但现有链条没有说明{_compact(missing, 92)}。"
                ),
                "from_claim": from_claim,
                "to_conclusion": conclusion,
                "missing_bridge": missing,
                "impact_on_conclusion": candidate.get("possible_effect", "相关结论需要降权。"),
                "preserved_fragment": candidate.get("preserved_valid_part", "保留被材料直接支持的局部观察。"),
                "competing_explanation": (_strings(candidate.get("competing_explanations")) or ["现有省略可能只是表达压缩。"])[0],
                "falsification_test": "若补充材料能直接证明这一步推断成立并排除竞争解释，这条批评应撤销或降级。",
                "evidence_refs": evidence_refs,
                "source_detector_ids": [candidate.get("detector_id")],
                "conclusion_leverage": "high" if not raw_verdicts else "medium",
            }
        )
        used_groups.add(group)
        if len(raw_verdicts) == 3:
            break
    unresolved = [] if raw_verdicts else ["现有链条没有足够材料形成具体且可证伪的判读。"]
    validated = validate_semantic_verdict_output(
        {"person_id": case.get("person_id"), "verdicts": raw_verdicts, "unresolved": unresolved},
        case,
    )
    validated["source"] = "deterministic_fallback"
    return validated


def run_semantic_verdict_batch(
    question: str,
    personas: list[dict[str, Any]],
    evidence_ledgers: list[dict[str, Any]],
    person_outputs: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    *,
    output_dir: str | Path,
    model: str,
    max_tokens: int = 8192,
    concurrency: int = 6,
    retries: int = 2,
    resume: bool = True,
    client_factory: Callable[[], DeepSeekClient] | None = None,
) -> list[dict[str, Any]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    persona_map = {item["id"]: item for item in personas}
    ledger_map = {item["person_id"]: item for item in evidence_ledgers}
    scan_map = {item["person_id"]: item for item in diagnosis.get("cognitive_chain_scans", [])}
    output_map = {
        item["person_id"]: item.get("output", item)
        for item in person_outputs
        if item.get("person_id")
    }
    cases: dict[str, dict[str, Any]] = {}
    for person_id, output in output_map.items():
        if person_id not in persona_map or person_id not in scan_map:
            continue
        cases[person_id] = build_semantic_verdict_case(
            question,
            persona_map[person_id],
            ledger_map.get(person_id, {}),
            output,
            scan_map[person_id],
        )
    existing = {
        item["person_id"]: item
        for item in load_semantic_verdict_records(output_path)
    } if resume else {}
    results: dict[str, dict[str, Any]] = {}
    for person_id, item in existing.items():
        current_case = cases.get(person_id)
        if not current_case:
            continue
        saved_fingerprint = str(item.get("input_fingerprint") or "")
        case_matches = saved_fingerprint == _case_fingerprint(current_case)
        if not saved_fingerprint and isinstance(item.get("case"), dict):
            case_matches = _case_fingerprint(item["case"]) == _case_fingerprint(current_case)
        if (
            case_matches
            and item.get("status") in {"succeeded", "no_finding"}
            and item.get("verdict", {}).get("validation", {}).get("valid")
        ):
            results[person_id] = item
    tasks = [
        case
        for person_id, case in cases.items()
        if person_id not in results
    ]

    factory = client_factory or (lambda: DeepSeekClient(model=model))
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = {
            executor.submit(
                _run_one_semantic_verdict,
                case,
                factory,
                model,
                max_tokens,
                retries,
            ): case["person_id"]
            for case in tasks
        }
        for future in as_completed(futures):
            person_id = futures[future]
            record = future.result()
            record["input_fingerprint"] = _case_fingerprint(cases[person_id])
            previous = existing.get(person_id) or {}
            if previous and previous.get("status") not in {"succeeded", "no_finding"}:
                record["recovery_history"] = [
                    *(previous.get("recovery_history") or []),
                    {
                        "status": previous.get("status"),
                        "attempts": previous.get("attempts", 0),
                        "errors": previous.get("errors", []),
                        "usage": previous.get("usage", {}),
                    },
                ][-10:]
                record["attempts"] = int(previous.get("attempts") or 0) + int(
                    record.get("attempts") or 0
                )
                record["usage"] = _merge_usage(
                    dict(previous.get("usage") or {}),
                    record.get("usage") or {},
                )
            results[person_id] = record
            _write_json(output_path / f"{person_id}.json", record)
    return [results[key] for key in sorted(results)]


def semantic_case_fingerprint(
    case: dict[str, Any],
    *,
    execution_mode: str = "provider_default",
) -> str:
    """Stable identity for one evidence-locked semantic case."""

    if execution_mode == "provider_default":
        return _case_fingerprint(case)
    payload = {
        "case": case,
        "execution_mode": execution_mode,
    }
    return _case_fingerprint(payload)


def run_semantic_verdict_case(
    case: dict[str, Any],
    *,
    model: str,
    max_tokens: int = 8192,
    retries: int = 2,
    client_factory: Callable[[], DeepSeekClient] | None = None,
    inference_mode: str = "provider_default",
) -> dict[str, Any]:
    """Run the same judge used by the batch path for a single ready person."""

    factory = client_factory or (lambda: DeepSeekClient(model=model))
    return _run_one_semantic_verdict(
        case,
        factory,
        model,
        max_tokens,
        retries,
        inference_mode,
    )


def load_semantic_verdict_records(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir)
    if not path.exists():
        return []
    records = []
    for file_path in sorted(path.glob("*.json")):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("person_id"):
            case = payload.get("case")
            verdict = payload.get("verdict")
            if (
                payload.get("status") == "fallback"
                and isinstance(case, dict)
                and not case.get("diagnostic_candidates")
            ):
                payload["status"] = "no_finding"
                payload["verdict"] = build_fallback_semantic_verdict(case)
                payload["verdict"]["source"] = "deterministic_no_finding"
                payload["reclassified_from_fallback"] = True
                payload["errors"] = []
                _write_json(file_path, payload)
                verdict = payload["verdict"]
            if isinstance(case, dict) and isinstance(verdict, dict):
                normalized = validate_semantic_verdict_output(verdict, case)
                normalized["source"] = verdict.get("source", "llm_semantic_judge")
                payload["verdict"] = normalized
            if payload.get("status") == "fallback" and isinstance(case, dict):
                message = ((payload.get("raw", {}).get("choices") or [{}])[0].get("message") or {})
                content = message.get("content")
                if content:
                    try:
                        recovered = validate_semantic_verdict_output(parse_json_content(content), case)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        recovered = {}
                    if recovered.get("validation", {}).get("valid"):
                        recovered["source"] = "llm_semantic_judge"
                        payload["verdict"] = recovered
                        payload["previous_validation_errors"] = payload.get("errors", [])
                        payload["status"] = "succeeded"
                        payload["recovered_after_validator_upgrade"] = True
            records.append(payload)
    return records


def attach_semantic_verdicts(diagnosis: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_person = {item.get("person_id"): item for item in records}
    attached = 0
    live = 0
    no_finding = 0
    fallback = 0
    for scan in diagnosis.get("cognitive_chain_scans", []):
        record = by_person.get(scan.get("person_id"))
        if not record:
            continue
        verdict = record.get("verdict", {})
        scan["semantic_verdict"] = {
            **verdict,
            "execution_status": record.get("status"),
            "model": record.get("model"),
            "attempts": record.get("attempts"),
        }
        attached += 1
        if record.get("status") == "succeeded":
            live += 1
        elif record.get("status") == "no_finding":
            no_finding += 1
        else:
            fallback += 1
    diagnosis["semantic_verdict_audit"] = {
        "scope": "逐人判读；不使用其他数字人的输出。",
        "record_count": len(records),
        "attached_person_count": attached,
        "live_person_count": live,
        "no_finding_person_count": no_finding,
        "fallback_person_count": fallback,
        "rule": "用户结论必须引用合法锚点、合并同根问题，并明确指出从哪句话跨到了哪句话。",
    }
    return diagnosis


def summarize_semantic_verdict_usage(records: list[dict[str, Any]], *, model: str) -> dict[str, Any]:
    prompt_tokens = sum(int(item.get("usage", {}).get("prompt_tokens") or 0) for item in records)
    completion_tokens = sum(int(item.get("usage", {}).get("completion_tokens") or 0) for item in records)
    total_tokens = sum(int(item.get("usage", {}).get("total_tokens") or 0) for item in records)
    if not total_tokens:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "model": model,
        "succeeded": sum(item.get("status") == "succeeded" for item in records),
        "no_finding": sum(item.get("status") == "no_finding" for item in records),
        "fallback": sum(item.get("status") == "fallback" for item in records),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _run_one_semantic_verdict(
    case: dict[str, Any],
    client_factory: Callable[[], DeepSeekClient],
    model: str,
    max_tokens: int,
    retries: int,
    inference_mode: str = "provider_default",
) -> dict[str, Any]:
    if inference_mode not in {"provider_default", "validated_fast_v1"}:
        raise ValueError(
            "inference_mode must be 'provider_default' or 'validated_fast_v1'"
        )
    started = time.perf_counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not case.get("diagnostic_candidates"):
        verdict = build_fallback_semantic_verdict(case)
        verdict["source"] = "deterministic_no_finding"
        return {
            "person_id": case["person_id"],
            "person_name": case["person_name"],
            "status": "no_finding",
            "model": model,
            "attempts": 0,
            "inference_mode": inference_mode,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "case": case,
            "verdict": verdict,
            "usage": usage,
            "raw": {},
        }
    attempts = 0
    retryable = True
    last_errors: list[str] = []
    last_raw: dict[str, Any] = {}
    attempt_log: list[dict[str, Any]] = []
    messages = render_semantic_verdict_messages(case)
    max_attempts = max(
        retries + 1,
        2 if inference_mode == "validated_fast_v1" else 1,
    )
    for attempt in range(max_attempts):
        attempts = attempt + 1
        thinking_mode = (
            "disabled"
            if inference_mode == "validated_fast_v1" and attempt == 0
            else "enabled"
            if inference_mode == "validated_fast_v1"
            else "provider_default"
        )
        try:
            chat_kwargs: dict[str, Any] = {
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            if thinking_mode != "provider_default":
                chat_kwargs["thinking"] = {"type": thinking_mode}
                if thinking_mode == "enabled":
                    chat_kwargs["reasoning_effort"] = "high"
            response = client_factory().chat(messages, **chat_kwargs)
            usage = _merge_usage(usage, response.usage)
            last_raw = response.raw
            parsed = parse_json_content(response.content)
            validated = validate_semantic_verdict_output(parsed, case)
            if validated.get("validation", {}).get("valid"):
                attempt_log.append(
                    {
                        "attempt": attempts,
                        "status": "succeeded",
                        "thinking_mode": thinking_mode,
                        "accepted_verdict_count": validated.get("validation", {}).get(
                            "accepted_verdict_count", 0
                        ),
                    }
                )
                validated["source"] = "llm_semantic_judge"
                return {
                    "person_id": case["person_id"],
                    "person_name": case["person_name"],
                    "status": "succeeded",
                    "model": model,
                    "attempts": attempts,
                    "inference_mode": inference_mode,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "case": case,
                    "verdict": validated,
                    "usage": usage,
                    "raw": response.raw,
                    "attempt_log": attempt_log,
                }
            last_errors = validated.get("validation", {}).get("errors", [])
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "invalid",
                    "thinking_mode": thinking_mode,
                    "errors": list(last_errors),
                    "dropped_verdicts": validated.get("validation", {}).get(
                        "dropped_verdicts", []
                    )[:2],
                }
            )
            messages = render_semantic_verdict_messages(case) + [
                {
                    "role": "user",
                    "content": "上一次输出没有通过证据锁定校验。请完全重写，修复这些问题："
                    + "；".join(last_errors + [str(item) for item in validated.get("validation", {}).get("dropped_verdicts", [])[:2]]),
                }
            ]
        except DeepSeekTimeoutError as exc:
            last_errors = [str(exc)]
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "timeout",
                    "thinking_mode": thinking_mode,
                    "errors": last_errors,
                }
            )
            retryable = False
            break
        except DeepSeekAPIError as exc:
            last_errors = [str(exc)]
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "provider_error",
                    "thinking_mode": thinking_mode,
                    "errors": last_errors,
                    "retryable": exc.retryable,
                }
            )
            if not exc.retryable:
                retryable = False
                break
            if attempt < retries:
                time.sleep(retry_delay_seconds(exc, attempt))
                continue
            break
        except Exception as exc:  # noqa: BLE001 - preserve a usable fallback.
            last_errors = [str(exc)]
            attempt_log.append(
                {
                    "attempt": attempts,
                    "status": "failed",
                    "thinking_mode": thinking_mode,
                    "errors": last_errors,
                }
            )
        if attempt + 1 < max_attempts:
            time.sleep(min(2**attempt, 8))

    fallback = build_fallback_semantic_verdict(case)
    return {
        "person_id": case["person_id"],
        "person_name": case["person_name"],
        "status": "fallback",
        "model": model,
        "attempts": attempts,
        "inference_mode": inference_mode,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "errors": last_errors,
        "retryable": retryable,
        "case": case,
        "verdict": fallback,
        "usage": usage,
        "raw": last_raw,
        "attempt_log": attempt_log,
    }


def _validate_one_verdict(
    raw: dict[str, Any],
    case: dict[str, Any],
    anchor_map: dict[str, dict[str, str]],
    conclusion: str,
    allowed_detectors: set[str],
) -> tuple[dict[str, Any], list[str]]:
    fields = (
        "issue",
        "from_claim",
        "to_conclusion",
        "missing_bridge",
        "impact_on_conclusion",
        "preserved_fragment",
        "competing_explanation",
        "falsification_test",
    )
    verdict = {field: _compact(raw.get(field), 520) for field in fields}
    errors = [f"缺少 {field}" for field in fields if not verdict[field]]
    if verdict["falsification_test"] and not _reverses_diagnosis(verdict["falsification_test"]):
        errors.append("falsification_test 没有说明何时应撤销或降级元模型这条批评")
    refs = list(dict.fromkeys(_strings(raw.get("evidence_refs"))))
    invalid_refs = [ref for ref in refs if ref not in anchor_map]
    if invalid_refs:
        errors.append("引用了不存在的证据编号：" + ",".join(invalid_refs))
    if conclusion and "C1" not in refs:
        errors.append("没有引用结论锚点 C1")
    if not any(
        ref != "C1"
        and ref in anchor_map
        and anchor_map[ref].get("stage") != "conclusion"
        for ref in refs
    ):
        errors.append("没有引用非结论锚点")
    cited_text = " ".join(anchor_map[ref]["text"] for ref in refs if ref in anchor_map)
    unsupported_numbers = sorted(_numeric_facts(" ".join(verdict.values())) - _numeric_facts(cited_text))
    if unsupported_numbers:
        errors.append("加入了所引证据中不存在的数字事实：" + "、".join(unsupported_numbers))

    source_detector_ids = list(dict.fromkeys(_strings(raw.get("source_detector_ids"))))
    invalid_detectors = [item for item in source_detector_ids if item not in allowed_detectors]
    if invalid_detectors:
        errors.append("引用了未检出的检测器：" + ",".join(invalid_detectors))
    if not source_detector_ids:
        errors.append("没有标记候选问题来源")

    source_text = " ".join(
        anchor_map[ref]["text"]
        for ref in refs
        if ref in anchor_map and ref != "C1" and anchor_map[ref].get("stage") != "conclusion"
    )
    from_overlap = _anchor_overlap(verdict["from_claim"] + " " + verdict["issue"], source_text)
    conclusion_overlap = _anchor_overlap(verdict["to_conclusion"] + " " + verdict["issue"], conclusion)
    issue_overlap = _anchor_overlap(verdict["issue"], source_text + " " + conclusion)
    if source_text and from_overlap < 1:
        errors.append("from_claim 没有钉住所引用的链条原话")
    if conclusion and conclusion_overlap < 1:
        errors.append("to_conclusion 没有钉住实际结论")
    if issue_overlap < 1:
        errors.append("issue 仍是可套用到任何问题的通用模板")
    if any(token in verdict["issue"] for token in DETECTOR_IDS):
        errors.append("issue 泄露了内部检测器名称")

    leverage = str(raw.get("conclusion_leverage", "")).lower()
    if leverage not in {"high", "medium", "low"}:
        errors.append("conclusion_leverage 必须是 high、medium 或 low")
        leverage = "medium"
    specificity = round(
        min(1.0, from_overlap / 8) * 0.35
        + min(1.0, conclusion_overlap / 8) * 0.35
        + min(1.0, issue_overlap / 6) * 0.30,
        3,
    )
    if specificity < 0.9:
        errors.append("issue 没有同时钉住具体推理起点和实际结论")
    verdict["falsification_test"] = _canonical_falsification_test(verdict)
    verdict.update(
        {
            "evidence_refs": [ref for ref in refs if ref in anchor_map],
            "source_detector_ids": source_detector_ids,
            "conclusion_leverage": leverage,
            "evidence_validation": {
                "valid": not errors,
                "invalid_refs": invalid_refs,
                "from_claim_overlap": from_overlap,
                "to_conclusion_overlap": conclusion_overlap,
                "issue_overlap": issue_overlap,
                "specificity_score": specificity,
                "question": case.get("question", ""),
            },
        }
    )
    return verdict, errors


def _summary_from_verdicts(verdicts: list[dict[str, Any]], unresolved: list[str]) -> dict[str, Any]:
    key_problems = [
        {
            "problem": item["issue"],
            "where_it_appeared": (
                f"它从“{_compact(item['from_claim'], 125)}”推到“{_compact(item['to_conclusion'], 125)}”；"
                f"中间缺少：{_compact(item['missing_bridge'], 145)}"
            ),
            "how_it_affected_thinking": item["impact_on_conclusion"],
            "what_can_still_be_kept": item["preserved_fragment"],
            "competing_explanation": item["competing_explanation"],
            "falsification_test": item["falsification_test"],
            "evidence_refs": item["evidence_refs"],
            "conclusion_leverage": item["conclusion_leverage"],
        }
        for item in verdicts
    ]
    if verdicts:
        headline = verdicts[0]["issue"]
    elif unresolved:
        headline = "现有材料还不足以指出一个具体且能改变结论的思考问题。"
    else:
        headline = "现有材料没有形成可审计的具体判读。"
    return {
        "headline": _compact(headline, 260),
        "key_problems": key_problems,
        "unresolved_checks": unresolved,
        "display_rule": "每个问题必须属于这个数字人、这条推理和这个问题，换个人不能原样套用。",
    }


def _duplicates_existing(candidate: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    candidate_refs = set(candidate.get("evidence_refs", [])) - {"C1"}
    candidate_chain_refs = {ref for ref in candidate_refs if not ref.startswith("E")}
    candidate_tokens = _ngrams(candidate.get("issue", ""))
    candidate_from_tokens = _ngrams(candidate.get("from_claim", ""))
    candidate_sources = _strings(candidate.get("source_detector_ids"))
    candidate_root = _root_group(candidate_sources[0]) if candidate_sources else ""
    for item in existing:
        item_sources = _strings(item.get("source_detector_ids"))
        item_root = _root_group(item_sources[0]) if item_sources else ""
        if candidate_root and candidate_root == item_root:
            return True
        refs = set(item.get("evidence_refs", [])) - {"C1"}
        chain_refs = {ref for ref in refs if not ref.startswith("E")}
        ref_union = candidate_refs | refs
        ref_overlap = len(candidate_refs & refs) / len(ref_union) if ref_union else 0.0
        tokens = _ngrams(item.get("issue", ""))
        token_union = candidate_tokens | tokens
        token_overlap = len(candidate_tokens & tokens) / len(token_union) if token_union else 0.0
        from_tokens = _ngrams(item.get("from_claim", ""))
        from_containment = (
            len(candidate_from_tokens & from_tokens) / min(len(candidate_from_tokens), len(from_tokens))
            if candidate_from_tokens and from_tokens
            else 0.0
        )
        if candidate_chain_refs & chain_refs and from_containment >= 0.25:
            return True
        if ref_overlap >= 0.67 or token_overlap >= 0.72:
            return True
    return False


def _root_group(detector_id: str) -> str:
    return {
        "bias_heuristics": "information_weighting",
        "social_influence": "information_weighting",
        "argument_forensics": "reasoning_integrity",
        "interpreter": "reasoning_integrity",
        "local_whole_overreach": "reasoning_integrity",
        "expert_failure": "frame_boundary",
        "paradigm": "frame_boundary",
        "ecological_rationality": "frame_boundary",
        "emotion_reason": "human_value_weighting",
        "noise": "judgment_stability",
    }.get(detector_id, detector_id)


def _anchor_overlap(left: str, right: str) -> int:
    if not left or not right:
        return 0
    return len(_ngrams(left) & _ngrams(right))


def _ngrams(value: Any) -> set[str]:
    text = str(value or "")
    chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9.%+-]{2,}", text)
    grams: set[str] = set()
    stop = {"这个", "那个", "因为", "所以", "认为", "结论", "问题", "可能", "没有", "需要", "材料", "推理"}
    for chunk in chunks:
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
            for size in (2, 3, 4):
                for index in range(max(0, len(chunk) - size + 1)):
                    gram = chunk[index : index + size]
                    if gram not in stop:
                        grams.add(gram)
        else:
            grams.add(chunk.lower())
    return grams


def _evidence_excerpt(text: str) -> str:
    if "原话：" in text:
        return text.split("原话：", 1)[1].strip()
    return text


def _reverses_diagnosis(value: Any) -> bool:
    text = str(value or "")
    return bool(
        re.search(
            r"(?:本|这|该)?(?:条)?(?:批评|诊断|判读).{0,8}(?:应|可)?(?:撤销|降级|不成立|收回)",
            text,
        )
    )


def _canonical_falsification_test(verdict: dict[str, Any]) -> str:
    bridge = _compact(verdict.get("missing_bridge"), 180)
    competitor = _compact(verdict.get("competing_explanation"), 120)
    conclusion = _compact(verdict.get("to_conclusion"), 120)
    return (
        f"若有可核验证据能够解决以下缺口：“{bridge}”，并在排除竞争解释“{competitor}”后"
        f"仍支持原结论“{conclusion}”，这条批评应撤销或降级。"
    )


def _numeric_facts(value: Any) -> set[str]:
    text = str(value or "")
    text = re.sub(r"\b[QFARVUMEC]\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:^|[；;，,\s])\d{1,2}(?:[)、:：]|\.(?!\d))",
        " ",
        text,
    )
    pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:"
        r"(?:18|19|20)\d{2}年?"
        r"|\d+(?:\.\d+)?\s*(?:[-‑–—~至]\s*\d+(?:\.\d+)?)?\s*(?:%|年|个月|月|周|天|倍|条|次)"
        r"|0\.\d+"
        r")(?![A-Za-z0-9])"
    )
    facts = set()
    for match in pattern.finditer(text):
        fact = re.sub(r"\s+", "", match.group(0))
        fact = re.sub(r"[-‑–—~至]", "-", fact)
        fact = re.sub(r"\d+\.\d+", lambda item: format(float(item.group(0)), "g"), fact)
        fact = re.sub(r"^((?:18|19|20)\d{2})年$", r"\1", fact)
        facts.add(fact)
    return facts


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _merge_usage(base: dict[str, int], incoming: dict[str, Any]) -> dict[str, int]:
    prompt = int(incoming.get("prompt_tokens") or incoming.get("input_tokens") or 0)
    completion = int(incoming.get("completion_tokens") or incoming.get("output_tokens") or 0)
    total = int(incoming.get("total_tokens") or prompt + completion)
    return {
        "prompt_tokens": int(base.get("prompt_tokens", 0)) + prompt,
        "completion_tokens": int(base.get("completion_tokens", 0)) + completion,
        "total_tokens": int(base.get("total_tokens", 0)) + total,
    }


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def _case_fingerprint(case: dict[str, Any]) -> str:
    encoded = json.dumps(
        case,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
