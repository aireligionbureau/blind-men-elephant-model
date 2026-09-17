from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bme_model.meta_model import diagnostics as legacy
from bme_model.meta_model.lens_registry import load_lenses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the legacy per-person shadow rules with cognitive-chain scanning on one saved run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--historical-run", type=Path, default=None)
    args = parser.parse_args()

    comparison = compare_run(args.run_dir, historical_run=args.historical_run)
    json_path = args.run_dir / "meta_model_upgrade_comparison.json"
    markdown_path = args.run_dir / "meta_model_upgrade_comparison.md"
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    print(json.dumps(comparison["headline_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False, indent=2))


def compare_run(run_dir: Path, *, historical_run: Path | None = None) -> dict[str, Any]:
    retrieval = _read_json(run_dir / "retrieval.json")
    diagnosis = _read_json(run_dir / "diagnosis.json")
    person_results = [_read_json(path) for path in sorted((run_dir / "person_outputs").glob("*.json"))]
    outputs_by_person = {
        item["person_id"]: item.get("output", {})
        for item in person_results
        if item.get("status") == "succeeded"
    }
    ledgers_by_person = {item["person_id"]: item for item in retrieval["evidence_ledgers"]}
    lens_map = load_lenses()

    old_by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for persona in retrieval["personas"]:
        person_id = persona["id"]
        ledger = ledgers_by_person[person_id]
        output = outputs_by_person.get(person_id, {})
        old_by_person[person_id].extend(legacy._diagnose_information_shadows(persona, ledger, lens_map))
        old_by_person[person_id].extend(legacy._diagnose_bias_shadows(persona, ledger, lens_map))
        old_by_person[person_id].extend(legacy._diagnose_argument_shadows(persona, ledger, lens_map))
        old_by_person[person_id].extend(legacy._diagnose_local_whole_overreach(persona, ledger, output))
        old_by_person[person_id].extend(legacy._diagnose_expert_shadows(persona, ledger, lens_map))
        old_by_person[person_id].extend(legacy._diagnose_emotion_shadows(persona, ledger, lens_map))

    scans = diagnosis.get("cognitive_chain_scans", [])
    scans_by_person = {scan["person_id"]: scan for scan in scans}
    old_findings = [item for values in old_by_person.values() for item in values]
    detector_results = [item for scan in scans for item in scan.get("detector_results", [])]
    detected = [item for item in detector_results if item.get("status") == "detected"]
    insufficient = [item for item in detector_results if item.get("status") == "insufficient_evidence"]

    required_detected_fields = (
        "chain_locations",
        "observed_evidence",
        "plain_language_diagnosis",
        "mechanism",
        "consequence_for_conclusion",
        "competing_explanations",
        "falsification_test",
        "preserved_valid_part",
    )
    new_grounded = sum(all(item.get(field) for field in required_detected_fields) for item in detected)
    old_identity_assumptions = [
        item
        for item in old_findings
        if "内置偏差为" in str(item.get("description", ""))
    ]
    old_stage_counts = {
        person_id: len({item.get("source_stage") for item in findings if item.get("source_stage")})
        for person_id, findings in old_by_person.items()
    }
    new_problem_counts = [
        len(scan.get("plain_language_summary", {}).get("key_problems", []))
        for scan in scans
    ]
    user_problems = [
        problem
        for scan in scans
        for problem in scan.get("plain_language_summary", {}).get("key_problems", [])
    ]
    domain_terms = (
        "美股",
        "泡沫",
        "估值",
        "利率",
        "盈利",
        "资本开支",
        "现金流",
        "市场",
        "破灭",
        "回调",
        "AI",
        "科技股",
        "流动性",
    )
    domain_specific_user_problems = sum(
        any(
            term in "".join(
                str(problem.get(field, ""))
                for field in ("problem", "where_it_appeared", "how_it_affected_thinking")
            )
            for term in domain_terms
        )
        for problem in user_problems
    )
    detected_with_specific_excerpt = sum(
        any(
            isinstance(evidence, dict)
            and evidence.get("excerpt")
            and any(term in str(evidence["excerpt"]) for term in domain_terms)
            for evidence in item.get("observed_evidence", [])
        )
        for item in detected
    )

    per_person = []
    for persona in retrieval["personas"]:
        person_id = persona["id"]
        scan = scans_by_person.get(person_id, {})
        statuses = Counter(item.get("status") for item in scan.get("detector_results", []))
        per_person.append(
            {
                "person_id": person_id,
                "person_name": persona["name"],
                "legacy_finding_count": len(old_by_person[person_id]),
                "legacy_stages_appearing_in_findings": old_stage_counts.get(person_id, 0),
                "legacy_findings": [item.get("description", "") for item in old_by_person[person_id]],
                "new_chain_coverage": scan.get("chain_coverage", {}).get("score", 0),
                "new_detector_statuses": dict(statuses),
                "new_key_problems": scan.get("plain_language_summary", {}).get("key_problems", []),
            }
        )

    historical = _historical_metrics(historical_run) if historical_run else None
    people_count = len(retrieval["personas"])
    headline_metrics = {
        "question": retrieval["question"],
        "people": people_count,
        "live_chain_execution_score": diagnosis.get("cognitive_chain_evaluation", {}).get("score"),
        "legacy_reconstructed_findings": len(old_findings),
        "new_detected_findings": len(detected),
        "legacy_explicit_detector_decisions": 0,
        "new_explicit_detector_decisions": len(detector_results),
        "legacy_findings_with_full_proof_packet": 0,
        "new_findings_with_full_proof_packet": new_grounded,
        "legacy_identity_labels_treated_as_findings": len(old_identity_assumptions),
        "new_insufficient_evidence_decisions": len(insufficient),
        "legacy_average_stages_appearing_in_findings": round(
            sum(old_stage_counts.values()) / people_count, 2
        ),
        "new_complete_chain_people": sum(
            scan.get("chain_coverage", {}).get("score") == 100 for scan in scans
        ),
        "legacy_average_findings_produced_per_person": round(len(old_findings) / people_count, 2),
        "new_max_plain_language_issues_per_person": max(new_problem_counts, default=0),
        "new_detected_findings_with_domain_specific_excerpt": detected_with_specific_excerpt,
        "new_user_problem_count": len(user_problems),
        "new_unique_user_problem_templates": len({item.get("problem", "") for item in user_problems}),
        "new_domain_specific_user_problems": domain_specific_user_problems,
    }

    return {
        "comparison_type": "same-input A/B",
        "baseline_definition": (
            "用强化前保留的六组逐人规则重建旧式诊断；同一批数字人、检索账本和 DeepSeek 输出同时进入新旧逻辑。"
        ),
        "caution": (
            "这是诊断能力与可审计性比较，不是真相准确率。检出数量更多不自动代表更强。"
        ),
        "headline_metrics": headline_metrics,
        "historical_baseline": historical,
        "per_person": per_person,
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    metrics = comparison["headline_metrics"]
    lines = [
        "# 元模型强化前后实测对比",
        "",
        f"问题：{metrics['question']}",
        "",
        comparison["baseline_definition"],
        "",
        "| 指标 | 强化前 | 强化后 |",
        "| --- | ---: | ---: |",
        f"| 可核对的‘检测器×人’审计槽位 | {metrics['legacy_explicit_detector_decisions']} | {metrics['new_explicit_detector_decisions']} |",
        f"| 带完整证据包的问题 | {metrics['legacy_findings_with_full_proof_packet']} | {metrics['new_findings_with_full_proof_packet']} |",
        f"| 完整覆盖九段认知链的人数 | 无覆盖记录 | {metrics['new_complete_chain_people']}/{metrics['people']} |",
        f"| 能克制为证据不足的决定 | 无此状态 | {metrics['new_insufficient_evidence_decisions']} |",
        f"| 直接把身份预设当问题 | {metrics['legacy_identity_labels_treated_as_findings']} | 0（必须看到实际表现） |",
        f"| 旧规则产出/新用户层展示 | 平均 {metrics['legacy_average_findings_produced_per_person']} 条 | 每人最多 {metrics['new_max_plain_language_issues_per_person']} 条 |",
        "",
        f"新版本认知链执行评分：{metrics['live_chain_execution_score']}/100。",
        "",
        "注意：检出数量多少不是强弱标准。关键差异是每个检出是否有链条证据、影响机制、竞争解释、推翻条件和可保留部分。",
        "",
        "## 仍未达到满分的地方",
        "",
        f"后台 {metrics['new_detected_findings']} 个检出中，有 {metrics['new_detected_findings_with_domain_specific_excerpt']} 个钉住了本题的具体原话；但用户层 {metrics['new_user_problem_count']} 条摘要只收敛成 {metrics['new_unique_user_problem_templates']} 种问题模板，其中直接带出本题具体市场内容的只有 {metrics['new_domain_specific_user_problems']} 条。后台已经会解剖，前台压缩仍把不少具体性抹平了。",
        "",
        "## 逐人例子",
        "",
    ]
    for person in comparison["per_person"][:3]:
        lines.append(f"### {person['person_name']}")
        lines.append("")
        lines.append("强化前：")
        for finding in person["legacy_findings"][:3]:
            lines.append(f"- {finding}")
        lines.append("")
        lines.append("强化后：")
        for problem in person["new_key_problems"]:
            lines.append(f"- {problem.get('problem', '')}")
            lines.append(f"  出现位置：{problem.get('where_it_appeared', '')}")
            lines.append(f"  对判断的影响：{problem.get('how_it_affected_thinking', '')}")
        lines.append("")
    return "\n".join(lines)


def _historical_metrics(run_dir: Path) -> dict[str, Any]:
    diagnosis = _read_json(run_dir / "diagnosis.json")
    summary = _read_json(run_dir / "summary.json")
    return {
        "run_dir": str(run_dir),
        "question": summary.get("question"),
        "people_succeeded": summary.get("usage", {}).get("succeeded"),
        "shadow_count": len(diagnosis.get("shadow_registry", [])),
        "has_explicit_chain_scans": "cognitive_chain_scans" in diagnosis,
        "meta_modules": list(diagnosis.get("meta_modules", {})),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
