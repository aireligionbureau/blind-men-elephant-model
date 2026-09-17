import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from bme_model import build_shadow_puzzle, build_profile_plan, build_run_plan, run_filtered_retrieval, run_shadow_diagnosis
from bme_model.agentic_search import _planner_messages
from bme_model.batch import _make_run_dir, summarize_usage, validate_digital_person_output
from bme_model.cohort import (
    _cohort_request,
    _is_directly_relevant,
    _normalize_generated_payload,
    _select_persona_payloads,
    _unwrap_cohort_payload,
    build_fixture_cohort,
    build_fixture_question_frame,
    generate_live_cohort,
)
from bme_model.context import ContextIsolationError, validate_run_isolation
from bme_model.evidence_normalization import normalize_output_evidence_entries
from bme_model.json_utils import parse_json_content
from bme_model.pipeline import PIPELINE_STAGES
from bme_model.meta_model import (
    load_calibration_cases,
    load_capture_protocols,
    load_lens_conflicts,
    load_lenses,
    load_micro_lenses,
    load_shadow_mechanisms,
)
from bme_model.meta_model.diagnostics import _classify_stance
from bme_model.prompts import render_meta_model_messages
from bme_model.prompts import (
    build_digital_person_context_packet,
    validate_digital_person_context_packet,
)
from bme_model.providers.deepseek import LLMResponse
from bme_model.report import _classic_knowledge_story, build_run_report
from bme_model.search import MockSearchProvider, _parse_generic_json_results, choose_regional_provider


def test_build_run_plan_contains_core_sections():
    plan = build_run_plan("是否应该加快部署自动驾驶卡车？")

    assert plan["question"] == "是否应该加快部署自动驾驶卡车？"
    assert plan["status"] == "protocol_plan_only"
    assert plan["personas"]
    assert plan["search_plans"]
    assert plan["evidence_ledgers"]
    assert "classic_lenses" in plan


def test_digital_person_output_validation_rejects_truncated_inner_object():
    output = {
        "preferred_queries": ["AI consciousness"],
        "trusted_sources": ["peer reviewed"],
    }

    errors = validate_digital_person_output("AI 会不会产生意识", "person-01", output)

    assert "person_id is missing or mismatched" in errors
    assert "core_assumptions is missing or empty" in errors
    assert "conclusion is missing or empty" in errors
    assert "answer_position is missing or not an object" in errors


def test_personas_have_information_filters():
    plan = build_run_plan("AI 是否会重塑教育？")

    for persona in plan["personas"]:
        assert persona["information_filter"]["trusted_sources"]
        assert persona["information_filter"]["distrusted_sources"]
        assert persona["information_filter"]["preferred_evidence"]
        assert persona["information_filter"]["ignored_evidence"]


def test_standard_profile_uses_24_people():
    plan = build_profile_plan("城市是否应该全面推广无人配送？", "standard")

    assert plan["profile"]["person_count"] == 24
    assert len(plan["personas"]) == 24


def test_deep_profile_uses_50_people():
    plan = build_profile_plan("城市是否应该全面推广无人配送？", "deep")

    assert plan["profile"]["person_count"] == 50
    assert len(plan["personas"]) == 50


def test_each_question_gets_a_fresh_cohort():
    first = build_profile_plan("城市是否应该全面推广无人配送？", "standard")
    second = build_profile_plan("AI 是否会重塑教育？", "standard")

    assert first["cohort_id"] != second["cohort_id"]
    assert first["personas"][0]["id"] != second["personas"][0]["id"]


def test_prediction_questions_get_relevant_proof_obligations_without_strategy_bloat():
    for question in ("韩国股市会不会崩盘", "AI会不会产生意识"):
        frame = build_fixture_question_frame(question)
        assert frame["question_types"] == ["prediction", "causal"]
        assert frame["primary_question_type"] == "prediction"
        assert frame["one_pass_coverage"]["automatic_second_round"] is False
        assert frame["one_pass_coverage"]["remaining_gaps_become_explicit_unknowns"] is True
        assert frame["reasoning_grammar"]["adds_model_calls"] is False
        assert frame["reasoning_grammar"]["adds_pipeline_stages"] is False


def test_cohort_geometry_is_measured_without_an_epistemic_second_round():
    cohort = build_fixture_cohort("韩国股市会不会崩盘", 24)
    geometry = cohort.generation["cohort_geometry"]
    assert len(cohort.personas) == 24
    assert cohort.generation["epistemic_passes"] == 1
    assert geometry["person_count"] == 24
    assert geometry["adds_model_calls"] is False
    assert geometry["automatic_second_round"] is False


def test_pre_release_upgrade_keeps_the_serial_stage_topology_unchanged():
    assert PIPELINE_STAGES == [
        "preflight",
        "cohort",
        "retrieval",
        "digital_people",
        "diagnosis",
        "semantic_verdicts",
        "shadow_puzzle",
        "relational_synthesis",
        "finalize",
    ]
    assert not any("binding" in stage or "blind" in stage for stage in PIPELINE_STAGES)


def test_filtered_search_prompt_uses_lossless_compact_json():
    person = build_fixture_cohort("韩国股市会不会崩盘", 1).personas[0]
    content = _planner_messages("韩国股市会不会崩盘", person)[1]["content"]
    decoded = json.loads(content)
    pretty = json.dumps(decoded, ensure_ascii=False, indent=2)

    assert json.loads(content) == decoded
    assert len(content) < len(pretty)


def test_filtered_retrieval_builds_evidence_ledgers():
    result = run_filtered_retrieval("是否应该加快部署自动驾驶卡车？", person_count=3, results_per_query=3)

    assert result["search_provider"] == "mock"
    assert result["retrieval_records"]
    assert result["evidence_ledgers"]
    for ledger in result["evidence_ledgers"]:
        assert ledger["search_strategy"]
        assert ledger["source_decisions"]
        assert "accepted_sources" in ledger
        assert "rejected_sources" in ledger
        assert "ignored_sources" in ledger
        for entry in ledger["entries"]:
            assert entry["entry_id"] == entry["source_id"]
            assert entry["query"]
            assert entry["evidence_use_role"] in {
                "shadow_only",
                "candidate_observation",
                "directional_eligible",
            }


def test_hybrid_retrieval_stack_combines_prior_and_search_layers():
    result = run_filtered_retrieval(
        "城市是否应该延长公共图书馆开放时间？",
        person_count=2,
        provider_name="hybrid-mock",
        results_per_query=2,
    )

    assert result["search_provider"] == "model-prior+mock"
    assert result["retrieval_stack"]["model_prior_enabled"]
    assert result["retrieval_stack"]["external_provider"] == "mock"

    first_record = result["retrieval_records"][0]
    layers = {source["retrieval_layer"] for source in first_record["candidate_sources"]}
    assert "model_prior" in layers
    assert "mock_search" in layers
    assert first_record["search_strategy"]["search_tool_requests"]

    first_ledger = result["evidence_ledgers"][0]
    decision_layers = {decision["result"]["retrieval_layer"] for decision in first_ledger["source_decisions"]}
    assert {"model_prior", "mock_search"} <= decision_layers
    accepted_entries = first_ledger["entries"]
    assert accepted_entries
    assert all("retrieval_layer" in entry and "verification_status" in entry for entry in accepted_entries)


def test_lens_library_loads_classic_diagnostic_tools():
    lenses = load_lenses()
    micro_lenses = load_micro_lenses()
    conflict_rules = load_lens_conflicts()
    calibration_cases = load_calibration_cases()
    shadow_mechanisms = load_shadow_mechanisms()
    capture_protocols = load_capture_protocols()

    assert len(lenses) >= 9
    assert len(micro_lenses) >= 40
    assert len(conflict_rules) >= 10
    assert len(calibration_cases) >= 5
    assert len(shadow_mechanisms) >= 18
    assert len(capture_protocols) >= 10
    assert "bias_heuristics" in lenses
    assert lenses["bias_heuristics"].inversion_rules
    for lens in lenses.values():
        assert lens.mechanism_chain
        assert lens.boundary_conditions
        assert lens.anti_misuse_rules
        assert lens.model_training_notes
    assert {lens.parent_lens for lens in micro_lenses.values()} >= set(lenses)


def test_shadow_diagnosis_uses_lenses_and_builds_inversion_plan():
    result = run_shadow_diagnosis("是否应该加快部署自动驾驶卡车？", person_count=3, results_per_query=3)

    meta_model = result["meta_model"]
    assert meta_model["lens_count"] >= 9
    assert meta_model["micro_lens_count"] >= 40
    assert meta_model["lens_conflict_rule_count"] >= 10
    assert meta_model["calibration_case_count"] >= 5
    assert meta_model["shadow_mechanism_count"] >= 18
    assert meta_model["capture_protocol_stage_count"] >= 10
    assert "shadow_registry" in meta_model
    assert "bias_attribution" in meta_model
    assert "inversion_plan" in meta_model
    assert not meta_model["shadow_registry"], "no-output protocol run must not manufacture a diagnosis"
    assert meta_model["classic_grounding"]
    assert meta_model["shadow_mechanism_ontology"]
    assert meta_model["capture_protocol"]
    assert meta_model["mechanism_coverage"]
    assert len(meta_model["cognitive_chain_scans"]) == 3
    assert all(len(scan["detector_results"]) == 10 for scan in meta_model["cognitive_chain_scans"])
    assert len(meta_model["chain_detector_audit"]) == 10
    assert "differential_diagnosis" in meta_model
    assert meta_model["misdiagnosis_audit"]["audit_questions"]
    assert meta_model["mechanism_audit"]["core_audit_questions"]
    assert meta_model["calibration_reminders"]
    assert meta_model["meta_reflection"]["anti_misuse_rules"]
    assert meta_model["cognitive_chain_evaluation"]["score"] >= 90
    assert meta_model["cognitive_chain_evaluation"]["passed"]
    assert meta_model["meta_modules"]["cognitive_chain_scan"]
    assert meta_model["meta_modules"]["ten_detector_panel"]
    assert meta_model["meta_modules"]["shadow_inverter"]
    assert result["shadow_puzzle"]["negative_space_holes"]
    assert result["shadow_puzzle"]["negative_space_holes"]
    assert result["shadow_puzzle"]["puzzle_quality"]["score"] > 0


def test_meta_model_prompt_includes_full_lens_protocol():
    messages = render_meta_model_messages("是否应该加快部署自动驾驶卡车？", [])
    payload = messages[1]["content"]

    assert "diagnostic_lens_protocol" in payload
    assert "available_lenses" in payload
    assert "available_micro_lenses" in payload
    assert "lens_conflict_rules" in payload
    assert "calibration_cases" in payload
    assert "shadow_mechanisms" in payload
    assert "capture_protocols" in payload
    assert "mechanism_chain" in payload
    assert "boundary_conditions" in payload
    assert "anti_misuse_rules" in payload
    assert "inversion_rules" in payload
    assert "cognitive_chain_scan_protocol" in payload
    assert "cognitive_chain_conclusions" in payload
    assert "chain_detector_coverage" in payload
    assert "plain_language_compressor" in payload
    assert "local_whole_overreach" in payload
    assert "triad_linkage_map" not in payload
    assert "shadow_inverter" in payload


def test_shadow_puzzle_builds_materials_from_diagnosis():
    result = run_shadow_diagnosis("是否应该加快部署自动驾驶卡车？", person_count=3, results_per_query=3)
    puzzle = build_shadow_puzzle(result["question"], result["meta_model"])

    assert puzzle["negative_space_holes"]
    assert not puzzle["bias_corrected_fragments"], "no-output run must not invent bias-corrected fragments"
    assert "interlocking_anchors" in puzzle
    assert puzzle["next_probe"]["diagnostic_stress_tests"]


def test_shadow_puzzle_accepts_structured_and_plain_text_evidence_ledgers():
    diagnosis = {
        "shadow_registry": [],
        "inversion_plan": [],
        "blind_spot_registry": [],
        "disagreement_structure": {},
        "readiness_evaluation": {"score": 0},
        "misdiagnosis_audit": {"audit_questions": []},
    }
    shared_claim = "同一条被不同数字人采信的材料"
    puzzle = build_shadow_puzzle(
        "测试问题",
        diagnosis,
        person_outputs=[
            {
                "person_id": "p01",
                "output": {"evidence_ledger": [shared_claim]},
            },
            {
                "person_id": "p02",
                "output": {
                    "evidence_ledger": {"entries": [{"claim": shared_claim}]}
                },
            },
        ],
    )

    shared = [
        item
        for item in puzzle["interlocking_anchors"]
        if item["type"] == "shared_evidence_claim"
    ]
    assert shared[0]["support_count"] == 2
    assert shared[0]["person_ids"] == ["p01", "p02"]


def test_plain_text_evidence_ledger_preserves_embedded_source_id():
    entries = normalize_output_evidence_entries(
        ["采信一条材料 (ID: duckduckgo-html_abc123)。"]
    )

    assert entries == [
        {
            "claim": "采信一条材料",
            "source_id": "duckduckgo-html_abc123",
        }
    ]


def test_usage_summary_counts_tokens_and_costs():
    summary = summarize_usage(
        [
            {"status": "succeeded", "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}},
            {"status": "failed", "usage": {}},
        ],
        model="deepseek-v4-pro",
        input_price_per_1m_cny=2.0,
        output_price_per_1m_cny=8.0,
    )

    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["total_tokens"] == 1500
    assert summary["pricing"]["estimated_cost_cny"] == 0.006


def test_generic_pricing_settings_take_priority_over_legacy_provider_settings():
    with patch.dict(
        os.environ,
        {
            "BME_INPUT_PRICE_PER_1M_CNY": "2",
            "BME_OUTPUT_PRICE_PER_1M_CNY": "4",
            "DEEPSEEK_INPUT_PRICE_PER_1M_CNY": "20",
            "DEEPSEEK_OUTPUT_PRICE_PER_1M_CNY": "40",
        },
    ):
        summary = summarize_usage(
            [
                {
                    "status": "succeeded",
                    "usage": {
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 1_000_000,
                        "total_tokens": 2_000_000,
                    },
                }
            ],
            model="generic-model",
        )

    assert summary["pricing"]["input_price_per_1m_cny"] == 2.0
    assert summary["pricing"]["output_price_per_1m_cny"] == 4.0
    assert summary["pricing"]["estimated_cost_cny"] == 6.0


def test_other_provider_does_not_inherit_deepseek_pricing():
    with patch.dict(
        os.environ,
        {
            "BME_LLM_PROVIDER_PRESET": "openai",
            "BME_INPUT_PRICE_PER_1M_CNY": "",
            "BME_OUTPUT_PRICE_PER_1M_CNY": "",
            "DEEPSEEK_INPUT_PRICE_PER_1M_CNY": "20",
            "DEEPSEEK_OUTPUT_PRICE_PER_1M_CNY": "40",
        },
    ):
        summary = summarize_usage(
            [{"status": "succeeded", "usage": {"prompt_tokens": 100, "completion_tokens": 50}}],
            model="gpt-4.1",
        )

    assert summary["pricing"]["estimated_cost_cny"] is None


def test_parse_json_content_recovers_fenced_or_trailing_model_output():
    fenced = '```json\n{"conclusion": "ok"}\n```'
    trailing = '模型输出如下：\n{"conclusion": "ok", "confidence": 0.7}\n请审阅。'

    assert parse_json_content(fenced)["conclusion"] == "ok"
    assert parse_json_content(trailing)["confidence"] == 0.7


def test_classify_stance_uses_structured_topic_neutral_answer_position():
    question = "城市是否应该延长公共图书馆开放时间？"

    assert _classify_stance("支持。", question, {"directness": "direct", "direction": "yes"}) == "likely_yes"
    assert _classify_stance("反对。", question, {"directness": "direct", "direction": "no"}) == "likely_no"
    assert _classify_stance("视条件而定。", question, {"directness": "direct", "direction": "conditional"}) == "conditional"
    assert _classify_stance("证据不足。", question, {"directness": "direct", "direction": "uncertain"}) == "undetermined"
    assert _classify_stance("换一个问题。", question, {"directness": "reframed", "direction": "not_applicable"}) == "reframed"


def test_classify_stance_legacy_fallback_stays_topic_neutral():
    question = "项目能否按期完成？"

    assert _classify_stance("目前无法断言项目能否按期完成。", question) == "undetermined"
    assert _classify_stance("目前不能排除按期完成的可能性。", question) == "possibility_open"
    assert _classify_stance("是否按期完成取决于供应商交付。", question) == "conditional"
    assert _classify_stance("核心不在于日期，而在于验收标准。", question) == "reframed"


def test_cross_question_retrieval_context_is_isolated():
    first = run_filtered_retrieval(
        "ALPHA_ONLY：城市是否应该实行四天工作制？",
        person_count=3,
        provider_name="hybrid-mock",
        results_per_query=2,
    )
    second = run_filtered_retrieval(
        "BETA_ONLY：深海采矿是否应该暂停？",
        person_count=3,
        provider_name="hybrid-mock",
        results_per_query=2,
    )

    first_text = json.dumps(first, ensure_ascii=False)
    second_text = json.dumps(second, ensure_ascii=False)
    assert "ALPHA_ONLY" in first_text and "BETA_ONLY" not in first_text
    assert "BETA_ONLY" in second_text and "ALPHA_ONLY" not in second_text
    assert first["cohort_id"] != second["cohort_id"]
    assert first["evidence_mode"] == second["evidence_mode"] == "test_fixture"


def test_mock_provider_is_neutral_and_explicitly_synthetic():
    text = json.dumps(
        [item.__dict__ for item in MockSearchProvider().search("GAMMA_ONLY：测试问题", limit=5)],
        ensure_ascii=False,
    )

    assert "GAMMA_ONLY" in text
    assert "synthetic_fixture" in text
    assert "mock_search" in text
    for stale_term in ["美股", "股价", "互联网泡沫", "主观体验"]:
        assert stale_term not in text


def test_global_live_search_never_silently_falls_back_to_mock():
    with patch.dict(os.environ, {}, clear=False):
        for key in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "BME_SEARCH_ALLOW_MOCK_FALLBACK"]:
            os.environ.pop(key, None)
        provider = choose_regional_provider("global")

    assert not isinstance(provider, MockSearchProvider)
    assert provider.name == "federated-html"
    assert provider.providers[0].name == "google-news-rss"


def test_context_validator_rejects_foreign_person_output():
    retrieval = run_filtered_retrieval(
        "DELTA_ONLY：社区是否应该建设共享厨房？",
        person_count=2,
        provider_name="hybrid-mock",
        results_per_query=1,
    )

    try:
        validate_run_isolation(
            retrieval["question"],
            retrieval["cohort_id"],
            retrieval["personas"],
            question_frame=retrieval["question_frame"],
            cohort_generation=retrieval["cohort_generation"],
            retrieval_records=retrieval["retrieval_records"],
            evidence_ledgers=retrieval["evidence_ledgers"],
            person_results=[{"person_id": "foreign_cohort_person", "output": {}}],
        )
    except ContextIsolationError as exc:
        assert "foreign person_id" in str(exc)
    else:
        raise AssertionError("foreign person output was accepted")


def test_context_validator_rejects_legacy_run_without_isolation_contract():
    try:
        validate_run_isolation(
            "旧格式问题",
            "legacy-cohort",
            [{"id": "legacy-person"}],
            question_frame=None,
            cohort_generation=None,
            retrieval_records=[],
            evidence_ledgers=[],
        )
    except ContextIsolationError as exc:
        assert "legacy run is not isolation-safe" in str(exc)
    else:
        raise AssertionError("legacy run without context isolation was accepted")


def test_run_directories_are_unique_even_when_created_immediately():
    with tempfile.TemporaryDirectory() as tmp:
        first = _make_run_dir(tmp, "standard")
        second = _make_run_dir(tmp, "standard")

    assert first != second
    assert first.name != second.name


def test_live_cohort_parser_accepts_known_json_envelopes_only():
    direct = {"question_frame": {"judgment_target": "x"}, "personas": [{"name": "p"}]}
    wrapped = {"CURRENT_QUESTION": "q", "required_output": direct}
    unrelated = {"payload": direct}

    assert _unwrap_cohort_payload(direct) is direct
    assert _unwrap_cohort_payload(wrapped) is direct
    assert _unwrap_cohort_payload(unrelated) is unrelated


def test_digital_person_context_packet_is_lossless_and_rejects_tampering():
    source = {
        "id": "source_1",
        "query": "test query",
        "title": "A report",
        "url": "https://example.com/report",
        "snippet": "A concrete finding",
        "source_name": "Example Institute",
        "source_type": "研究报告",
        "evidence_type": "统计数据",
        "retrieval_layer": "external_search",
        "verification_status": "external_unverified",
        "retrieval_note": "Live search result.",
    }
    decision = {
        "decision": "accept",
        "trust_score": 0.8,
        "reasons": ["matches the persona filter"],
        "shadow_hint": "may overweight official reports",
        "result": source,
    }
    context = {
        "question_context_id": "question_1",
        "question_frame": {"judgment_target": "test"},
        "search_strategy": {
            "preferred_queries": ["test query"],
            "search_tool_requests": [
                {
                    "tool_call_id": "call_1",
                    "query": "test query",
                    "why_this_person_searches_it": "trusted source",
                    "evidence_sought": "a measured value",
                    "requested_limit": 4,
                    "providers_attempted": ["provider-a"],
                    "providers_succeeded": ["provider-a"],
                    "provider_errors": [],
                    "result_count": 1,
                    "status": "executed",
                }
            ],
        },
        "retrieval_stack": {
            "name": "live",
            "evidence_mode": "three_layer_live",
            "agentic_tool_calls": True,
            "model_prior_enabled": True,
            "external_provider": "provider-a",
            "external_errors": [],
        },
        "candidate_sources": [source],
        "evidence_ledger": {
            "source_decisions": [decision],
            "accepted_sources": [decision],
            "rejected_sources": [],
            "ignored_sources": [],
            "information_shadow_hints": ["one visible filter"],
            "entries": [{"claim": "A concrete finding", "source": source["url"]}],
        },
        "source_layer_rules": {"external_search": "external evidence"},
        "instruction": "stay inside the persona",
    }

    packet = build_digital_person_context_packet(context)
    assert validate_digital_person_context_packet(packet, context) == []
    assert json.dumps(packet).count("A concrete finding") < json.dumps(context).count(
        "A concrete finding"
    )

    packet["source_catalog"][0]["snippet"] = "changed"
    assert "source material changed" in " ".join(
        validate_digital_person_context_packet(packet, context)
    )


def test_digital_person_context_packet_accepts_semantically_empty_fallback_fields():
    context = {
        "question_context_id": "question_1",
        "question_frame": {"judgment_target": "test"},
        "search_strategy": {
            "preferred_queries": ["test query"],
            "search_tool_requests": [
                {
                    "tool": "web_search",
                    "query": "test query",
                    "why_this_person_searches_it": "persona-filtered fallback",
                    "filter_bias": {"trusted_sources": ["official data"]},
                }
            ],
        },
        "retrieval_stack": {
            "name": "live",
            "evidence_mode": "three_layer_live",
            "agentic_tool_calls": True,
            "model_prior_enabled": True,
            "external_provider": "federated-html",
            "external_errors": ["selection unavailable"],
        },
        "candidate_sources": [],
        "evidence_ledger": {
            "source_decisions": [],
            "accepted_sources": [],
            "rejected_sources": [],
            "ignored_sources": [],
            "information_shadow_hints": [],
            "entries": [],
        },
        "source_layer_rules": {"external_search": "external evidence"},
        "instruction": "stay inside the persona",
    }

    packet = build_digital_person_context_packet(context)

    assert validate_digital_person_context_packet(packet, context) == []
    assert packet["search_requests"][0]["providers_attempted"] == []
    assert packet["search_requests"][0]["provider_failure_count"] == 0


def test_cohort_direct_relevance_uses_explicit_contract_without_keyword_guessing():
    assert _is_directly_relevant(
        {
            "relevance_class": "direct",
            "role_summary": "市场观察者",
            "expertise_strong": ["证券市场微观结构"],
            "topic_relevance": "追踪程序化抛售如何扩散",
        }
    )
    assert _is_directly_relevant(
        {
            "relevance_class": "direct_mechanism",
            "role_summary": "市场观察者",
            "expertise_strong": ["证券市场微观结构"],
            "topic_relevance": "追踪程序化抛售如何扩散",
        }
    )
    assert not _is_directly_relevant(
        {
            "relevance_class": "adjacent",
            "role_summary": "相邻领域研究者",
            "expertise_strong": ["比较政治经济学"],
            "topic_relevance": "提供制度背景",
        }
    )


def test_cohort_direct_relevance_keeps_legacy_payload_compatibility():
    assert _is_directly_relevant(
        {
            "role_summary": "证券市场领域专家",
            "expertise_strong": ["波动率"],
            "topic_relevance": "研究市场失灵",
        }
    )


def test_cohort_invalid_combined_label_falls_back_to_actual_relevance_content():
    copied_schema_label = "direct_domain/direct_mechanism/adjacent/affected/executor/counter/institutional"
    assert _is_directly_relevant(
        {
            "relevance_class": copied_schema_label,
            "role_summary": "韩国证券市场领域专家",
            "expertise_strong": ["市场微观结构", "强平传导"],
            "topic_relevance": "直接研究韩国股市崩盘机制",
        }
    )
    assert not _is_directly_relevant(
        {
            "relevance_class": copied_schema_label,
            "role_summary": "相邻领域观察者",
            "expertise_strong": ["比较政治"],
            "topic_relevance": "只提供制度背景",
        }
    )


def test_cohort_request_demonstrates_one_relevance_enum_value():
    request = _cohort_request("韩国股市会不会崩盘", 24)
    example = request["required_output"]["personas"][0]
    assert example["relevance_class"] == "direct_domain"
    assert "/" not in example["relevance_class"]


def _complete_cohort_candidate(name: str, relevance_class: str) -> dict:
    return {
        "name": name,
        "role_summary": f"{name}的角色",
        "relevance_class": relevance_class,
        "cognitive_frames": ["框架"],
        "expertise_strong": ["专长一", "专长二"],
        "expertise_weak": ["边界"],
        "values": {"准确": 1.0},
        "time_horizon": "一年",
        "risk_attitude": "审慎",
        "analysis_levels": ["机制"],
        "hidden_biases": ["确认偏误"],
        "temperament": "克制",
        "information_filter": {
            "trusted_sources": ["一手数据"],
            "distrusted_sources": ["匿名传闻"],
            "preferred_evidence": ["时间序列"],
            "ignored_evidence": ["轶事"],
            "query_style": ["精确检索"],
        },
        "topic_relevance": "与本题有明确关系",
    }


def test_missing_query_style_is_derived_from_the_same_persons_filter():
    question = "如果海外股市见顶，A股会不会跟随见顶？"
    candidate = _complete_cohort_candidate("跨市场传染研究者", "direct")
    candidate["information_filter"]["query_style"] = []
    _frame, people, _selection = _normalize_generated_payload(
        {
            "question_frame": {
                "judgment_target": "判断海外股市见顶向A股传导的条件",
                "question_type": "预测",
                "key_terms": ["海外股市见顶", "A股见顶"],
                "disputed_definitions": ["见顶的时间尺度"],
                "scope": {
                    "population": "股票市场",
                    "geography": "海外与中国",
                    "time_horizon": "未来一年",
                },
                "relevant_knowledge_domains": ["跨市场传染", "中国资本市场"],
                "relevant_evidence_dimensions": [
                    "估值",
                    "流动性",
                    "盈利",
                    "资金流",
                ],
                "misleading_substitutions": ["同步下跌等于共同见顶"],
            },
            "personas": [candidate],
        },
        question,
        1,
        "test-cohort",
    )

    assert people[0].information_filter.query_style == [
        "优先检索一手数据中的时间序列，并降低只提供轶事的材料权重"
    ]


def test_surplus_cohort_selection_preserves_constraints_instead_of_truncating():
    # The first item is the surplus. A positional truncation would retain it and
    # discard the final affected-person perspective.
    candidates = [{"name": "不完整候选", "relevance_class": "unknown"}]
    candidates.extend(
        _complete_cohort_candidate(f"直接专家{i}", "direct_domain") for i in range(8)
    )
    candidates.extend(
        _complete_cohort_candidate(f"相邻观察者{i}", "adjacent") for i in range(11)
    )
    candidates.extend(
        [
            _complete_cohort_candidate("执行者", "executor"),
            _complete_cohort_candidate("反证者", "counter"),
            _complete_cohort_candidate("制度视角", "institutional"),
            _complete_cohort_candidate("受影响者", "affected"),
            _complete_cohort_candidate("补充相邻者", "adjacent"),
        ]
    )

    selected, metadata = _select_persona_payloads(candidates, 24)

    assert len(selected) == 24
    assert all(raw.get("name") != "不完整候选" for raw in selected)
    assert sum(1 for raw in selected if _is_directly_relevant(raw)) == 8
    assert {raw["relevance_class"] for raw in selected}.issuperset(
        {"adjacent", "affected", "executor", "counter", "institutional"}
    )
    assert metadata["candidate_count"] == 25
    assert metadata["surplus_count"] == 1
    assert metadata["excluded_candidates"][0]["name"] == "不完整候选"


def test_live_cohort_repairs_one_missing_direct_expert_without_rewriting_the_group():
    question = "韩国股市会不会崩盘"
    frame = {
        "judgment_target": "未来一年韩国股市是否会发生系统性大跌",
        "question_type": "预测",
        "key_terms": ["韩国股市", "崩盘"],
        "disputed_definitions": ["崩盘阈值"],
        "scope": {"geography": "韩国", "time_horizon": "未来一年"},
        "relevant_knowledge_domains": ["韩国资本市场", "金融稳定"],
        "relevant_evidence_dimensions": [
            "估值",
            "盈利",
            "信用",
            "流动性",
        ],
        "misleading_substitutions": [],
    }
    personas = [
        _complete_cohort_candidate(f"直接专家{i}", "direct_domain")
        for i in range(1, 8)
    ]
    non_direct_classes = [
        "adjacent",
        "affected",
        "executor",
        "counter",
        "institutional",
    ]
    personas.extend(
        _complete_cohort_candidate(
            f"非直接观察者{i}", non_direct_classes[(i - 1) % len(non_direct_classes)]
        )
        for i in range(1, 18)
    )
    initial_payload = {"question_frame": frame, "personas": personas}
    replacement = _complete_cohort_candidate("信用周期专家", "direct_mechanism")
    responses = iter(
        [
            LLMResponse(
                content=json.dumps(initial_payload, ensure_ascii=False),
                raw={"choices": [{"finish_reason": "stop"}]},
                usage={"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                message={"role": "assistant", "content": "cohort"},
                tool_calls=[],
                reasoning_content="",
            ),
            LLMResponse(
                content=json.dumps({"replacements": [replacement]}, ensure_ascii=False),
                raw={"choices": [{"finish_reason": "stop"}]},
                usage={"prompt_tokens": 50, "completion_tokens": 80, "total_tokens": 130},
                message={"role": "assistant", "content": "repair"},
                tool_calls=[],
                reasoning_content="",
            ),
        ]
    )

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, *_args: object, **_kwargs: object) -> LLMResponse:
            return next(responses)

    shared_client = FakeClient()
    cohort = generate_live_cohort(
        question,
        24,
        model="fixture-model",
        retries=0,
        client_factory=lambda: shared_client,
    )

    repair = cohort.generation["targeted_repairs"][0]
    assert len(cohort.personas) == 24
    assert cohort.generation["attempts"] == 1
    assert cohort.generation["usage"]["api_calls"] == 2
    assert cohort.generation["mode"] == "llm_question_native_with_targeted_repair"
    assert repair["found_before"] == 7
    assert repair["required"] == 8
    assert repair["replacement_count"] == 1
    assert repair["preserved_person_count"] == 23
    assert repair["replacements"][0]["added_name"] == "信用周期专家"


def test_live_cohort_fills_one_missing_person_without_rewriting_twenty_three():
    question = "韩国股市会不会崩盘"
    frame = {
        "judgment_target": "未来一年韩国股市是否会发生系统性大跌",
        "question_type": "预测",
        "key_terms": ["韩国股市", "崩盘"],
        "disputed_definitions": ["崩盘阈值"],
        "scope": {"geography": "韩国", "time_horizon": "未来一年"},
        "relevant_knowledge_domains": ["韩国资本市场", "金融稳定"],
        "relevant_evidence_dimensions": ["估值", "盈利", "信用", "流动性"],
        "misleading_substitutions": [],
    }
    personas = [
        _complete_cohort_candidate(f"直接专家{i}", "direct_domain")
        for i in range(1, 8)
    ]
    non_direct_classes = [
        "adjacent",
        "affected",
        "executor",
        "counter",
        "institutional",
    ]
    personas.extend(
        _complete_cohort_candidate(
            f"非直接观察者{i}", non_direct_classes[(i - 1) % len(non_direct_classes)]
        )
        for i in range(1, 17)
    )
    assert len(personas) == 23
    addition = _complete_cohort_candidate("市场流动性专家", "direct_mechanism")
    responses = iter(
        [
            LLMResponse(
                content=json.dumps(
                    {"question_frame": frame, "personas": personas},
                    ensure_ascii=False,
                ),
                raw={"choices": [{"finish_reason": "stop"}]},
                usage={"prompt_tokens": 100, "completion_tokens": 190, "total_tokens": 290},
                message={"role": "assistant", "content": "short cohort"},
                tool_calls=[],
                reasoning_content="",
            ),
            LLMResponse(
                content=json.dumps({"additions": [addition]}, ensure_ascii=False),
                raw={"choices": [{"finish_reason": "stop"}]},
                usage={"prompt_tokens": 40, "completion_tokens": 70, "total_tokens": 110},
                message={"role": "assistant", "content": "addition"},
                tool_calls=[],
                reasoning_content="",
            ),
        ]
    )

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def chat(self, *_args: object, **_kwargs: object) -> LLMResponse:
            return next(responses)

    with patch("bme_model.cohort.DeepSeekClient", FakeClient):
        cohort = generate_live_cohort(
            question,
            24,
            model="fixture-model",
            retries=2,
        )

    repair = cohort.generation["targeted_repairs"][0]
    assert len(cohort.personas) == 24
    assert cohort.generation["attempts"] == 1
    assert cohort.generation["usage"]["api_calls"] == 2
    assert cohort.generation["attempt_errors"] == []
    assert len(cohort.generation["targeted_repairs"]) == 1
    assert repair["policy"] == "targeted_persona_shortage_completion_v1"
    assert repair["found_before"] == 23
    assert repair["required"] == 24
    assert repair["preserved_person_count"] == 23
    assert repair["minimum_direct_additions"] == 1
    assert repair["direct_additions"] == 1
    assert repair["additions"][0]["slot"] == 24
    assert repair["additions"][0]["added_name"] == "市场流动性专家"
    assert cohort.personas[-1].name == "市场流动性专家#24"


def test_baidu_json_parser_supports_nested_result_path():
    payload = {
        "response": {
            "docs": [
                {"headline": "标题", "target": "https://example.com/a", "abstract": "摘要"}
            ]
        }
    }

    results = _parse_generic_json_results(
        "baidu-json",
        "测试",
        payload,
        limit=3,
        results_path="response.docs",
    )

    assert len(results) == 1
    assert results[0].title == "标题"
    assert results[0].url == "https://example.com/a"


def test_build_report_contains_truth_contour_and_person_appendix():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "person_outputs").mkdir()
        summary = {
            "question": "复杂问题",
            "cohort_id": "abc123",
            "model": "deepseek-v4-pro",
            "profile": {"person_count": 1},
            "usage": {"succeeded": 1, "total_tokens": 100},
            "readiness_evaluation": {"score": 90},
            "puzzle_quality": {"material_count": 2, "score": 80},
        }
        diagnosis = {
            "shadow_registry": [
                {
                    "person_id": "p1",
                    "description": "偏好单一证据",
                    "source_stage": "information_filter",
                    "puzzle_use": "negative_space",
                }
            ],
            "collective_blind_spots": [
                {
                    "dimension": "直接服务数据",
                    "description": "被忽略",
                    "ignored_count": 2,
                    "accepted_count": 0,
                    "relevance_status": "candidate_unverified",
                }
            ],
            "inversion_plan": [
                {"person_id": "p1", "resulting_material": "负空间洞", "estimated_bias_vector": {"direction": "低估直接服务数据"}}
            ],
            "disagreement_structure": {
                "output_based": {
                    "stance_clusters": {"likely_yes": ["p1"]},
                    "conclusion_samples": [
                        {"person_id": "p1", "stance": "likely_yes", "confidence": 0.7, "conclusion": "应该推进"}
                    ],
                }
            },
        }
        puzzle = {
            "interlocking_anchors": [],
            "next_probe": {"blind_spot_probes": ["补直接服务数据"], "diagnostic_stress_tests": [], "recommended_new_personas": []},
            "puzzle_quality": {"material_count": 2, "score": 80},
        }
        retrieval = {
            "question": "复杂问题",
            "cohort_id": "abc123",
            "search_provider": "mock",
            "evidence_mode": "test_fixture",
            "question_frame": {
                "context_id": "abc123",
                "exact_question": "复杂问题",
            },
            "cohort_generation": {
                "mode": "deterministic_question_isolated_fixture",
                "context_isolation": "only_current_question",
            },
            "personas": [{"id": "p1", "name": "测试数字人"}],
            "retrieval_records": [
                {
                    "person_id": "p1",
                    "question_context_id": "abc123",
                    "question_frame": {
                        "context_id": "abc123",
                        "exact_question": "复杂问题",
                    },
                    "retrieval_stack": {"external_errors": []},
                }
            ],
            "evidence_ledgers": [
                {
                    "person_id": "p1",
                    "question": "复杂问题",
                    "search_strategy": {"preferred_queries": ["复杂问题 直接服务数据"], "query_intent": ["寻找服务证据"]},
                    "source_decisions": [{"decision": "accept"}],
                    "accepted_sources": [],
                    "ignored_sources": [],
                    "rejected_sources": [],
                }
            ],
        }
        person = {
            "person_id": "p1",
            "person": {
                "id": "p1",
                "name": "测试数字人",
                "role_summary": "测试",
                "cognitive_frames": ["系统思维"],
                "expertise_strong": ["公共服务"],
                "expertise_weak": ["法律"],
                "values": {"safety": 0.5},
                "hidden_biases": ["确认偏误"],
                "risk_attitude": "谨慎",
                "information_filter": {
                    "trusted_sources": ["直接服务数据"],
                    "distrusted_sources": ["宣传稿"],
                    "preferred_evidence": ["直接服务数据"],
                    "ignored_evidence": ["营销叙事"],
                },
            },
            "output": {
                "core_assumptions": ["延时服务确有稳定需求"],
                "reasoning_path": ["从直接服务数据推理"],
                "conclusion": "应该推进",
                "confidence": 0.7,
                "what_i_underweighted": ["反向证据"],
                "what_would_change_my_mind": ["需求不足或成本过高"],
            },
        }
        for name, payload in {
            "summary.json": summary,
            "diagnosis.json": diagnosis,
            "shadow_puzzle.json": puzzle,
            "retrieval.json": retrieval,
            "person_outputs/p1.json": person,
        }.items():
            with (run_dir / name).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

        report_path = build_run_report(run_dir)
        report = report_path.read_text(encoding="utf-8")

    assert "综合结论：真相轮廓" in report
    assert "1 个数字人的判断与阴影" in report
    assert "测试证据模式" in report
    assert "不能单独证明现实方向" in report
    assert "candidate_unverified" not in report
    assert "查询关键词" in report
    assert "bme:puzzle-handoff" in report
    assert "puzzle-arrival" in report


def test_classic_knowledge_audit_is_visible_without_treating_books_as_evidence():
    detective = {
        "classic_diagnostic_traces": [{"trace_id": "trace_1"}],
        "inversion_certificates": [
            {"certificate_id": "certificate_1", "authorization": "restricted"}
        ],
        "diagnostic_bridges": [{"bridge_id": "bridge_1"}],
        "puzzle_pieces": [
            {"knowledge_trace": {"classic_trace_ids": ["trace_1"]}}
        ],
        "relation_certificates": [
            {
                "relation_id": "relation_1",
                "status": "accepted",
                "plain_language_explanation": "两块材料共享同一个来源，不能重复计票。",
                "knowledge_provenance": {
                    "classic_trace_ids": ["trace_1"],
                    "relation_use_mode": "lineage_carried_not_relation_basis",
                    "inference_permission": "structural_constraint_only",
                },
            }
        ],
        "detective_evaluation": {"classic_used_as_world_evidence_count": 0},
    }
    contour = {
        "classic_knowledge_audit": {
            "statement_count": 3,
            "classic_traced_statement_count": 1,
            "classic_used_as_world_evidence_count": 0,
        }
    }

    rendered = _classic_knowledge_story(detective, contour)

    assert "经典知识在这轮判断里做了什么" in rendered
    assert "不靠经典成立" in rendered
    assert "把经典当现实证据 0 次" in rendered
