from __future__ import annotations

import re
from typing import Any


QUESTION_TYPES = (
    "fact",
    "causal",
    "prediction",
    "normative",
    "conceptual",
    "strategic",
)


_TYPE_LABELS = {
    "fact": "事实判断",
    "causal": "因果解释",
    "prediction": "趋势预测",
    "normative": "规范判断",
    "conceptual": "概念辨析",
    "strategic": "战略决策",
}


_TYPE_MARKERS = {
    "prediction": (
        "会不会",
        "是否会",
        "将会",
        "未来",
        "很快",
        "趋势",
        "前景",
        "风险",
        "概率",
        "预测",
    ),
    "causal": (
        "为什么",
        "为何",
        "原因",
        "导致",
        "造成",
        "机制",
        "因果",
    ),
    "normative": (
        "该不该",
        "应不应该",
        "是否应该",
        "正当",
        "公平",
        "伦理",
        "权利",
    ),
    "conceptual": (
        "是什么",
        "何谓",
        "定义",
        "算不算",
        "意味着什么",
        "本质",
    ),
    "strategic": (
        "怎么办",
        "怎么做",
        "如何",
        "选择",
        "决策",
        "策略",
        "方案",
    ),
}


_PROOF_OBLIGATIONS = {
    "fact": (
        "明确命题、对象、范围和时间",
        "核对一手记录或直接测量",
        "寻找来源独立的交叉印证",
        "检查反例、缺失数据和测量边界",
    ),
    "causal": (
        "把原因到结果的中间机制逐段展开",
        "检查时间先后、混杂因素和反向因果",
        "比较能够解释同一现象的替代机制",
        "寻找反事实、自然对照或失败案例",
        "说明机制成立的对象、范围和边界条件",
    ),
    "prediction": (
        "明确预测对象、判断指标和时间窗口",
        "建立历史基准率与可比情景",
        "识别当前领先指标及其数据截止时间",
        "展开触发结果的因果链与反馈回路",
        "保留相反情景、尾部风险和失效条件",
        "列出会改写判断的可观察更新信号",
    ),
    "normative": (
        "分开事实判断与价值判断",
        "识别受影响者、权利、伤害和责任主体",
        "公开相互冲突的价值排序与取舍",
        "检查执行可行性、分配后果和替代方案",
        "保留不可通约的价值分支",
    ),
    "conceptual": (
        "列出竞争定义及各自判定标准",
        "区分必要条件、充分条件与典型特征",
        "用边界案例和反例测试定义",
        "说明采用不同定义会怎样改写答案",
    ),
    "strategic": (
        "明确目标、约束、时间尺度和决策主体",
        "比较可行选项、机会成本和基准方案",
        "分析其他行动者的反应与二阶后果",
        "区分可逆试验与不可逆承诺",
        "列出触发调整、退出或止损的信号",
    ),
}


_ANSWER_FORMS = {
    "fact": "先回答当前能确认什么，再说明证据边界。",
    "causal": "给出最能解释现象的机制链，同时保留替代解释和失效边界。",
    "prediction": "给出当前主倾向、时间窗口、触发条件和会改写判断的信号。",
    "normative": "先交代事实基础，再公开价值分支、受影响者和取舍。",
    "conceptual": "先说明采用的定义，再展示定义分支和边界案例。",
    "strategic": "给出当前最合适的行动方向、适用条件、退出条件和替代方案。",
}


def build_reasoning_grammar(
    question: str,
    declared_type: Any = None,
    *,
    disputed_definitions: list[str] | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a multi-label proof contract without another model call."""

    question_types = detect_question_types(question, declared_type)
    primary = question_types[0]
    obligations: list[str] = []
    for question_type in question_types:
        obligations.extend(_PROOF_OBLIGATIONS[question_type])
    obligations = _unique(obligations)[:10]

    ambiguities = list(disputed_definitions or [])
    active_scope = scope or {}
    for key, label in (
        ("population", "对象范围尚未明确"),
        ("geography", "地理范围尚未明确"),
        ("time_horizon", "时间窗口尚未明确"),
    ):
        value = str(active_scope.get(key) or "").strip()
        if not value or "待界定" in value or "未明确" in value:
            ambiguities.append(label)

    return {
        "version": "bme.question-grammar.v1",
        "question_types": question_types,
        "question_type_labels": [_TYPE_LABELS[item] for item in question_types],
        "primary_question_type": primary,
        "proof_obligations": obligations,
        "answer_form": _ANSWER_FORMS[primary],
        "known_ambiguities": _unique(ambiguities),
        "adds_model_calls": False,
        "adds_pipeline_stages": False,
    }


def detect_question_types(question: str, declared_type: Any = None) -> list[str]:
    declared = str(declared_type or "").casefold()
    if declared.strip() in {"判断或决策", "判断", "开放问题"}:
        declared = ""
    text = f"{question} {declared}".casefold()
    detected: list[str] = []
    for question_type in (
        "prediction",
        "causal",
        "normative",
        "conceptual",
        "strategic",
    ):
        if any(marker.casefold() in text for marker in _TYPE_MARKERS[question_type]):
            detected.append(question_type)

    aliases = {
        "预测": "prediction",
        "解释": "causal",
        "因果": "causal",
        "规范": "normative",
        "伦理": "normative",
        "概念": "conceptual",
        "定义": "conceptual",
        "决策": "strategic",
        "战略": "strategic",
        "方案": "strategic",
        "事实": "fact",
    }
    for marker, question_type in aliases.items():
        if marker in declared and question_type not in detected:
            detected.append(question_type)

    if not detected:
        detected.append("fact")
    if "prediction" in detected and "causal" not in detected:
        detected.append("causal")
    return detected[:3]


def merge_evidence_dimensions(
    declared_dimensions: list[str], grammar: dict[str, Any]
) -> list[str]:
    """Keep model-native dimensions and fill only proof obligations it omitted."""

    dimensions = _unique(declared_dimensions)
    for obligation in grammar.get("proof_obligations", []) or []:
        if _covered_by_existing_dimension(obligation, dimensions):
            continue
        dimensions.append(str(obligation))
        if len(dimensions) >= 10:
            break
    return dimensions


def _covered_by_existing_dimension(obligation: str, dimensions: list[str]) -> bool:
    obligation_terms = _terms(obligation)
    if not obligation_terms:
        return False
    for dimension in dimensions:
        terms = _terms(dimension)
        if terms and len(obligation_terms.intersection(terms)) >= min(2, len(obligation_terms)):
            return True
    return False


def _terms(value: Any) -> set[str]:
    text = str(value or "").casefold()
    chunks = re.findall(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", text)
    return {chunk for chunk in chunks if chunk not in {"判断", "问题", "说明", "检查"}}


def _unique(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
