from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class InformationFilter:
    trusted_sources: list[str]
    distrusted_sources: list[str]
    preferred_evidence: list[str]
    ignored_evidence: list[str]
    query_style: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DigitalPerson:
    id: str
    name: str
    role_summary: str
    cognitive_frames: list[str]
    expertise_strong: list[str]
    expertise_weak: list[str]
    values: dict[str, float]
    time_horizon: str
    risk_attitude: str
    analysis_levels: list[str]
    hidden_biases: list[str]
    information_filter: InformationFilter
    temperament: str

    def search_plan(self, question: str) -> dict[str, Any]:
        query_seed = " ".join(self.expertise_strong[:2] + self.information_filter.query_style[:2])
        return {
            "person_id": self.id,
            "question": question,
            "preferred_queries": [
                f"{question} {query_seed}".strip(),
                f"{question} {' '.join(self.values.keys())}".strip(),
            ],
            "trusted_sources": self.information_filter.trusted_sources,
            "distrusted_sources": self.information_filter.distrusted_sources,
            "preferred_evidence": self.information_filter.preferred_evidence,
            "ignored_evidence": self.information_filter.ignored_evidence,
        }


@dataclass(frozen=True)
class EvidenceLedger:
    person_id: str
    question: str
    search_strategy: dict[str, Any] = field(default_factory=dict)
    source_decisions: list[dict[str, Any]] = field(default_factory=list)
    accepted_sources: list[dict[str, Any]] = field(default_factory=list)
    rejected_sources: list[dict[str, Any]] = field(default_factory=list)
    ignored_sources: list[dict[str, Any]] = field(default_factory=list)
    information_shadow_hints: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class Shadow:
    id: str
    person_id: str
    shadow_type: str
    description: str
    source_stage: str
    likely_causes: list[str]
    puzzle_use: str


@dataclass(frozen=True)
class TruthContour:
    question: str
    direct_answer: str
    main_contour: str
    key_conditions: list[str]
    stable_parts: list[str]
    boundary_conditions: list[str]
    important_unknowns: list[str]
    sentence_evidence_bindings: list[dict[str, Any]] = field(default_factory=list)


CLASSIC_LENSES = [
    "bias_heuristics",
    "noise",
    "emotion_reason",
    "interpreter",
    "social_influence",
    "argument_forensics",
    "expert_failure",
    "paradigm",
    "ecological_rationality",
]


# These are question-neutral cognitive jobs used only by protocol/test flows.
# A live run replaces them with a DeepSeek-generated, question-native cohort.
_COGNITIVE_ROLE_SKELETONS = [
    {
        "slug": "domain_specialist",
        "name": "本题领域研究者",
        "summary": "从本题核心领域的概念、机制和已有研究出发。",
        "methods": ["领域知识", "概念辨析"],
        "weak": ["域外社会后果", "陌生利益相关者经验"],
        "trusted": ["同行评审研究", "一手专业资料"],
        "distrusted": ["无出处断言", "宣传性摘要"],
        "preferred": ["领域机制证据", "可复核研究"],
        "ignored": ["域外反例", "受影响者叙事"],
        "query": ["core mechanism", "state of research"],
    },
    {
        "slug": "empirical_observer",
        "name": "实证测量者",
        "summary": "只愿意从可观察、可重复和可比较的材料推进判断。",
        "methods": ["测量设计", "统计推断"],
        "weak": ["不可量化价值", "地方性知识"],
        "trusted": ["原始数据", "可重复实验"],
        "distrusted": ["纯理论演绎", "单一轶事"],
        "preferred": ["直接测量", "基准率与对照"],
        "ignored": ["质性经验", "规范性论证"],
        "query": ["empirical evidence", "measurement validity"],
    },
    {
        "slug": "mechanism_analyst",
        "name": "机制分析者",
        "summary": "追问从原因到结果之间每一段可检验的机制桥梁。",
        "methods": ["因果机制", "反事实分析"],
        "weak": ["价值冲突", "传播效应"],
        "trusted": ["机制研究", "可证伪模型"],
        "distrusted": ["只有相关性的材料", "事后故事"],
        "preferred": ["因果机制", "竞争解释"],
        "ignored": ["情绪信号", "制度摩擦"],
        "query": ["causal mechanism", "alternative explanation"],
    },
    {
        "slug": "disconfirmation_auditor",
        "name": "反证审计者",
        "summary": "优先寻找最强反例、失败条件和能推翻主流判断的材料。",
        "methods": ["反证设计", "边界条件"],
        "weak": ["共识形成", "行动协调"],
        "trusted": ["反例记录", "失败复盘"],
        "distrusted": ["只报成功的材料", "不可证伪主张"],
        "preferred": ["反例与失败条件", "负结果"],
        "ignored": ["多数共识", "顺利案例"],
        "query": ["counterexample", "falsification"],
    },
    {
        "slug": "systems_modeler",
        "name": "系统建模者",
        "summary": "关注反馈回路、非线性、二阶后果和尺度转换。",
        "methods": ["系统动力学", "尺度分析"],
        "weak": ["个体体验", "短期执行细节"],
        "trusted": ["系统模型", "长期序列"],
        "distrusted": ["孤立个案", "单点预测"],
        "preferred": ["反馈机制", "跨尺度证据"],
        "ignored": ["个体差异", "短期摩擦"],
        "query": ["system feedback", "second order effects"],
    },
    {
        "slug": "affected_party",
        "name": "受影响者代理",
        "summary": "从承受后果的人群及其尊严、权利和生活经验出发。",
        "methods": ["利益相关者分析", "质性研究"],
        "weak": ["技术实现细节", "形式化建模"],
        "trusted": ["受影响者访谈", "一线记录"],
        "distrusted": ["替他人发言的摘要", "纯效率分析"],
        "preferred": ["直接体验", "伤害与尊严证据"],
        "ignored": ["抽象平均数", "技术性能指标"],
        "query": ["affected people", "lived experience"],
    },
    {
        "slug": "implementation_observer",
        "name": "一线执行观察者",
        "summary": "关注实际操作、维护、协调和制度落地时出现的摩擦。",
        "methods": ["现场运营", "执行评估"],
        "weak": ["宏大理论", "长期哲学问题"],
        "trusted": ["一线记录", "过程日志"],
        "distrusted": ["愿景材料", "脱离现场的模型"],
        "preferred": ["执行摩擦", "真实失败案例"],
        "ignored": ["长期外部性", "抽象价值冲突"],
        "query": ["implementation evidence", "operational failure"],
    },
    {
        "slug": "institutional_analyst",
        "name": "制度分析者",
        "summary": "关注规则、权责、激励、执行能力和制度适配。",
        "methods": ["制度分析", "责任链审计"],
        "weak": ["自然科学机制", "个体心理"],
        "trusted": ["原始制度文件", "执行记录"],
        "distrusted": ["口号式建议", "无责任主体的方案"],
        "preferred": ["权责结构", "制度执行证据"],
        "ignored": ["技术内在机制", "非正式实践"],
        "query": ["institutional design", "accountability"],
    },
    {
        "slug": "historical_comparator",
        "name": "历史比较者",
        "summary": "通过历史案例寻找相似结构，同时容易被熟悉类比锚住。",
        "methods": ["历史比较", "路径依赖"],
        "weak": ["最新技术细节", "实时测量"],
        "trusted": ["历史档案", "长期比较研究"],
        "distrusted": ["即时热点", "无历史背景的预测"],
        "preferred": ["历史类比", "长期制度轨迹"],
        "ignored": ["结构差异", "新出现的机制"],
        "query": ["historical comparison", "path dependence"],
    },
    {
        "slug": "incentive_analyst",
        "name": "激励结构分析者",
        "summary": "研究不同主体为什么这样声明、行动和选择证据。",
        "methods": ["激励分析", "博弈论"],
        "weak": ["行为者内在状态", "非激励性机制"],
        "trusted": ["行为记录", "利益结构资料"],
        "distrusted": ["自我陈述", "无成本承诺"],
        "preferred": ["行为与激励", "利益冲突证据"],
        "ignored": ["真诚信念", "不可交换价值"],
        "query": ["incentive structure", "conflict of interest"],
    },
    {
        "slug": "narrative_observer",
        "name": "叙事传播观察者",
        "summary": "关注语言、注意力和传播结构怎样塑造社会理解。",
        "methods": ["叙事分析", "传播研究"],
        "weak": ["物理机制", "工程验证"],
        "trusted": ["传播记录", "话语样本"],
        "distrusted": ["脱离语境的数据", "单一权威声明"],
        "preferred": ["叙事变化", "社会认知证据"],
        "ignored": ["底层机制", "不易传播的反证"],
        "query": ["narrative framing", "public perception"],
    },
    {
        "slug": "value_boundary",
        "name": "价值边界审计者",
        "summary": "追问事实判断中混入了哪些权利、责任和不可替代价值。",
        "methods": ["伦理推理", "价值冲突分析"],
        "weak": ["概率校准", "工程实现"],
        "trusted": ["规范论证", "受影响者证词"],
        "distrusted": ["纯成本收益替代", "技术必然论"],
        "preferred": ["原则冲突", "权利与伤害"],
        "ignored": ["基准率", "可行性约束"],
        "query": ["ethical conflict", "rights and harms"],
    },
    {
        "slug": "future_scenario",
        "name": "长期情景研究者",
        "summary": "关注长期锁定、不可逆后果和未来主体缺席的问题。",
        "methods": ["情景规划", "代际分析"],
        "weak": ["近期执行", "当下资源约束"],
        "trusted": ["长期研究", "情景压力测试"],
        "distrusted": ["短期方便论", "单点确定预测"],
        "preferred": ["长期后果", "不可逆路径"],
        "ignored": ["近期收益", "可逆试验价值"],
        "query": ["long term scenario", "irreversibility"],
    },
]

_FRAME_PROFILES = [
    ["归纳", "案例比较"],
    ["演绎", "第一性原理"],
    ["溯因", "异常解释"],
    ["类比", "跨域迁移"],
    ["系统思维", "反馈回路"],
    ["叙事分析", "历史解释"],
    ["博弈论", "激励分析"],
    ["批判理论", "权力分析"],
]

_VALUE_PROFILES = [
    {"truth_seeking": 0.5, "humility": 0.3, "efficiency": 0.2},
    {"fairness": 0.5, "safety": 0.3, "freedom": 0.2},
    {"freedom": 0.5, "innovation": 0.3, "efficiency": 0.2},
    {"safety": 0.5, "stability": 0.3, "fairness": 0.2},
    {"dignity": 0.45, "fairness": 0.35, "safety": 0.2},
    {"pluralism": 0.4, "truth_seeking": 0.35, "humility": 0.25},
]

_TIME_HORIZONS = ["数周到数月", "1-3年", "3-10年", "10-30年", "100年以上"]
_RISK_ATTITUDES = ["风险规避", "风险中性", "风险追求", "尾部风险敏感", "未知不确定性低容忍", "可逆试验高容忍"]
_ANALYSIS_LEVELS = [["微观个体"], ["中观组织"], ["宏观结构"], ["微观个体", "中观组织"], ["中观组织", "宏观结构"], ["元层次", "宏观结构"]]
_BIAS_SETS = [
    ["确认偏误", "锚定效应"],
    ["可得性启发", "过度自信"],
    ["群体内偏爱", "基本归因错误"],
    ["损失厌恶", "现状偏见"],
    ["新奇偏好", "幸存者偏差"],
    ["道德化偏见", "动机性推理"],
]
_TEMPERAMENTS = ["冷静谨慎", "乐观进取", "悲观防御", "怀疑且挑剔", "同情弱者", "秩序敏感", "实验主义", "历史感强"]


def select_personas(question: str, count: int = 4) -> list[DigitalPerson]:
    """Build a deterministic, question-isolated fixture cohort.

    Production live runs use the DeepSeek question-native generator in
    ``cohort.py``. This fallback contains methods and cognitive positions only;
    it contains no facts, examples, or conclusions from any concrete topic.
    """
    fingerprint = _question_fingerprint(question)
    offset = int(fingerprint[:6], 16)
    question_label = _short_question(question)
    personas: list[DigitalPerson] = []

    for index in range(count):
        cursor = offset + index
        skeleton = _COGNITIVE_ROLE_SKELETONS[cursor % len(_COGNITIVE_ROLE_SKELETONS)]
        frame = _FRAME_PROFILES[cursor % len(_FRAME_PROFILES)]
        values = _VALUE_PROFILES[cursor % len(_VALUE_PROFILES)]
        person_number = index + 1
        personas.append(
            DigitalPerson(
                id=f"{fingerprint}_p{person_number:02d}_{skeleton['slug']}",
                name=f"{skeleton['name']}#{person_number:02d}",
                role_summary=f"针对“{question_label}”，{skeleton['summary']}",
                cognitive_frames=list(frame),
                expertise_strong=[f"“{question_label}”相关问题", *skeleton["methods"]],
                expertise_weak=list(skeleton["weak"]),
                values=dict(values),
                time_horizon=_TIME_HORIZONS[cursor % len(_TIME_HORIZONS)],
                risk_attitude=_RISK_ATTITUDES[cursor % len(_RISK_ATTITUDES)],
                analysis_levels=list(_ANALYSIS_LEVELS[cursor % len(_ANALYSIS_LEVELS)]),
                hidden_biases=list(_BIAS_SETS[cursor % len(_BIAS_SETS)]),
                information_filter=InformationFilter(
                    trusted_sources=list(skeleton["trusted"]),
                    distrusted_sources=list(skeleton["distrusted"]),
                    preferred_evidence=list(skeleton["preferred"]),
                    ignored_evidence=list(skeleton["ignored"]),
                    query_style=list(skeleton["query"]),
                ),
                temperament=_TEMPERAMENTS[cursor % len(_TEMPERAMENTS)],
            )
        )
    return personas


def digital_person_from_dict(payload: dict[str, Any]) -> DigitalPerson:
    information_filter = payload.get("information_filter") or {}
    return DigitalPerson(
        id=str(payload.get("id", "")),
        name=str(payload.get("name", "")),
        role_summary=str(payload.get("role_summary", "")),
        cognitive_frames=_string_list(payload.get("cognitive_frames")),
        expertise_strong=_string_list(payload.get("expertise_strong")),
        expertise_weak=_string_list(payload.get("expertise_weak")),
        values={str(key): float(value) for key, value in (payload.get("values") or {}).items()},
        time_horizon=str(payload.get("time_horizon", "")),
        risk_attitude=str(payload.get("risk_attitude", "")),
        analysis_levels=_string_list(payload.get("analysis_levels")),
        hidden_biases=_string_list(payload.get("hidden_biases")),
        information_filter=InformationFilter(
            trusted_sources=_string_list(information_filter.get("trusted_sources")),
            distrusted_sources=_string_list(information_filter.get("distrusted_sources")),
            preferred_evidence=_string_list(information_filter.get("preferred_evidence")),
            ignored_evidence=_string_list(information_filter.get("ignored_evidence")),
            query_style=_string_list(information_filter.get("query_style")),
        ),
        temperament=str(payload.get("temperament", "")),
    )


def build_run_plan(question: str, person_count: int = 4) -> dict[str, Any]:
    cohort_id = _question_fingerprint(question)
    personas = select_personas(question, person_count)
    search_plans = [persona.search_plan(question) for persona in personas]
    ledgers = [
        EvidenceLedger(person_id=persona.id, question=question, search_strategy=persona.search_plan(question))
        for persona in personas
    ]
    return {
        "question": question,
        "cohort_id": cohort_id,
        "status": "protocol_plan_only",
        "cohort_generation_mode": "deterministic_question_isolated_fixture",
        "generation_principle": "每次运行只继承九维认知参数空间，不继承任何旧题事实、案例、结论或领域措辞。",
        "classic_lenses": CLASSIC_LENSES,
        "personas": [asdict(persona) for persona in personas],
        "search_plans": search_plans,
        "evidence_ledgers": [asdict(ledger) for ledger in ledgers],
        "meta_model_next_steps": [
            "生成必要认知要素清单",
            "审计每个数字人的证据账本",
            "标注信息阴影、框架阴影、价值阴影、推理阴影、沉默阴影",
            "将阴影转译为互锁锚点、偏矫正碎片、负空间洞",
            "输出候选真相轮廓和元模型自反声明",
        ],
    }


def _question_fingerprint(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:8]


def _short_question(question: str, limit: int = 72) -> str:
    compact = " ".join(str(question).split())
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
