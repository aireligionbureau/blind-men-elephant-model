from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .context import validate_artifact_questions, validate_run_isolation
from .runtime import atomic_write_text


STANCE_LABELS = {
    "likely_yes": "明确倾向会发生",
    "possibility_open": "只认为不能排除",
    "likely_no": "明确倾向不会发生",
    "undetermined": "认为证据不足，暂不判断",
    "reframed": "没有直答，重新定义了问题",
    "conditional": "认为取决于条件",
    "support": "倾向支持",
    "oppose": "倾向反对",
    "missing": "输出缺失或被截断",
    "unclear": "结论未能归类",
}

REASON_PATTERNS = [
    ("可观察证据与测量", ["证据", "实验", "测量", "指标", "可观察", "实证", "数据", "检验"]),
    ("工程机制与可实现性", ["架构", "机制", "参数", "计算", "涌现", "复杂系统", "工程", "技术可能"]),
    ("安全风险与预防", ["安全", "事故", "漏洞", "失效", "尾部风险", "不可逆", "预防原则"]),
    ("历史与制度类比", ["历史", "类比", "制度演化", "制度重构", "周期", "路径依赖"]),
    ("社会认知与叙事", ["情绪", "叙事", "拟人化", "传播", "舆论", "信任", "社会承认", "文化"]),
    ("政策、监管与法律", ["监管", "法律", "诉讼", "版权", "责任", "合规", "政策", "程序正义"]),
    ("伦理、权利与伤害", ["伦理", "权利", "伤害", "道德", "价值对齐", "主体地位"]),
    ("经济激励与现实效用", ["估值", "现金流", "盈利", "成本", "市场", "激励", "生产率", "现实效用"]),
    ("劳动、健康与社会影响", ["劳动", "工人", "失业", "弱势", "公共卫生", "心理健康", "社会安全网"]),
    ("生态、能源与基础设施", ["生态", "能源", "水", "碳", "气候", "基础设施", "电力"]),
]


def build_run_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    run_path = Path(run_dir)
    summary = _read_json(run_path / "summary.json")
    diagnosis = _read_json(run_path / "diagnosis.json")
    puzzle = _read_json(run_path / "shadow_puzzle.json")
    retrieval = _read_json(run_path / "retrieval.json") if (run_path / "retrieval.json").exists() else {}
    detective = _read_optional_json(run_path / "detective.json")
    truth_contour = _read_optional_json(run_path / "truth_contour.json")
    person_results = _read_person_outputs(run_path / "person_outputs")
    output_file = Path(output_path) if output_path is not None else run_path / "report.html"

    question = str(summary.get("question") or retrieval.get("question") or diagnosis.get("question") or "").strip()
    validate_artifact_questions(
        question,
        {"summary": summary, "diagnosis": diagnosis, "shadow_puzzle": puzzle, "retrieval": retrieval},
    )
    if retrieval:
        if not retrieval.get("cohort_id") or not retrieval.get("personas"):
            raise RuntimeError("retrieval artifact lacks an isolation-safe cohort")
        summary_cohort = str(summary.get("cohort_id", "")).strip()
        retrieval_cohort = str(retrieval.get("cohort_id", "")).strip()
        if summary_cohort and summary_cohort != retrieval_cohort:
            raise RuntimeError("summary and retrieval belong to different cohorts")
        validate_run_isolation(
            question,
            retrieval_cohort,
            retrieval["personas"],
            question_frame=retrieval.get("question_frame"),
            cohort_generation=retrieval.get("cohort_generation"),
            retrieval_records=retrieval.get("retrieval_records"),
            evidence_ledgers=retrieval.get("evidence_ledgers"),
            person_results=person_results,
        )

    html_text = _render_report(
        summary,
        diagnosis,
        puzzle,
        retrieval,
        person_results,
        run_path,
        detective=detective,
        truth_contour=truth_contour,
    )
    atomic_write_text(output_file, html_text)
    return output_file


def _render_report(
    summary: dict[str, Any],
    diagnosis: dict[str, Any],
    puzzle: dict[str, Any],
    retrieval: dict[str, Any],
    person_results: list[dict[str, Any]],
    run_path: Path,
    *,
    detective: dict[str, Any] | None = None,
    truth_contour: dict[str, Any] | None = None,
) -> str:
    question = summary.get("question", diagnosis.get("question", "未命名问题"))
    usage = summary.get("combined_usage") or summary.get("usage", {})
    reasons = _reason_clusters(person_results)
    truth = (
        _truth_contour_view(truth_contour)
        if truth_contour and truth_contour.get("direct_answer")
        else _build_truth_contour(
            question, summary, diagnosis, puzzle, retrieval, person_results, reasons
        )
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(question)} - 盲人摸象模型运行报告</title>
  <script>
    try {{
      if (window.sessionStorage.getItem("bme:puzzle-handoff") === "1") {{
        document.documentElement.classList.add("puzzle-arrival");
        window.sessionStorage.removeItem("bme:puzzle-handoff");
      }}
    }} catch (_error) {{}}
  </script>
  <style>
    :root {{
      color-scheme: light;
      --ink: #162027;
      --muted: #68737d;
      --line: #d9dee3;
      --paper: #f5f6f2;
      --panel: #ffffff;
      --soft: #eef2f4;
      --amber: #bd7b2c;
      --green: #2f806a;
      --red: #b84f48;
      --blue: #426fa8;
      --violet: #725aa6;
      --coal: #2e3942;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.58;
    }}
    html.puzzle-arrival header,
    html.puzzle-arrival main {{
      opacity: 0;
      transform: translateY(12px);
      animation: report-arrive .72s ease .12s forwards;
    }}
    html.puzzle-arrival main {{ animation-delay: .3s; }}
    @keyframes report-arrive {{
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    header {{
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px;
    }}
    h1, h2, h3, h4 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 34px; line-height: 1.18; }}
    h2 {{ font-size: 22px; margin-bottom: 14px; }}
    h3 {{ font-size: 17px; margin-bottom: 8px; }}
    h4 {{ font-size: 14px; margin: 12px 0 6px; }}
    p {{ margin: 8px 0; }}
    section {{ margin: 18px 0; }}
    a {{ color: #2d6093; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .sub {{ color: var(--muted); margin-top: 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .two {{
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(0, .92fr);
      gap: 14px;
    }}
    .three {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-label {{ color: var(--muted); font-size: 13px; }}
    .metric-value {{ font-size: 26px; font-weight: 720; margin-top: 4px; }}
    .truth {{
      background: #fff;
      border: 1px solid #d9d1c4;
      border-left: 5px solid var(--amber);
      border-radius: 8px;
      padding: 18px;
    }}
    .truth .statement {{
      font-size: 18px;
      font-weight: 720;
      line-height: 1.58;
      margin: 4px 0 12px;
    }}
    .notice {{
      border-left: 4px solid var(--amber);
      background: #fff8ea;
      padding: 12px 14px;
      border-radius: 6px;
      color: #60451d;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 158px minmax(0, 1fr) 52px;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
      font-size: 14px;
    }}
    .track {{
      height: 13px;
      background: #e8ecef;
      border-radius: 999px;
      overflow: hidden;
    }}
    .fill {{ height: 100%; border-radius: 999px; background: var(--blue); }}
    .likely_yes .fill, .support .fill {{ background: var(--green); }}
    .likely_no .fill, .oppose .fill {{ background: var(--red); }}
    .conditional .fill, .possibility_open .fill {{ background: var(--amber); }}
    .undetermined .fill, .reframed .fill {{ background: var(--violet); }}
    .unclear .fill {{ background: var(--coal); }}
    .missing .fill {{ background: #a7b0b8; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 25px;
      padding: 3px 9px;
      border-radius: 999px;
      background: var(--soft);
      color: #34404b;
      font-size: 12px;
      margin: 3px 4px 3px 0;
      word-break: break-all;
    }}
    .pill.good {{ background: #e7f4ef; color: #1e624f; }}
    .pill.bad {{ background: #f7e9e7; color: #8b3b36; }}
    .pill.mid {{ background: #f6eddc; color: #7b541d; }}
    .muted {{ color: var(--muted); }}
    .small {{ font-size: 13px; }}
    .list {{ display: grid; gap: 10px; }}
    .item {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    .item:first-child {{ border-top: 0; padding-top: 0; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 650; background: #fbfcfd; }}
    details.person {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      overflow: hidden;
    }}
    details.person + details.person {{ margin-top: 12px; }}
    details.person summary {{
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
      background: #fbfcfd;
    }}
    details.person summary::-webkit-details-marker {{ display: none; }}
    .summary-title {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin-bottom: 5px;
    }}
    .conclusion {{
      color: var(--ink);
      font-size: 14px;
    }}
    .shadow-line {{
      color: #7b3f38;
      font-size: 13px;
      margin-top: 6px;
    }}
    .chain-headline {{
      font-size: 16px;
      font-weight: 700;
      line-height: 1.55;
      color: #6f302c;
    }}
    .chain-issue {{
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }}
    .chain-issue p {{ margin: 6px 0; }}
    .person-body {{ padding: 16px; border-top: 1px solid var(--line); }}
    .person-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    ul, ol {{ padding-left: 20px; margin: 8px 0; }}
    li {{ margin: 4px 0; }}
    .links a {{ display: inline-block; margin-right: 12px; }}
    .knowledge-flow {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 0;
      margin: 16px 0;
      border-top: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
    }}
    .knowledge-step {{ padding: 14px; border-left: 1px solid var(--line); }}
    .knowledge-step:first-child {{ border-left: 0; }}
    .knowledge-step strong {{ display: block; margin-bottom: 5px; }}
    details.audit {{ border-top: 1px solid var(--line); padding-top: 12px; }}
    details.audit summary {{ cursor: pointer; font-weight: 700; }}
    details.audit .audit-body {{ padding-top: 10px; }}
    @media (max-width: 900px) {{
      .grid, .two, .three, .person-grid, .knowledge-flow {{ grid-template-columns: 1fr; }}
      .knowledge-step {{ border-left: 0; border-top: 1px solid var(--line); }}
      .knowledge-step:first-child {{ border-top: 0; }}
      h1 {{ font-size: 26px; }}
      .bar-row {{ grid-template-columns: 125px minmax(0, 1fr) 42px; }}
      table {{ font-size: 13px; }}
      details.person summary {{ grid-template-columns: 1fr; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      html.puzzle-arrival header,
      html.puzzle-arrival main {{
        opacity: 1;
        transform: none;
        animation: none;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>{_e(question)}</h1>
      <div class="sub">盲人摸象模型 · 标准版 {_profile_people(summary)} 数字人 · cohort {_e(summary.get("cohort_id", ""))} · 模型 {_e(summary.get("model", ""))} · 搜索 {_e(retrieval.get("search_provider", "unknown"))} · 证据模式 {_e(retrieval.get("evidence_mode", "unknown"))}</div>
      <div class="grid">
        {_metric("数字人", str(usage.get("succeeded", len(person_results))) + "/" + str(_profile_people(summary)))}
        {_metric("总 Token", _num(usage.get("total_tokens")))}
        {_metric("可见阴影", _num(len(diagnosis.get("shadow_registry", []))))}
        {_metric("待核验漏看候选", _num(len(diagnosis.get("collective_blind_spots", []))))}
      </div>
    </div>
  </header>
  <main>
    <div class="wrap">
      <section class="truth">
        <h2>综合结论：真相轮廓</h2>
        <div class="statement">{_e(truth["current_formulation"])}</div>
        <div class="three">
          {_mini_block(truth["first_label"], truth["dominant_tendency"])}
          {_mini_block(truth["second_label"], truth["main_reasons"])}
           {_mini_block(truth["third_label"], truth["fragile_points"])}
        </div>
        {_evidence_status_line(retrieval, truth_contour)}
      </section>

      <section>
        <h2>{_profile_people(summary)} 个数字人的判断与阴影</h2>
        <p class="muted">每一条先给结论摘要和“它的问题在哪里”。点击展开，可以看到认知身份证、它怎么上网、核心假设、推理路径、结论与自我修正条件，以及元模型指出的具体阴影。</p>
        {_person_cards(person_results, retrieval, diagnosis)}
      </section>

      <section>
        <details class="audit">
          <summary>查看这条轮廓怎样形成，以及经典知识怎样参与</summary>
          <div class="audit-body">
            {_evidence_mode_notice(retrieval)}
            {_contour_notice(truth_contour)}
            {_detective_relation_story(detective)}
            {_classic_knowledge_story(detective, truth_contour)}
          </div>
        </details>
      </section>

      <section class="card">
        <h2>原始文件</h2>
        <div class="links">
          <a href="summary.json">summary.json</a>
          <a href="diagnosis.json">diagnosis.json</a>
          <a href="shadow_puzzle.json">shadow_puzzle.json</a>
          <a href="puzzle_pieces.json">puzzle_pieces.json</a>
          <a href="detective.json">detective.json</a>
          <a href="contour_constraints.json">contour_constraints.json</a>
          <a href="contour_candidates.json">contour_candidates.json</a>
          <a href="truth_contour.json">truth_contour.json</a>
          <a href="retrieval.json">retrieval.json</a>
          <a href="run.json">run.json</a>
        </div>
        <p class="muted">报告文件位置：{_e(str(run_path / "report.html"))}</p>
      </section>
    </div>
  </main>
</body>
</html>
"""


def _evidence_mode_notice(retrieval: dict[str, Any]) -> str:
    mode = str(retrieval.get("evidence_mode", "unknown"))
    records = retrieval.get("retrieval_records", []) or []
    external_errors = [
        str(error)
        for record in records
        for error in record.get("retrieval_stack", {}).get("external_errors", []) or []
        if str(error).strip()
    ]
    if mode == "test_fixture":
        return (
            '<section class="notice"><strong>测试证据模式：</strong>'
            "本报告使用隔离的 mock 材料验证流程。它们不是真实搜索结果，不能支撑现实结论；"
            "此处只能检查数字人的过滤与元模型的诊断是否按预期工作。</section>"
        )
    if mode == "prior_only":
        return (
            '<section class="notice"><strong>仅先验模式：</strong>'
            "本轮只有数字人的模型先验，没有外部事实。报告可以显影它们原本相信什么，"
            "不能据此把群体倾向当作现实判断。</section>"
        )
    if external_errors:
        return (
            '<section class="notice"><strong>外部检索不完整：</strong>'
            f"本轮记录到 {len(external_errors)} 次检索失败。报告会保留这些失败，"
            "不会把缺失的外部证据伪装成已经搜索过。</section>"
        )
    if mode in {"hybrid_live", "live_external"}:
        return (
            '<section class="notice"><strong>真实外部证据模式：</strong>'
            "搜索结果仍是待核验线索；来源真实性、相关性和对结论的支持力度要分别审计。</section>"
        )
    if mode == "three_layer_live":
        return (
            '<section class="notice"><strong>三层真实检索模式：</strong>'
            "本轮同时记录数字人的模型先验、模型实际工具调用，以及外部搜索源返回的网页。"
            "搜索结果仍是待核验线索，工具调用过程和来源失败会保留在证据账本中。</section>"
        )
    return (
        '<section class="notice"><strong>证据模式未标明：</strong>'
        "无法确认本轮材料来自真实搜索、模型先验还是测试夹具，结论应暂缓使用。</section>"
    )


def _evidence_status_line(
    retrieval: dict[str, Any], contour: dict[str, Any] | None
) -> str:
    accepted_external = 0
    accepted_prior = 0
    verified = 0
    for ledger in retrieval.get("evidence_ledgers", []) or []:
        for decision in ledger.get("source_decisions", []) or []:
            if str(decision.get("decision") or "").lower() not in {
                "accept",
                "accepted",
            }:
                continue
            result = decision.get("result") or {}
            if result.get("retrieval_layer") == "model_prior":
                accepted_prior += 1
                continue
            accepted_external += 1
            if result.get("verification_status") in {
                "page_verified",
                "primary_verified",
                "cross_verified",
            }:
                verified += 1

    evaluation = (contour or {}).get("contour_evaluation") or {}
    bound_sentences = int(evaluation.get("sentence_binding_count") or 0)
    binding_text = (
        f"最终轮廓的 {bound_sentences} 个句子分别绑定了自己的依据。"
        if (contour or {}).get("evidence_binding_contract") == "sentence_level.v1"
        else "本轮结果仍采用整段依据记录。"
    )
    if verified:
        evidence_text = (
            f"本轮采信 {accepted_external} 条外部材料，其中 {verified} 条完成页面、原始来源或交叉核验；"
            "其余材料只能作为待核验线索。"
        )
    else:
        evidence_text = (
            f"本轮采信 {accepted_external} 条外部线索，但没有材料完成页面、原始来源或交叉核验；"
            "这些线索可以帮助形成条件和发现阴影，不能单独证明现实方向。"
        )
    prior_text = (
        f"另记录 {accepted_prior} 条模型先验，只用于显影数字人原本相信什么。"
        if accepted_prior
        else ""
    )
    return (
        '<p class="muted small" style="margin-top:16px">'
        + _e(evidence_text + prior_text + binding_text)
        + "</p>"
    )


def _truth_contour_view(contour: dict[str, Any]) -> dict[str, str]:
    conditions = "；".join(
        str(item).strip()
        for item in contour.get("key_conditions", [])[:4]
        if str(item).strip()
    )
    if not conditions:
        conditions = str(contour.get("why_this_contour") or "尚未识别出会改写主轮廓的关键条件。")
    unknowns = "；".join(
        str(item).strip()
        for item in contour.get("important_unknowns", [])[:4]
        if str(item).strip()
    )
    if not unknowns:
        unknowns = str(contour.get("confidence_statement") or "暂无额外登记。")
    return {
        "current_formulation": str(contour.get("direct_answer", "")),
        "dominant_tendency": str(contour.get("main_contour", "")),
        "main_reasons": conditions,
        "fragile_points": unknowns,
        "first_label": "当前主轮廓",
        "second_label": "在什么条件下会改写",
        "third_label": "仍然不知道什么",
    }


def _contour_notice(contour: dict[str, Any] | None) -> str:
    if not contour:
        return (
            '<section class="notice">当前页面仍使用旧版群体概览，尚未运行侦探层与轮廓层。'
            "它只能展示人群怎样回答，不能把人群倾向当作真相。</section>"
        )
    if contour.get("generation_mode") == "live_adversarial_inference":
        return (
            '<section class="notice"><strong>这条轮廓怎样产生：</strong>'
            "先连接不同数字人看对、看错和没看见的部分，再让最强替代轮廓来攻击主候选，最后按来源独立性、条件兼容和解释力裁决。"
            "人数没有被当作真相权重；每句实质判断都能在原始文件中追到具体材料和关系。</section>"
        )
    return (
        '<section class="notice"><strong>轮廓尚未完成实时反演：</strong>'
        "当前只完成了可重复的材料整理和硬规则检查。页面会明确停在证据允许的位置，不会用多数意见补出答案。"
        "运行侦探层的语义连接、最强反轮廓和最终裁决后，才会生成实质性真相轮廓。</section>"
    )


def _detective_relation_story(detective: dict[str, Any] | None) -> str:
    if not detective:
        return ""
    accepted = [
        item
        for item in detective.get("relation_certificates", []) or []
        if item.get("status") == "accepted"
    ]
    priority = {
        "independent_convergence": 0,
        "conditional_complement": 1,
        "scope_refinement": 2,
        "causal_relay": 3,
        "direct_conflict": 4,
        "apparent_conflict": 5,
        "blind_spot_fill": 6,
        "correlated_error": 7,
        "shared_source": 8,
        "shared_model_prior": 9,
    }
    accepted.sort(
        key=lambda item: (priority.get(str(item.get("relation_type")), 20), item.get("relation_id", ""))
    )
    if not accepted:
        body = (
            '<p class="muted">侦探层目前没有找到经得起反驳的跨人连接。'
            "这不是空白报告，而是在阻止相似意见被误写成相互印证。</p>"
        )
    else:
        items = []
        for relation in accepted[:10]:
            explanation = str(relation.get("plain_language_explanation", "")).strip()
            if not explanation:
                continue
            endpoint_count = len(relation.get("piece_ids", []) or [])
            items.append(
                '<div class="item"><p>'
                + _e(explanation)
                + f'</p><p class="muted small">这条连接落在 {endpoint_count} 块可追溯材料上；竞争解释：'
                + _e(relation.get("competing_explanation", ""))
                + "</p></div>"
            )
        body = '<div class="list">' + "".join(items) + "</div>"
    return (
        '<section class="card"><h2>这些局部怎样彼此连接</h2>'
        '<p class="muted">这里只展示已经通过对抗裁决的具体连接；同方向但同源的声音不会重复计票。</p>'
        + body
        + "</section>"
    )


def _classic_knowledge_story(
    detective: dict[str, Any] | None,
    contour: dict[str, Any] | None,
) -> str:
    """Show how classic diagnostics constrained this run without becoming evidence."""

    if not detective or not detective.get("classic_diagnostic_traces"):
        return ""
    traces = detective.get("classic_diagnostic_traces", []) or []
    certificates = detective.get("inversion_certificates", []) or []
    bridges = detective.get("diagnostic_bridges", []) or []
    pieces = detective.get("puzzle_pieces", []) or []
    relations = detective.get("relation_certificates", []) or []
    evaluation = detective.get("detective_evaluation", {}) or {}
    contour_audit = (contour or {}).get("classic_knowledge_audit", {}) or {}
    diagnostic_piece_count = sum(
        bool((item.get("knowledge_trace") or {}).get("classic_trace_ids"))
        for item in pieces
    )
    authorizations = Counter(
        str(item.get("authorization", "unknown")) for item in certificates
    )
    accepted_count = sum(item.get("status") == "accepted" for item in relations)
    rejected_count = sum(item.get("status") == "rejected" for item in relations)
    misuse_count = (
        int(evaluation.get("classic_used_as_world_evidence_count", 0) or 0)
        + int(contour_audit.get("classic_used_as_world_evidence_count", 0) or 0)
    )

    relation_rows = []
    reviewable = [
        item
        for item in relations
        if (item.get("knowledge_provenance") or {}).get("classic_trace_ids")
    ]
    reviewable.sort(
        key=lambda item: (
            item.get("status") != "accepted",
            item.get("relation_id", ""),
        )
    )
    use_explanations = {
        "diagnostic_context_considered": (
            "审理这条连接时实际查看了经典诊断；它只限制材料怎样使用，不能证明市场事实。"
        ),
        "lineage_carried_not_relation_basis": (
            "经典痕迹随材料保留下来，但这条连接本身不靠经典成立。"
        ),
        "not_applicable": "这条连接只比较内容、条件或来源，经典没有参与证明。",
    }
    permission_explanations = {
        "structural_constraint_only": "最多只能形成结构边界或降低独立性。",
        "diagnostic_only": "只能说明思考链的问题，不能进入方向性答案。",
        "blocked": "不能参与拼图，只能登记为未知。",
        "not_applicable": "这条连接不需要经典授权。",
    }
    for relation in reviewable[:6]:
        knowledge = relation.get("knowledge_provenance", {}) or {}
        status = {
            "accepted": "保留",
            "rejected": "否决",
            "unresolved": "暂不裁决",
        }.get(str(relation.get("status")), str(relation.get("status", "未知")))
        relation_rows.append(
            '<div class="item"><p><strong>'
            + _e(status)
            + "：</strong>"
            + _e(relation.get("plain_language_explanation", ""))
            + "</p><p class=\"muted small\">经典在这里的作用："
            + _e(
                use_explanations.get(
                    str(knowledge.get("relation_use_mode")),
                    "这条连接保留了诊断谱系，但没有获得额外事实权重。",
                )
            )
            + " "
            + _e(
                permission_explanations.get(
                    str(knowledge.get("inference_permission")),
                    "推断权限以原始关系证书为准。",
                )
            )
            + f" 关联诊断轨迹 {len(knowledge.get('classic_trace_ids', []) or [])} 条。</p></div>"
        )
    relation_story = (
        '<div class="list">' + "".join(relation_rows) + "</div>"
        if relation_rows
        else '<p class="muted">本轮采纳关系没有实际调用经典诊断；系统没有为了显示知识库而强行套用。</p>'
    )
    statement_count = int(contour_audit.get("statement_count", 0) or 0)
    traced_statement_count = int(
        contour_audit.get("classic_traced_statement_count", 0) or 0
    )
    auth_text = "，".join(
        f"{key} {value} 条" for key, value in sorted(authorizations.items())
    ) or "无"
    return f"""<section class="card">
      <h2>经典知识在这轮判断里做了什么</h2>
      <p>经典不是用来替市场作证的。它只帮助元模型说明一条思考为什么可能偏、这块阴影能怎样进入拼图，以及必须在哪里停手。</p>
      <div class="knowledge-flow">
        <div class="knowledge-step"><strong>1. 找到思考链依据</strong><span class="muted">留下 {len(traces)} 条诊断轨迹，每条都指向具体数字人和具体思考环节。</span></div>
        <div class="knowledge-step"><strong>2. 限定阴影用法</strong><span class="muted">{len(bridges)} 条桥接依据把诊断连到 {diagnostic_piece_count} 块阴影材料；授权结果：{_e(auth_text)}。</span></div>
        <div class="knowledge-step"><strong>3. 审问跨人连接</strong><span class="muted">审理 {len(relations)} 条候选连接，保留 {accepted_count} 条，否决 {rejected_count} 条；经典相同不算相互印证。</span></div>
        <div class="knowledge-step"><strong>4. 约束最终轮廓</strong><span class="muted">最终 {statement_count} 项陈述中，{traced_statement_count} 项继承了经典诊断边界；把经典当现实证据 {misuse_count} 次。</span></div>
      </div>
      <details class="audit">
        <summary>展开查看经典怎样限制具体连接</summary>
        <div class="audit-body">{relation_story}</div>
      </details>
      <p class="muted small">完整依据可在 <a href="puzzle_pieces.json">puzzle_pieces.json</a>、<a href="detective.json">detective.json</a> 和 <a href="truth_contour.json">truth_contour.json</a> 中按 ID 逐步反查。</p>
    </section>"""


def _build_truth_contour(
    question: str,
    summary: dict[str, Any],
    diagnosis: dict[str, Any],
    puzzle: dict[str, Any],
    retrieval: dict[str, Any],
    person_results: list[dict[str, Any]],
    reasons: list[dict[str, Any]],
) -> dict[str, str]:
    del summary, puzzle
    clusters = _stance_clusters(diagnosis)
    total = max(1, sum(len(ids) for ids in clusters.values()))
    ordered = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
    dominant_key, dominant_ids = ordered[0] if ordered else ("unclear", [])
    dominant_label = STANCE_LABELS.get(dominant_key, dominant_key)
    counterweights = [
        f"{STANCE_LABELS.get(key, key)} {len(ids)} 人"
        for key, ids in ordered[1:4]
        if key != "missing"
    ]
    blind_spots = diagnosis.get("collective_blind_spots", [])[:3]
    blind_text = "；".join(
        _evidence_gap_story(item.get("dimension", "未知证据"), question)["short"]
        for item in blind_spots
    ) or "暂无显著登记"
    reason_text = "、".join(f"{item['label']}({item['count']}人)" for item in reasons[:3]) or "结论理由分散，尚未形成稳定主题"
    provider = retrieval.get("search_provider", "unknown")
    evidence_mode = retrieval.get("evidence_mode", "unknown")
    missing_count = len(clusters.get("missing", []))

    if dominant_key == "reframed":
        formulation = (
            f"对于“{question}”，人数最多的一组没有直接回答，而是“{dominant_label}”"
            f"（{len(dominant_ids)}/{total}）。改写问题有时能补充背景，但不能代替对用户原问题的回答；"
            "应逐一检查它们究竟补充了必要前提，还是避开了最难判断的部分。"
        )
    elif dominant_key in {
        "likely_yes",
        "likely_no",
        "possibility_open",
        "undetermined",
        "conditional",
        "support",
        "oppose",
    }:
        formulation = (
            f"对于“{question}”，这群数字人中人数最多的回答方向是“{dominant_label}”"
            f"（{len(dominant_ids)}/{total}）。这说明哪种判断在本轮认知样本中占上风，"
            "不等于该判断在现实中的概率；它仍要经受反向证据、关键前提和漏看事实的检验。"
        )
    else:
        formulation = (
            f"对于“{question}”，这群数字人尚未摸出可稳定归类的回答。最大一组是“{dominant_label}”"
            f"（{len(dominant_ids)}/{total}）。当前材料更适合展示分歧和推理缺口，还不足以形成可靠轮廓。"
        )

    return {
        "current_formulation": formulation,
        "dominant_tendency": f"{dominant_label}，约 {round(len(dominant_ids) / total * 100)}%。反向或补充方向：{_join_or(counterweights, '暂未形成稳定反向簇')}。",
        "main_reasons": reason_text,
        "fragile_points": (
            f"需要优先核验的漏看候选是：{blind_text}。它们只有被证明与本题直接相关后，才算真正盲区。"
            f"本轮证据模式为 {evidence_mode}，来源为 {provider}，缺失或截断输出 {missing_count} 个。"
        ),
        "first_label": "这群数字人的主倾向",
        "second_label": "他们主要抓住了什么",
        "third_label": "这轮判断最脆弱的地方",
    }


def _stance_examples(samples: list[dict[str, Any]]) -> str:
    by_stance: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_stance[str(sample.get("stance", "unknown"))].append(sample)
    chunks = []
    for stance, items in sorted(by_stance.items(), key=lambda pair: len(pair[1]), reverse=True)[:4]:
        examples = " / ".join(
            _short(item.get("conclusion", ""), 82)
            for item in items[:2]
            if item.get("conclusion")
        )
        if not examples:
            continue
        chunks.append(
            f"""<div class="item">
              <h3>{_e(STANCE_LABELS.get(stance, stance))}</h3>
              <p>{_e(examples)}</p>
            </div>"""
        )
    return "".join(chunks)


def _reason_story(reasons: list[dict[str, Any]]) -> str:
    if not reasons:
        return '<p class="muted">这轮数字人的理由太分散，尚未形成明显共同抓手。</p>'
    chunks = []
    for item in reasons[:6]:
        examples = "".join(f"<li>{_e(example)}</li>" for example in item["examples"][:3])
        chunks.append(
            f"""<div class="item">
              <h3>{_e(item["label"])} <span class="pill">{item["count"]} 人提到</span></h3>
              <ul>{examples}</ul>
            </div>"""
        )
    return "".join(chunks)


def _blind_spots_plain(question: str, items: list[dict[str, Any]], retrieval: dict[str, Any]) -> str:
    if not items:
        return '<p class="muted">暂未登记明显的共同漏看候选。</p>'
    provider = retrieval.get("search_provider", "unknown")
    evidence_mode = retrieval.get("evidence_mode", "unknown")
    chunks = [
        f"<p class=\"muted\">下面列的是账本中被多人绕开的证据候选，不是已经证实的盲区。"
        f"先要证明它与“{_e(question)}”直接相关，再看具体事实会把结论推向哪里。"
        f"当前证据模式：{_e(evidence_mode)}；来源：{_e(provider)}。</p>"
    ]
    for item in items[:6]:
        story = _evidence_gap_story(item.get("dimension", "未知证据"), question)
        relevance_status = item.get("relevance_status", "candidate_unverified")
        chunks.append(
            f"""<div class="item">
              <h3>{_e(story["title"])}</h3>
              <p>{_e(story["why"])}</p>
              <p>{_e(story["missing"])}</p>
              <p class="muted">本轮这类证据被忽略 {item.get("ignored_count", 0)} 次，只被采信 {item.get("accepted_count", 0)} 次；相关性状态：{_e(relevance_status)}。{_e(story["could_change"])}</p>
            </div>"""
        )
    return "".join(chunks)


def _evidence_gap_story(dimension: str, question: str) -> dict[str, str]:
    return {
        "title": f"“{dimension}”是否藏着会改写本题判断的事实",
        "why": (
            f"账本显示，多名数字人在回答“{question}”时绕开了这类材料。"
            "这只证明共同的信息选择，不自动证明这类材料就是本题的关键证据。"
        ),
        "missing": (
            f"还缺两步：先说明“{dimension}”与本题判断目标之间的具体机制关系；"
            "再给出可核验的事实、数据、案例或反例，并说明它支持或反驳哪一个前提。"
        ),
        "could_change": (
            "如果相关性成立，反向材料会移动轮廓，同向材料会让轮廓更稳；"
            "如果无法说明相关性，这一项应从盲区名单删除。"
        ),
        "short": f"{dimension} 是否与本题直接相关，以及相关证据指向什么",
    }


def _person_cards(
    person_results: list[dict[str, Any]],
    retrieval: dict[str, Any],
    diagnosis: dict[str, Any],
) -> str:
    if not person_results:
        return '<div class="card"><p class="muted">暂无数字人输出。</p></div>'
    question = retrieval.get("question") or diagnosis.get("question", "")
    ledgers = {ledger.get("person_id"): ledger for ledger in retrieval.get("evidence_ledgers", [])}
    shadows_by_person = defaultdict(list)
    for shadow in diagnosis.get("shadow_registry", []):
        shadows_by_person[shadow.get("person_id")].append(shadow)
    chain_scans_by_person = {
        scan.get("person_id"): scan
        for scan in diagnosis.get("cognitive_chain_scans", [])
    }
    stance_by_person = {
        sample.get("person_id"): sample.get("stance")
        for sample in _conclusion_samples(diagnosis)
    }

    rendered = []
    for record in sorted(person_results, key=lambda item: item.get("person_id", "")):
        person = record.get("person", {})
        output = record.get("output", {})
        person_id = record.get("person_id") or person.get("id") or output.get("person_id", "")
        ledger = ledgers.get(person_id, {})
        shadows = shadows_by_person.get(person_id, [])
        chain_scan = chain_scans_by_person.get(person_id, {})
        stance = str(stance_by_person.get(person_id, "unknown"))
        issue_line = _one_line_issue(chain_scan, question, person, ledger, shadows, output)
        rendered.append(
            f"""<details class="person">
              <summary>
                <div>
                  <div class="summary-title">
                    <strong>{_e(person.get("name", person_id))}</strong>
                    <span class="pill">{_e(STANCE_LABELS.get(stance, stance))}</span>
                  </div>
                  <div class="conclusion">{_e(_short(output.get("conclusion", ""), 190))}</div>
                  <div class="shadow-line">它的阴影：{_e(issue_line)}</div>
                </div>
                <span class="pill mid">点击展开</span>
              </summary>
              <div class="person-body">
                <div class="person-grid">
                  {_identity_block(person)}
                  {_filter_block(person)}
                  {_search_block(ledger)}
                  {_decision_block(ledger)}
                </div>
                <div class="person-grid">
                  {_chain_block("核心假设", output.get("core_assumptions", []))}
                  {_chain_block("推理路径", output.get("reasoning_path", []), ordered=True)}
                </div>
                <div class="person-grid">
                  {_conclusion_block(output)}
                  {_shadow_problem_block(chain_scan, shadows, output)}
                </div>
              </div>
            </details>"""
        )
    return "\n".join(rendered)


def _identity_block(person: dict[str, Any]) -> str:
    values = person.get("values", {})
    top_values = ", ".join(f"{key}:{value}" for key, value in sorted(values.items(), key=lambda item: item[1], reverse=True)[:3])
    return f"""<div class="card">
      <h3>认知身份证</h3>
      <p>{_e(person.get("role_summary", ""))}</p>
      <p><strong>框架：</strong>{_e(", ".join(person.get("cognitive_frames", [])))}</p>
      <p><strong>强专业：</strong>{_e(", ".join(person.get("expertise_strong", [])))}</p>
      <p><strong>弱专业：</strong>{_e(", ".join(person.get("expertise_weak", [])))}</p>
      <p><strong>价值权重：</strong>{_e(top_values)}</p>
      <p><strong>隐藏偏见：</strong>{_e(", ".join(person.get("hidden_biases", [])))}</p>
    </div>"""


def _filter_block(person: dict[str, Any]) -> str:
    filt = person.get("information_filter", {})
    return f"""<div class="card">
      <h3>信息过滤器</h3>
      {_pills("它信什么", filt.get("trusted_sources", []), "good")}
      {_pills("它不信什么", filt.get("distrusted_sources", []), "bad")}
      {_pills("它偏好什么证据", filt.get("preferred_evidence", []), "good")}
      {_pills("它容易忽略什么", filt.get("ignored_evidence", []), "bad")}
    </div>"""


def _search_block(ledger: dict[str, Any]) -> str:
    strategy = ledger.get("search_strategy", {})
    tool_requests = strategy.get("search_tool_requests", []) or []
    if strategy.get("search_tool_mode") in {
        "llm_tool_calls",
        "deepseek_tool_calls",
    } and tool_requests:
        request_lines = [
            (
                f"查询：{item.get('query', '')}；为什么这样搜：{item.get('why_this_person_searches_it', '')}；"
                f"想找：{item.get('evidence_sought', '')}；实际来源：{', '.join(item.get('providers_attempted', [])) or '未记录'}；"
                f"返回 {item.get('result_count', 0)} 条"
            )
            for item in tool_requests[:6]
        ]
        return f"""<div class="card">
          <h3>它怎么上网</h3>
          <p><span class="pill good">模型实际工具调用</span></p>
          {_list(request_lines)}
        </div>"""
    return f"""<div class="card">
      <h3>它怎么上网</h3>
      <h4>查询关键词</h4>
      {_list(strategy.get("preferred_queries", [])[:6])}
      <h4>为什么这样搜</h4>
      {_list(strategy.get("query_intent", [])[:4])}
    </div>"""


def _decision_block(ledger: dict[str, Any]) -> str:
    accepted = ledger.get("accepted_sources", [])
    rejected = ledger.get("rejected_sources", [])
    ignored = ledger.get("ignored_sources", [])
    return f"""<div class="card">
      <h3>它怎样筛选来源</h3>
      <p><span class="pill good">采信 {len(accepted)}</span><span class="pill bad">拒绝 {len(rejected)}</span><span class="pill mid">忽略 {len(ignored)}</span></p>
      <h4>采信样本</h4>
      {_source_list(accepted[:3])}
      <h4>忽略/拒绝样本</h4>
      {_source_list((ignored + rejected)[:3])}
    </div>"""


def _conclusion_block(output: dict[str, Any]) -> str:
    return f"""<div class="card">
      <h3>结论与自我修正条件</h3>
      <p><strong>结论：</strong>{_e(output.get("conclusion", ""))}</p>
      <p><strong>自报置信：</strong>{_e(output.get("confidence", "NA"))}</p>
      <h4>它承认自己低估了</h4>
      {_list(output.get("what_i_underweighted", [])[:6])}
      <h4>什么会改变它的想法</h4>
      {_list(output.get("what_would_change_my_mind", [])[:6])}
    </div>"""


def _shadow_problem_block(
    chain_scan: dict[str, Any],
    shadows: list[dict[str, Any]],
    output: dict[str, Any],
) -> str:
    summary = (
        chain_scan.get("semantic_verdict", {}).get("summary")
        or chain_scan.get("plain_language_summary", {})
    )
    headline = str(summary.get("headline", "")).strip()
    problems = summary.get("key_problems", [])
    unresolved_count = len(summary.get("unresolved_checks", []))
    if problems:
        problem_html = "".join(
            f"""<div class="chain-issue">
              <p><strong>问题：</strong>{_e(item.get("problem", ""))}</p>
              <p><strong>它出现在哪里：</strong>{_e(item.get("where_it_appeared", ""))}</p>
              <p><strong>它怎样影响后面的判断：</strong>{_e(item.get("how_it_affected_thinking", ""))}</p>
              <p><strong>仍然可以保留：</strong>{_e(item.get("what_can_still_be_kept", ""))}</p>
            </div>"""
            for item in problems
        )
    else:
        problem_html = _list(_reasoning_shadow_items(shadows, output)[:3])
    unresolved_html = (
        f'<p class="muted small">另有 {unresolved_count} 项检查因材料不足而没有下判断。</p>'
        if unresolved_count
        else ""
    )
    return f"""<div class="card">
      <h3>元模型指出的问题</h3>
      <p class="chain-headline">{_e(headline or '现有材料还不足以指出明确问题。')}</p>
      {problem_html}
      {unresolved_html}
    </div>"""


def _vision_shadow_items(question: str, person: dict[str, Any], ledger: dict[str, Any]) -> list[str]:
    items = []
    display_question = question or "这个问题"
    all_decisions = ledger.get("source_decisions", [])
    layer_counts = _decision_layer_counter(all_decisions)
    accepted_types = _decision_type_counter(ledger.get("accepted_sources", []))
    ignored_types = _decision_type_counter(ledger.get("ignored_sources", []) + ledger.get("rejected_sources", []))
    strategy = ledger.get("search_strategy", {})
    actual_requests = strategy.get("search_tool_requests", []) or []
    query_values = (
        [item.get("query", "") for item in actual_requests]
        if strategy.get("search_tool_mode")
        in {"llm_tool_calls", "deepseek_tool_calls"}
        else strategy.get("preferred_queries", [])
    )
    focuses = _query_focuses(query_values, question)
    filt = person.get("information_filter", {})

    if layer_counts.get("model_prior"):
        external_count = sum(count for layer, count in layer_counts.items() if layer not in {"model_prior", "mock_search"})
        mock_count = layer_counts.get("mock_search", 0)
        tail = []
        if external_count:
            tail.append(f"外部搜索材料 {external_count} 条")
        if mock_count:
            tail.append(f"模拟搜索材料 {mock_count} 条")
        items.append(
            f"它先带着 {layer_counts['model_prior']} 条脑中先验入场；这些先验能暴露它默认相信什么、害怕什么、会先找什么，但不能当作外部事实。"
            f"{'同时还看到：' + '，'.join(tail) + '。' if tail else '这轮几乎是在先验层上暴露认知阴影。'}"
        )
    if focuses:
        items.append(f"它搜索时已经把问题推向这些方向：{_join_or(focuses[:3], '暂无')}。所以它不是在看完整的“{display_question}”，而是在看自己更会看的那一片。")
    if accepted_types:
        stories = _evidence_count_stories(accepted_types, question)
        first_dimension = accepted_types.most_common(1)[0][0]
        first_story = _evidence_gap_story(first_dimension, question)
        items.append(f"它真正放进推理的材料主要是：{_join_or(stories, '暂无')}。这会让它优先把问题理解成“{first_story['title']}”。")
    else:
        items.append("它几乎没有采信搜索结果，结论更可能是由身份预设、原有知识和提示里的材料推出来的。")
    if ignored_types:
        stories = _evidence_count_stories(ignored_types, question)
        items.append(
            f"它绕开或没认真吸收的是：{_join_or(stories, '暂无')}。"
            "这些材料是否与本题有关，必须先验证；若相关，再检查补进来后会推翻哪一个前提或改变哪一步推理。"
        )
    preferred = filt.get("preferred_evidence", [])
    ignored = filt.get("ignored_evidence", [])
    if preferred or ignored:
        items.append(f"它的信息过滤器一开始就偏好“{_join_or([str(v) for v in preferred[:3]], '未写明')}”，容易忽略“{_join_or([str(v) for v in ignored[:3]], '未写明')}”。也就是说，阴影在上网之前已经开始形成。")
    return list(dict.fromkeys(item for item in items if item))


def _identity_shadow_items(person: dict[str, Any]) -> list[str]:
    items = []
    role = person.get("role_summary", "")
    risk = person.get("risk_attitude", "")
    if role or risk:
        items.append(f"它的角色出发点是“{role or '未写明'}”，风险姿态是“{risk or '未写明'}”。同一组事实经过这个身份，会更容易被解释成：{_risk_shadow(risk)}")
    biases = [f"{bias}：{_bias_plain(bias)}" for bias in person.get("hidden_biases", [])[:4]]
    if biases:
        items.append("它的隐藏偏见会这样牵引注意力：" + "；".join(biases) + "。")
    strong = person.get("expertise_strong", [])
    weak = person.get("expertise_weak", [])
    if strong or weak:
        items.append(f"它强在“{_join_or([str(v) for v in strong[:3]], '未写明')}”，弱在“{_join_or([str(v) for v in weak[:3]], '未写明')}”。所以它会把熟悉领域看得更清楚，把不熟领域压成粗略背景。")
    values = person.get("values", {})
    if values:
        top = sorted(values.items(), key=lambda item: item[1], reverse=True)[:2]
        items.append("它最看重 " + "、".join(f"{key}({value})" for key, value in top) + "。价值排序会影响它把“更危险”还是“更有机会”放在判断前面。")
    return list(dict.fromkeys(item for item in items if item))


def _reasoning_shadow_items(shadows: list[dict[str, Any]], output: dict[str, Any]) -> list[str]:
    items = []
    assumptions = output.get("core_assumptions", []) or []
    if assumptions:
        items.append(f"它把“{_short(assumptions[0], 96)}”当成推理支点。一旦这个支点不成立，后面对当前问题的结论就会明显松动。")
    for shadow in shadows[:4]:
        description = str(shadow.get("description", "")).strip()
        if description:
            items.append(f"{_stage_plain(shadow.get('source_stage', ''))}：{description}。这一步可能把某类证据放大，或把另一类证据提前排除。")
    for item in output.get("what_i_underweighted", [])[:3] or []:
        items.append(f"它自己也承认没看够：{item}。读它的结论时，这部分要作为折扣。")
    if not items:
        items.append("这条输出没有留下足够清楚的推理链，元模型只能先把它当作不完整样本，而不是稳定判断。")
    return list(dict.fromkeys(item for item in items if item))


def _shadow_use_items(puzzle_items: list[dict[str, Any]]) -> list[str]:
    items = []
    for item in puzzle_items[:5]:
        items.append(_shadow_use_plain(item))
    if not items:
        items.append("先把它当作一个局部视角：能提示风险，但不能单独当作问题答案。")
    return list(dict.fromkeys(item for item in items if item))


def _chain_block(title: str, values: list[Any], *, ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    return f"""<div class="card">
      <h3>{_e(title)}</h3>
      <{tag}>{''.join(f'<li>{_e(value)}</li>' for value in values[:8])}</{tag}>
    </div>"""


def _reason_clusters(person_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for label, tokens in REASON_PATTERNS:
        people = []
        examples = []
        for record in person_results:
            output = record.get("output", {})
            fragment = _matching_fragment(output, tokens)
            if fragment:
                people.append(record.get("person_id", ""))
                name = record.get("person", {}).get("name") or record.get("person_name") or record.get("person_id", "")
                examples.append(f"{name}：{_short(fragment, 110)}")
        if people:
            buckets.append({"label": label, "count": len(set(people)), "examples": list(dict.fromkeys(examples))})
    return sorted(buckets, key=lambda item: item["count"], reverse=True)


def _matching_fragment(output: dict[str, Any], tokens: list[str]) -> str:
    candidates = []
    candidates.extend(str(item) for item in output.get("core_assumptions", []) or [])
    candidates.extend(str(item) for item in output.get("reasoning_path", []) or [])
    if output.get("conclusion"):
        candidates.append(str(output.get("conclusion", "")))
    for candidate in candidates:
        if any(token in candidate for token in tokens):
            return candidate
    return ""


def _person_issue_items(
    question: str,
    person: dict[str, Any],
    ledger: dict[str, Any],
    shadows: list[dict[str, Any]],
    output: dict[str, Any],
) -> list[str]:
    issues = []
    issues.extend(_vision_shadow_items(question, person, ledger)[:2])
    issues.extend(_identity_shadow_items(person)[:1])
    issues.extend(_reasoning_shadow_items(shadows, output)[:2])
    return list(dict.fromkeys(issue for issue in issues if issue))


def _one_line_issue(
    chain_scan: dict[str, Any],
    question: str,
    person: dict[str, Any],
    ledger: dict[str, Any],
    shadows: list[dict[str, Any]],
    output: dict[str, Any],
) -> str:
    summary = (
        chain_scan.get("semantic_verdict", {}).get("summary")
        or chain_scan.get("plain_language_summary", {})
    )
    headline = str(summary.get("headline", "")).strip()
    if headline:
        return _short(headline, 150)
    issues = _person_issue_items(question, person, ledger, shadows, output)
    return _short(issues[0], 130) if issues else "暂未看到明确阴影，需要补充证据账本。"


def _top_underweighted(person_results: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for record in person_results:
        for item in record.get("output", {}).get("what_i_underweighted", []) or []:
            key = str(item).split("：")[0].split(":")[0].strip()
            if key:
                counter[key] += 1
    return counter.most_common(6)


def _decision_types(decisions: list[dict[str, Any]]) -> list[str]:
    counter = _decision_type_counter(decisions)
    return [f"{key}({count})" for key, count in counter.most_common()]


def _decision_type_counter(decisions: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for decision in decisions:
        result = decision.get("result", {})
        value = result.get("evidence_type") or result.get("source_type") or "未知证据"
        counter[str(value)] += 1
    return counter


def _decision_layer_counter(decisions: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for decision in decisions:
        result = decision.get("result", {})
        value = result.get("retrieval_layer") or "external_search"
        counter[str(value)] += 1
    return counter


def _evidence_count_stories(counter: Counter[str], question: str, limit: int = 3) -> list[str]:
    stories = []
    for dimension, count in counter.most_common(limit):
        short = _evidence_gap_story(dimension, question)["short"]
        stories.append(f"{short}（{count}条）")
    return stories


def _query_focuses(queries: list[Any], question: str) -> list[str]:
    focuses = []
    for query in queries[:4]:
        text = str(query)
        cleaned = text.replace(question, "").strip(" -_—:：，,|")
        if cleaned:
            focuses.append(_short(cleaned, 52))
    return list(dict.fromkeys(focuses))


def _risk_shadow(risk: Any) -> str:
    text = str(risk)
    if any(token in text for token in ["尾部", "规避", "谨慎", "保守", "悲观"]):
        return "先看到坏情景、失效点和不可逆后果，低估适应、缓冲、修复或条件改善的可能。"
    if any(token in text for token in ["乐观", "追求", "冒险", "进取"]):
        return "先看到收益、增长和试错空间，低估失败成本、约束收紧或负面反馈累积的速度。"
    if "中性" in text or "平衡" in text:
        return "倾向把风险和机会摊平，可能低估少数极端情形对整体判断的冲击。"
    return "更符合自己职业经验和价值偏好的那种故事。"


def _bias_plain(bias: Any) -> str:
    return {
        "可得性启发": "最近、最鲜明、最容易想起的案例会被它看得过重。",
        "正常化偏误": "异常信号持续存在时，它可能把异常当成新常态。",
        "确认偏误": "它更容易寻找能支持原方向的材料，较少主动撞反例。",
        "基本归因错误": "它可能把复杂结构问题归因到某一类人、公司或动机上。",
        "群体内偏爱": "它会更信任与自己阵营、行业或价值观相近的声音。",
        "复杂化偏误": "它可能把较直接的问题解释得过度复杂。",
        "锚定效应": "第一批数字、案例或历史类比会持续拉住后面的判断。",
        "损失厌恶": "它会把潜在损失看得比同等收益更重。",
        "代表性启发": "它会把眼前问题塞进一个熟悉模板，并忽略两者之间决定性的结构差异。",
        "过度自信": "它可能把局部专业经验误当成对整体问题的把握。",
    }.get(str(bias), "它会稳定地影响注意力分配和解释方向。")


def _stage_plain(value: Any) -> str:
    return {
        "information_filter": "看材料之前",
        "evidence_selection": "筛材料时",
        "assumption": "立前提时",
        "reasoning": "从证据推到结论时",
        "value_evaluation": "决定什么更重要时",
        "conclusion": "下结论时",
    }.get(str(value), "某一步")


def _shadow_use_plain(item: dict[str, Any]) -> str:
    use = item.get("puzzle_use", "")
    direction = str(item.get("estimated_bias_vector", {}).get("direction", "")).strip()
    tail = f"：{direction}" if direction else ""
    if use == "negative_space":
        return f"把它当成“哪里没被看见”的提醒{tail}。下一步要补具体事实，而不是只听它的结论。"
    if use == "constraint_band":
        return f"它摸到的方向可以保留，但幅度要收窄{tail}。也就是：方向有参考价值，强度不能全信。"
    if use == "downweight":
        return f"这部分结论先少信一点{tail}。原因通常不是它没价值，而是材料太偏或推理太快。"
    if use == "fracture_boundary":
        return f"这里提示一个推理跳步{tail}。从已看到的材料到最后结论，中间还缺能把因果链扣紧的证据。"
    if use == "anchor":
        return f"如果其他身份从不同材料也摸到同一方向，这条可以升级为共同线索{tail}。现在先记住，不急着当答案。"
    return f"把这条阴影当成读法提醒{tail}。它帮助我们知道这份结论该怎么用、该打多少折。"


def _source_decision_counts(retrieval: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for ledger in retrieval.get("evidence_ledgers", []):
        for decision in ledger.get("source_decisions", []):
            counter[str(decision.get("decision", "unknown"))] += 1
    return counter


def _stance_bars(clusters: dict[str, list[str]]) -> str:
    total = max(1, sum(len(ids) for ids in clusters.values()))
    rows = []
    for stance, ids in sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True):
        count = len(ids)
        pct = round(count / total * 100, 1)
        rows.append(
            f"""<div class="bar-row {_css_class(stance)}">
              <div>{_e(STANCE_LABELS.get(stance, stance))}</div>
              <div class="track"><div class="fill" style="width:{pct}%"></div></div>
              <div>{count}</div>
            </div>"""
        )
    return "\n".join(rows)


def _probe_block(title: str, values: list[Any]) -> str:
    return f"""<div>
      <h3>{_e(title)}</h3>
      {_list(values[:8])}
    </div>"""


def _mini_block(title: str, text: str) -> str:
    return f"""<div>
      <h3>{_e(title)}</h3>
      <p>{_e(text)}</p>
    </div>"""


def _metric(label: str, value: str) -> str:
    return f'<div class="card"><div class="metric-label">{_e(label)}</div><div class="metric-value">{_e(value)}</div></div>'


def _pills(label: str, values: list[Any], kind: str) -> str:
    pills = "".join(f'<span class="pill {kind}">{_e(value)}</span>' for value in values)
    return f"<p><strong>{_e(label)}：</strong>{pills}</p>"


def _source_list(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return '<p class="muted small">暂无。</p>'
    lines = []
    for decision in decisions:
        result = decision.get("result", {})
        reasons = "; ".join(str(reason) for reason in decision.get("reasons", []))
        title = result.get("title") or result.get("url") or "unknown"
        url = result.get("url", "")
        layer = _layer_label(result.get("retrieval_layer", "external_search"))
        status = _verification_label(result.get("verification_status", "external_unverified"))
        lines.append(
            f"""<li>
              <a href="{_e(url)}">{_e(title)}</a>
              <div class="muted small">{_e(layer)} / {_e(status)} · {_e(result.get("source_type", ""))} / {_e(result.get("evidence_type", ""))} · {_e(reasons)}</div>
            </li>"""
        )
    return "<ul>" + "".join(lines) + "</ul>"


def _layer_label(value: Any) -> str:
    return {
        "model_prior": "模型先验",
        "external_search": "外部搜索",
        "mock_search": "模拟搜索",
    }.get(str(value), str(value or "外部搜索"))


def _verification_label(value: Any) -> str:
    return {
        "unverified_model_prior": "未外部核验",
        "external_unverified": "外部来源未复核",
        "synthetic_fixture": "流程测试材料",
    }.get(str(value), str(value or "未复核"))


def _list(values: list[Any]) -> str:
    if not values:
        return '<p class="muted small">暂无。</p>'
    return "<ul>" + "".join(f"<li>{_e(value)}</li>" for value in values) + "</ul>"


def _stance_clusters(diagnosis: dict[str, Any]) -> dict[str, list[str]]:
    return (
        diagnosis.get("disagreement_structure", {})
        .get("output_based", {})
        .get("stance_clusters", {})
    )


def _conclusion_samples(diagnosis: dict[str, Any]) -> list[dict[str, Any]]:
    return (
        diagnosis.get("disagreement_structure", {})
        .get("output_based", {})
        .get("conclusion_samples", [])
    )


def _profile_people(summary: dict[str, Any]) -> int:
    return int(summary.get("profile", {}).get("person_count") or 24)


def _read_person_outputs(people_dir: Path) -> list[dict[str, Any]]:
    if not people_dir.exists():
        return []
    return [_read_json(path) for path in sorted(people_dir.glob("*.json"))]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _num(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _stage_label(value: str) -> str:
    return {
        "information_filter": "信息入口",
        "evidence_selection": "证据选择",
        "assumption": "假设形成",
        "reasoning": "推理链",
        "value_evaluation": "价值判断",
        "conclusion": "结论输出",
    }.get(value, value)


def _puzzle_use_label(value: str) -> str:
    return {
        "negative_space": "作为缺口提醒",
        "constraint_band": "作为偏差约束",
        "downweight": "降低这部分判断权重",
        "fracture_boundary": "标出推理断裂处",
        "anchor": "作为候选锚点",
    }.get(value, value or "拼图材料")


def _join_or(values: list[str], fallback: str) -> str:
    return "；".join(values) if values else fallback


def _css_class(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)
