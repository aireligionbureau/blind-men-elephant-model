# Roadmap

## v0.1: Protocol Skeleton

- Define the model manifesto.
- Define digital person schema.
- Define evidence ledger schema.
- Define shadow schema.
- Define truth contour schema.
- Provide prompt protocols for digital persons and meta-model.
- Provide a zero-dependency Python run-plan prototype.

## v0.2: LLM Adapter

- Add model provider abstraction. Done: OpenAI-compatible client with DeepSeek defaults.
- Add digital person prompt rendering. Done.
- Add structured output parsing. Done: JSON first, raw fallback.
- Add standard/deep profiles. Done: 24 and 50 digital persons.
- Add retry and validation strategy. Done: bounded retries, terminal-error classification, checkpoints, and schema gates.
- Preserve each digital person's cognitive identity across calls. Done: the full nine-dimensional identity and filter travel with retrieval and reasoning.

## v0.3: Filtered Search

- Add search provider abstraction. Done: `SearchProvider`.
- Let each digital person generate its own query plan. Done.
- Store accepted, rejected, and ignored sources. Done.
- Make information filtering auditable in the evidence ledger. Done.
- Add real search API provider. Done: Google Custom Search, Baidu JSON bridge, experimental Baidu HTML.
- Add source fetching and excerpt extraction.

## v0.4: Classic Lens Knowledge Base

- Convert classic works into diagnostic lens cards. Done: 9 expanded lens cards with mechanism chains, boundaries, anti-misuse rules, and model training notes.
- Deepen classic lenses into micro-lenses. Done: 40 micro-lenses.
- Add differential diagnosis between confusing lenses. Done: 10 conflict rules.
- Add first calibration cases. Done: 5 calibration cases.
- Add shadow mechanism ontology. Done: 18 shadow generation mechanisms.
- Add full-chain capture protocol. Done: 10 stages from problem framing to meta-audit.
- Add 90-point readiness evaluator. Done: `readiness_evaluation`.
- Add lens invocation protocol. Done: `LensCard` registry and loader.
- Add bias/noise/paradigm/argument-forensics diagnostics. Done: deterministic first-pass diagnosis engine.
- Track which lens produced which shadow attribution. Done: `bias_attribution` and `inversion_plan`.
- Add LLM-assisted semantic adjudication over the same first-pass cohort. Done; this does not add another epistemic cohort round.

## v0.5: Shadow Inversion Engine

- Convert shadows into anchors, constraint bands, negative spaces, and fracture boundaries. Done: `build_shadow_puzzle`.
- Add confidence and uncertainty annotations. Done: first-pass material confidence.
- Add conflict classification. Done: content and shadow relations are separately proposed and adjudicated.
- Generate candidate truth contours. Done: rival provisional contours remain hypotheses until final adjudication.

## v0.5a: Standard Live Batch

- Run standard 24-person live batch. Done: `run-standard-live`.
- Add concurrency and retries. Done.
- Save retrieval, person outputs, diagnosis, puzzle, summary, and full run JSON. Done.
- Add token and configurable cost summary. Done.
- Feed digital person outputs plus evidence ledgers into first-pass meta diagnosis. Done.

## v0.6: One-Pass Blind-Spot Planning

- Classify mixed question types and proof obligations before cohort generation. Done.
- Cover evidence, definitions, time horizons, counterexamples, and affected parties in the initial cohort. Done.
- Measure nine-dimensional identity spread without another model call. Done.
- Keep automatic epistemic second rounds disabled; unresolved gaps become explicit unknowns. Done.

## v0.7: Visualization

- Visualize the truth-contour assembly process. Done: 24 observers hand off scattered pieces that assemble with the live pipeline state.
- Show anchors, constraint bands, negative spaces, and disagreement clusters.
- Expose evidence lineage for every contour statement. Done: natural paragraphs in the page, sentence-level bindings in the audit artifact.
