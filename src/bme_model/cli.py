from __future__ import annotations

import argparse
import json
import sys

from .batch import (
    enrich_saved_run_with_semantic_verdicts,
    enrich_saved_run_with_truth_contour,
    rebuild_saved_run,
    resume_live_batch,
    run_live_batch,
)
from .config import DEFAULT_COHORT_MODEL, DEFAULT_MODEL, RUN_PROFILES
from .report import build_run_report
from .pipeline import (
    PipelineIncompleteError,
    read_run_status,
    recover_incomplete_runs,
)
from .model import build_run_plan
from .runner import (
    build_profile_plan,
    _get_search_provider,
    run_filtered_retrieval,
    run_one_person_live,
    run_shadow_diagnosis,
)


RETRIEVAL_PROVIDER_CHOICES = [
    "mock",
    "google",
    "google-news-rss",
    "baidu-html",
    "baidu-json",
    "bing-html",
    "bing-rss",
    "duckduckgo-html",
    "federated-html",
    "auto-cn",
    "auto-global",
    "model-prior",
    "deepseek-prior",
    "hybrid",
    "hybrid-mock",
    "hybrid-global",
    "hybrid-cn",
    "hybrid-google",
    "hybrid-baidu-json",
    "hybrid-baidu-html",
    "hybrid-federated",
    "three-layer",
    "agentic-federated",
]

SEARCH_TEST_PROVIDER_CHOICES = [
    "mock",
    "google",
    "google-news-rss",
    "baidu-html",
    "baidu-json",
    "bing-html",
    "bing-rss",
    "duckduckgo-html",
    "federated-html",
    "auto-cn",
    "auto-global",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the Blind Men Elephant Model.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Generate a dry-run protocol plan.")
    plan_parser.add_argument("question", help="The complex question to analyze.")
    plan_parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default=None,
        help="Run profile. standard=24 persons, deep=50 persons.",
    )
    plan_parser.add_argument("--person-count", type=int, default=4, help="Override number of starter digital persons.")

    live_person_parser = subparsers.add_parser("run-person", help="Run one digital person through DeepSeek.")
    live_person_parser.add_argument("question", help="The complex question to analyze.")
    live_person_parser.add_argument("person_id", help="Digital person id, e.g. crisis_engineer.")
    live_person_parser.add_argument("--model", default=DEFAULT_MODEL)
    live_person_parser.add_argument("--max-tokens", type=int, default=8192)

    retrieve_parser = subparsers.add_parser("retrieve", help="Run filtered retrieval and build evidence ledgers.")
    retrieve_parser.add_argument("question", help="The complex question to analyze.")
    retrieve_parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default=None,
        help="Run profile. standard=24 persons, deep=50 persons.",
    )
    retrieve_parser.add_argument("--person-count", type=int, default=4)
    retrieve_parser.add_argument(
        "--provider",
        default="mock",
        choices=RETRIEVAL_PROVIDER_CHOICES,
    )
    retrieve_parser.add_argument("--results-per-query", type=int, default=5)

    diagnose_parser = subparsers.add_parser("diagnose", help="Run retrieval and diagnose cognitive shadows.")
    diagnose_parser.add_argument("question", help="The complex question to analyze.")
    diagnose_parser.add_argument(
        "--profile",
        choices=sorted(RUN_PROFILES),
        default=None,
        help="Run profile. standard=24 persons, deep=50 persons.",
    )
    diagnose_parser.add_argument("--person-count", type=int, default=4)
    diagnose_parser.add_argument(
        "--provider",
        default="mock",
        choices=RETRIEVAL_PROVIDER_CHOICES,
    )
    diagnose_parser.add_argument("--results-per-query", type=int, default=5)

    batch_parser = subparsers.add_parser(
        "run-standard-live",
        help="Run the standard 24-person live batch: retrieval, LLM calls, diagnosis, puzzle, persistence.",
    )
    batch_parser.add_argument("question", help="The complex question to analyze.")
    batch_parser.add_argument(
        "--provider",
        default="three-layer",
        choices=RETRIEVAL_PROVIDER_CHOICES,
    )
    batch_parser.add_argument("--results-per-query", type=int, default=5)
    batch_parser.add_argument("--model", default=DEFAULT_MODEL)
    batch_parser.add_argument("--cohort-model", default=DEFAULT_COHORT_MODEL)
    batch_parser.add_argument("--max-tokens", type=int, default=8192)
    batch_parser.add_argument("--concurrency", type=int, default=24)
    batch_parser.add_argument(
        "--retrieval-concurrency",
        type=int,
        default=None,
        help="Concurrent retrieval jobs. Streaming mode defaults to min(concurrency, 12).",
    )
    batch_parser.add_argument("--semantic-concurrency", type=int, default=None)
    batch_parser.add_argument("--model-concurrency", type=int, default=None)
    batch_parser.add_argument(
        "--pipeline-mode",
        choices=["streaming_v2", "sequential"],
        default="sequential",
        help="Use the streaming pipeline or the retained comparison baseline.",
    )
    batch_parser.add_argument(
        "--synthesis-mode",
        choices=["optimized_v2", "legacy_sequential"],
        default="legacy_sequential",
        help="Use optimized detective/contour scheduling or the retained baseline.",
    )
    batch_parser.add_argument("--retries", type=int, default=1)
    batch_parser.add_argument(
        "--recovery-passes",
        type=int,
        default=0,
        help="Targeted recovery sweeps after ordinary per-call retries.",
    )
    batch_parser.add_argument(
        "--time-budget-seconds",
        type=int,
        default=570,
        help="End-to-end analysis budget before bounded recovery stops.",
    )
    batch_parser.add_argument("--output-dir", default=None)
    batch_parser.add_argument("--input-price-per-1m-cny", type=float, default=None)
    batch_parser.add_argument("--output-price-per-1m-cny", type=float, default=None)
    batch_parser.add_argument("--semantic-max-tokens", type=int, default=8192)
    batch_parser.add_argument("--relational-max-tokens", type=int, default=8192)
    batch_parser.add_argument(
        "--skip-semantic-verdicts",
        action="store_true",
        help="Skip the per-person DeepSeek semantic verdict pass.",
    )
    batch_parser.add_argument(
        "--skip-relational-synthesis",
        action="store_true",
        help="Skip live detective/contour calls and write a conservative deterministic fallback.",
    )

    resume_parser = subparsers.add_parser(
        "resume-run",
        help="Resume only missing, invalid, or operationally degraded stages in a saved run.",
    )
    resume_parser.add_argument("run_dir", help="Existing run directory.")
    resume_parser.add_argument("--concurrency", type=int, default=None)
    resume_parser.add_argument("--retrieval-concurrency", type=int, default=None)
    resume_parser.add_argument("--semantic-concurrency", type=int, default=None)
    resume_parser.add_argument("--model-concurrency", type=int, default=None)
    resume_parser.add_argument(
        "--pipeline-mode",
        choices=["streaming_v2", "sequential"],
        default=None,
    )
    resume_parser.add_argument(
        "--synthesis-mode",
        choices=["optimized_v2", "legacy_sequential"],
        default=None,
    )
    resume_parser.add_argument("--retries", type=int, default=None)
    resume_parser.add_argument("--recovery-passes", type=int, default=None)
    resume_parser.add_argument("--time-budget-seconds", type=int, default=None)

    status_parser = subparsers.add_parser(
        "run-status",
        help="Print checkpoint, progress, and quality state for a saved run.",
    )
    status_parser.add_argument("run_dir", help="Existing run directory.")

    recover_parser = subparsers.add_parser(
        "recover-runs",
        help="Resume abandoned recoverable runs, intended for service startup.",
    )
    recover_parser.add_argument("--output-dir", default="runs")
    recover_parser.add_argument("--limit", type=int, default=10)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Serve the homepage and persistent background run API.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--runs-dir", default="runs")
    serve_parser.add_argument("--homepage-dir", default=None)
    serve_parser.add_argument("--provider", default="three-layer", choices=RETRIEVAL_PROVIDER_CHOICES)
    serve_parser.add_argument("--model", default=None)
    serve_parser.add_argument("--cohort-model", default=None)
    serve_parser.add_argument("--run-workers", type=int, default=1)
    serve_parser.add_argument(
        "--pipeline-mode",
        choices=["streaming_v2", "sequential"],
        default="sequential",
    )
    serve_parser.add_argument(
        "--synthesis-mode",
        choices=["optimized_v2", "legacy_sequential"],
        default="legacy_sequential",
    )
    serve_parser.add_argument("--no-auto-recover", action="store_true")

    rebuild_parser = subparsers.add_parser(
        "rebuild-run",
        help="Rebuild diagnosis, shadow puzzle, summary, and run.json from an existing saved run.",
    )
    rebuild_parser.add_argument("run_dir", help="Existing run directory.")
    rebuild_parser.add_argument("--input-price-per-1m-cny", type=float, default=None)
    rebuild_parser.add_argument("--output-price-per-1m-cny", type=float, default=None)

    verdict_parser = subparsers.add_parser(
        "enrich-verdicts",
        help="Add evidence-locked per-person semantic verdicts to an existing saved run.",
    )
    verdict_parser.add_argument("run_dir", help="Existing run directory.")
    verdict_parser.add_argument("--model", default=DEFAULT_MODEL)
    verdict_parser.add_argument("--max-tokens", type=int, default=8192)
    verdict_parser.add_argument("--concurrency", type=int, default=6)
    verdict_parser.add_argument("--retries", type=int, default=2)
    verdict_parser.add_argument("--force", action="store_true")
    verdict_parser.add_argument("--input-price-per-1m-cny", type=float, default=None)
    verdict_parser.add_argument("--output-price-per-1m-cny", type=float, default=None)

    contour_parser = subparsers.add_parser(
        "enrich-contour",
        help="Run the live detective and truth-contour layers for an existing saved run.",
    )
    contour_parser.add_argument("run_dir", help="Existing run directory.")
    contour_parser.add_argument("--model", default=DEFAULT_MODEL)
    contour_parser.add_argument("--max-tokens", type=int, default=8192)
    contour_parser.add_argument("--retries", type=int, default=2)
    contour_parser.add_argument("--input-price-per-1m-cny", type=float, default=None)
    contour_parser.add_argument("--output-price-per-1m-cny", type=float, default=None)

    report_parser = subparsers.add_parser(
        "build-report",
        help="Build a readable static HTML report for an existing run.",
    )
    report_parser.add_argument("run_dir", help="Existing run directory.")
    report_parser.add_argument("--output", default=None, help="Output HTML path. Defaults to report.html in run_dir.")

    search_parser = subparsers.add_parser(
        "test-search",
        help="Test a configured search provider and print normalized SearchResult items.",
    )
    search_parser.add_argument("query", help="Search query.")
    search_parser.add_argument(
        "--provider",
        default="auto-global",
        choices=SEARCH_TEST_PROVIDER_CHOICES,
    )
    search_parser.add_argument("--limit", type=int, default=3)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "plan":
        if args.profile:
            result = build_profile_plan(args.question, args.profile)
        else:
            result = build_run_plan(args.question, person_count=args.person_count)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "run-person":
        result = run_one_person_live(
            args.question,
            args.person_id,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "retrieve":
        result = run_filtered_retrieval(
            args.question,
            profile_name=args.profile,
            person_count=args.person_count,
            provider_name=args.provider,
            results_per_query=args.results_per_query,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "diagnose":
        result = run_shadow_diagnosis(
            args.question,
            profile_name=args.profile,
            person_count=args.person_count,
            provider_name=args.provider,
            results_per_query=args.results_per_query,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "run-standard-live":
        try:
            result = run_live_batch(
                args.question,
                profile_name="standard",
                provider_name=args.provider,
                results_per_query=args.results_per_query,
                model=args.model,
                cohort_model=args.cohort_model,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                retrieval_concurrency=args.retrieval_concurrency,
                semantic_concurrency=args.semantic_concurrency,
                model_concurrency=args.model_concurrency,
                pipeline_mode=args.pipeline_mode,
                synthesis_mode=args.synthesis_mode,
                retries=args.retries,
                recovery_passes=args.recovery_passes,
                time_budget_seconds=args.time_budget_seconds,
                output_dir=args.output_dir,
                input_price_per_1m_cny=args.input_price_per_1m_cny,
                output_price_per_1m_cny=args.output_price_per_1m_cny,
                semantic_verdicts=not args.skip_semantic_verdicts,
                semantic_max_tokens=args.semantic_max_tokens,
                relational_synthesis=not args.skip_relational_synthesis,
                relational_max_tokens=args.relational_max_tokens,
            )
        except PipelineIncompleteError as exc:
            print(
                json.dumps(
                    {
                        "status": "recoverable_failed",
                        "stage": exc.stage,
                        "error": str(exc.original_error),
                        "run_dir": str(exc.run_dir),
                        "resume_command": f"bme-model resume-run \"{exc.run_dir}\"",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.get("combined_usage", result["usage"]), ensure_ascii=False, indent=2))
        print(json.dumps({"run_dir": result["run_dir"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "resume-run":
        try:
            result = resume_live_batch(
                args.run_dir,
                concurrency=args.concurrency,
                retrieval_concurrency=args.retrieval_concurrency,
                semantic_concurrency=args.semantic_concurrency,
                model_concurrency=args.model_concurrency,
                pipeline_mode=args.pipeline_mode,
                synthesis_mode=args.synthesis_mode,
                retries=args.retries,
                recovery_passes=args.recovery_passes,
                time_budget_seconds=args.time_budget_seconds,
            )
        except PipelineIncompleteError as exc:
            print(
                json.dumps(
                    {
                        "status": "recoverable_failed",
                        "stage": exc.stage,
                        "error": str(exc.original_error),
                        "run_dir": str(exc.run_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise SystemExit(2) from exc
        print(json.dumps(result.get("combined_usage", {}), ensure_ascii=False, indent=2))
        print(json.dumps({"run_dir": result["run_dir"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "run-status":
        print(json.dumps(read_run_status(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "recover-runs":
        print(
            json.dumps(
                recover_incomplete_runs(args.output_dir, limit=args.limit),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "serve":
        from .service import serve

        serve(
            host=args.host,
            port=args.port,
            runs_dir=args.runs_dir,
            homepage_dir=args.homepage_dir,
            provider=args.provider,
            model=args.model,
            cohort_model=args.cohort_model,
            run_workers=args.run_workers,
            pipeline_mode=args.pipeline_mode,
            synthesis_mode=args.synthesis_mode,
            auto_recover=not args.no_auto_recover,
        )
        return

    if args.command == "enrich-contour":
        result = enrich_saved_run_with_truth_contour(
            args.run_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            retries=args.retries,
            input_price_per_1m_cny=args.input_price_per_1m_cny,
            output_price_per_1m_cny=args.output_price_per_1m_cny,
        )
        print(json.dumps(result.get("truth_contour", {}), ensure_ascii=False, indent=2))
        print(json.dumps(result.get("relational_synthesis_usage", {}), ensure_ascii=False, indent=2))
        print(json.dumps({"run_dir": result["run_dir"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "enrich-verdicts":
        result = enrich_saved_run_with_semantic_verdicts(
            args.run_dir,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            retries=args.retries,
            force=args.force,
            input_price_per_1m_cny=args.input_price_per_1m_cny,
            output_price_per_1m_cny=args.output_price_per_1m_cny,
        )
        print(json.dumps(result["meta_model"].get("semantic_verdict_evaluation", {}), ensure_ascii=False, indent=2))
        print(json.dumps(result.get("combined_usage", {}), ensure_ascii=False, indent=2))
        print(json.dumps({"run_dir": result["run_dir"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "rebuild-run":
        result = rebuild_saved_run(
            args.run_dir,
            input_price_per_1m_cny=args.input_price_per_1m_cny,
            output_price_per_1m_cny=args.output_price_per_1m_cny,
        )
        print(json.dumps(result["usage"], ensure_ascii=False, indent=2))
        print(json.dumps({"run_dir": result["run_dir"], "rebuilt_at": result["rebuilt_at"]}, ensure_ascii=False, indent=2))
        return

    if args.command == "build-report":
        output = build_run_report(args.run_dir, output_path=args.output)
        print(json.dumps({"report": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "test-search":
        provider = _get_search_provider(args.provider)
        results = provider.search(args.query, limit=args.limit)
        print(
            json.dumps(
                {
                    "provider": provider.name,
                    "query": args.query,
                    "count": len(results),
                    "results": [result.__dict__ for result in results],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
