"""Run evaluation against the IndiaVC research API.

Usage:
    python scripts/evaluate_agents.py
    python scripts/evaluate_agents.py --dataset data/eval/v1_companies.json --api http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from config.settings import settings

LLM_JUDGE_PROMPT = """You are evaluating an automated investor research brief for quality.

Brief content:
{brief}

Original query: {query}

Score the brief 0-10 on each dimension:
- COMPLETENESS: Does the brief have all expected sections? (Company Overview, Funding, Competitors, Risks, Bull Case, Bear Case, Sources)
- SPECIFICITY: Are claims specific (real numbers, dates, names) vs vague?
- BALANCE: Does it present both bull AND bear cases honestly?
- CITATIONS: Are sources cited? Are they real (URLs that look legitimate)?
- ACCURACY: Based on what you know, does the content appear factual?

Respond with ONLY a JSON object:
{{
  "completeness": <0-10>,
  "specificity": <0-10>,
  "balance": <0-10>,
  "citations": <0-10>,
  "accuracy": <0-10>,
  "overall": <0-10>,
  "rationale": "<one sentence>"
}}"""


async def run_query(
    client: httpx.AsyncClient, api_url: str, query: str
) -> dict[str, Any]:
    """Run a single research query via SSE stream + auto-approve."""
    start = time.time()
    try:
        session_id: str | None = None

        async with client.stream(
            "POST",
            f"{api_url}/research/stream",
            json={"question": query},
            timeout=300.0,
        ) as response:
            if response.status_code != 200:
                return {
                    "query": query,
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "duration_seconds": time.time() - start,
                }

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                evt = event.get("event")
                if evt == "started":
                    session_id = event.get("session_id")
                elif evt == "refused":
                    return {
                        "query": query,
                        "refused": True,
                        "reason": event.get("reason"),
                        "duration_seconds": time.time() - start,
                    }
                elif evt == "awaiting_approval":
                    break

        if not session_id:
            return {
                "query": query,
                "success": False,
                "error": "No session_id received",
                "duration_seconds": time.time() - start,
            }

        # Auto-approve to get the final brief
        approval_resp = await client.post(
            f"{api_url}/research/approve",
            json={"session_id": session_id},
            timeout=180.0,
        )

        if approval_resp.status_code != 200:
            return {
                "query": query,
                "success": False,
                "error": f"Approval HTTP {approval_resp.status_code}",
                "duration_seconds": time.time() - start,
            }

        data = approval_resp.json()
        duration = time.time() - start

        total_calls = sum(
            o.get("tool_calls", 0)
            for o in data.get("agent_outputs", {}).values()
        )

        return {
            "query": query,
            "success": True,
            "brief": data["brief"],
            "agent_outputs": data.get("agent_outputs", {}),
            "total_tool_calls": total_calls,
            "duration_seconds": duration,
        }

    except Exception as exc:
        return {
            "query": query,
            "success": False,
            "error": str(exc),
            "duration_seconds": time.time() - start,
        }


def score_brief(openai_client: OpenAI, query: str, brief: str) -> dict[str, Any]:
    """Use gpt-4o-mini as judge to score the brief (cheaper than gpt-4o)."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research quality evaluator."},
            {
                "role": "user",
                "content": LLM_JUDGE_PROMPT.format(query=query, brief=brief[:4000]),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def check_structural_requirements(
    brief: str, test_case: dict[str, Any]
) -> dict[str, Any]:
    """Mechanical checks: sections present, expected phrases found."""
    brief_lower = brief.lower()

    expected_phrases = test_case.get("expected_in_brief", [])
    phrases_found = [p for p in expected_phrases if p.lower() in brief_lower]

    section_keywords = [
        "company overview",
        "funding",
        "competitor",
        "risk",
        "bull case",
        "bear case",
        "source",
    ]
    sections_present = [s for s in section_keywords if s in brief_lower]

    min_req = test_case.get("min_sections", 6)
    return {
        "expected_phrases_found": len(phrases_found),
        "expected_phrases_total": len(expected_phrases),
        "phrases_coverage": (
            len(phrases_found) / len(expected_phrases) if expected_phrases else 1.0
        ),
        "sections_present": len(sections_present),
        "sections_min_required": min_req,
        "structural_pass": len(sections_present) >= min_req,
    }


async def evaluate_dataset(
    api_url: str, dataset_path: Path, output_path: Path
) -> None:
    with open(dataset_path, encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    openai_client = OpenAI(api_key=settings.openai_api_key)
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(dataset, 1):
            print(f"\n[{i}/{len(dataset)}] {test_case['query']}")

            query_result = await run_query(client, api_url, test_case["query"])

            if not query_result.get("success") and not query_result.get("refused"):
                print(f"  FAILED: {query_result.get('error')}")
                results.append({**test_case, **query_result, "evaluation": None})
                continue

            if query_result.get("refused"):
                print(f"  REFUSED: {query_result.get('reason', '')[:120]}")
                results.append({**test_case, **query_result, "evaluation": "refused"})
                continue

            brief = query_result["brief"]
            dur = query_result["duration_seconds"]
            calls = query_result.get("total_tool_calls", 0)
            print(f"  Brief: {len(brief)} chars  |  {dur:.1f}s  |  {calls} tool calls")

            structural = check_structural_requirements(brief, test_case)
            llm_scores = score_brief(openai_client, test_case["query"], brief)

            coverage = structural["phrases_coverage"]
            sections = structural["sections_present"]
            overall = llm_scores["overall"]
            print(
                f"  Structural: {coverage:.0%} phrase coverage, {sections} sections  |"
                f"  LLM judge: {overall}/10 — {llm_scores.get('rationale', '')[:80]}"
            )

            results.append(
                {
                    **test_case,
                    **query_result,
                    "structural_checks": structural,
                    "llm_evaluation": llm_scores,
                }
            )

    # ── aggregate ─────────────────────────────────────────────────────────────

    successful = [r for r in results if r.get("success")]
    refused = [r for r in results if r.get("refused")]
    failed = [r for r in results if not r.get("success") and not r.get("refused")]

    def avg(seq: list[float]) -> float:
        return round(sum(seq) / len(seq), 2) if seq else 0.0

    if successful:
        metrics = {
            "avg_overall": avg([r["llm_evaluation"]["overall"] for r in successful]),
            "avg_completeness": avg([r["llm_evaluation"]["completeness"] for r in successful]),
            "avg_specificity": avg([r["llm_evaluation"]["specificity"] for r in successful]),
            "avg_balance": avg([r["llm_evaluation"]["balance"] for r in successful]),
            "avg_citations": avg([r["llm_evaluation"]["citations"] for r in successful]),
            "avg_accuracy": avg([r["llm_evaluation"]["accuracy"] for r in successful]),
            "avg_duration_seconds": avg([r["duration_seconds"] for r in successful]),
            "avg_tool_calls": avg([r.get("total_tool_calls", 0) for r in successful]),
        }
    else:
        metrics = {k: 0.0 for k in [
            "avg_overall", "avg_completeness", "avg_specificity",
            "avg_balance", "avg_citations", "avg_accuracy",
            "avg_duration_seconds", "avg_tool_calls",
        ]}

    summary = {
        "total_queries": len(dataset),
        "successful": len(successful),
        "refused": len(refused),
        "failed": len(failed),
        "metrics": metrics,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total queries  : {summary['total_queries']}")
    print(f"Successful     : {summary['successful']}")
    print(f"Refused (guard): {summary['refused']}")
    print(f"Failed         : {summary['failed']}")
    print("\nMetrics (averaged across successful queries):")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IndiaVC research agents")
    parser.add_argument("--dataset", default="data/eval/v1_companies.json")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"docs/eval_results/eval_{int(time.time())}.json"
    asyncio.run(
        evaluate_dataset(args.api, Path(args.dataset), Path(output))
    )


if __name__ == "__main__":
    main()
