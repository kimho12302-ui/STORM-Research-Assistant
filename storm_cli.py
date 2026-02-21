"""STORM Research Assistant CLI

Run a STORM research session from the command line.
Results are automatically saved to the Obsidian vault.

Usage:
    python storm_cli.py "AI 에이전트"
    python storm_cli.py "양자 컴퓨팅" --model google/gemini-2.5-pro
    python storm_cli.py "LLM Fine-tuning" --analysts 4 --turns 4
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Load environment variables before importing the graph
load_dotenv(Path(__file__).parent / ".env")

from storm_research.graph import build_research_graph  # noqa: E402


async def run_research(
    topic: str,
    model: str = "google/gemini-2.5-flash",
    max_analysts: int = 3,
    max_interview_turns: int = 3,
    obsidian_vault_path: str | None = None,
    obsidian_folder: str | None = None,
) -> dict:
    """Run a full STORM research session and return the result."""
    configurable = {
        "model": model,
        "max_analysts": max_analysts,
        "max_interview_turns": max_interview_turns,
    }
    if obsidian_vault_path:
        configurable["obsidian_vault_path"] = obsidian_vault_path
    if obsidian_folder:
        configurable["obsidian_folder"] = obsidian_folder

    config = RunnableConfig(configurable=configurable)
    graph = build_research_graph()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=topic)]},
        config,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="STORM Research Assistant CLI",
    )
    parser.add_argument("topic", help="Research topic")
    parser.add_argument(
        "--model",
        default="google/gemini-2.5-flash",
        help="LLM model (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--analysts",
        type=int,
        default=3,
        help="Number of analyst personas (default: 3)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=3,
        help="Max interview turns per analyst (default: 3)",
    )
    parser.add_argument(
        "--vault",
        default=None,
        help="Obsidian vault path (overrides default)",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Folder inside vault (default: Research)",
    )

    args = parser.parse_args()

    print(f"[STORM] Starting research on: {args.topic}")
    print(f"[STORM] Model: {args.model} | Analysts: {args.analysts} | Turns: {args.turns}")
    print()

    result = asyncio.run(
        run_research(
            topic=args.topic,
            model=args.model,
            max_analysts=args.analysts,
            max_interview_turns=args.turns,
            obsidian_vault_path=args.vault,
            obsidian_folder=args.folder,
        )
    )

    if result.get("file_path"):
        print(f"\n[STORM] Report saved to: {result['file_path']}")
    else:
        print("\n[STORM] Report generated (file_path not returned).")

    print("\n--- Final Report Preview (first 500 chars) ---")
    report = result.get("final_report", "")
    print(report[:500])
    if len(report) > 500:
        print("...")


if __name__ == "__main__":
    main()
