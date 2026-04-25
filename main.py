"""
Agentic RAG system for measuring devices.

Prerequisites:
  Start containers: podman-compose up -d  (or docker compose up -d)
  vLLM downloads the model on first start — allow a few minutes.

Usage:
  python main.py                                  # interactive mode
  python main.py -q "your question"               # single query mode
  python main.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 -q "…"
"""

import argparse
import os

from agent import Agent
from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool

DEFAULT_MODEL = os.getenv("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic RAG system for measuring devices.",
    )
    parser.add_argument(
        "-q", "--query",
        metavar="QUERY",
        help="Run a single query and exit (omit for interactive mode).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"HuggingFace model ID served by vLLM (default: {DEFAULT_MODEL}).",
    )
    return parser.parse_args()


def build_agent(model: str) -> tuple[Agent, KnowledgeGraphTool]:
    print("Initializing tools...")
    kg_tool = KnowledgeGraphTool()
    vs_tool = VectorStoreTool()
    print("Tools ready.\n")
    return Agent(tools=[kg_tool, vs_tool], model=model), kg_tool


def main():
    args = parse_args()
    agent, kg_tool = build_agent(args.model)

    try:
        if args.query:
            answer, _ = agent.run(args.query)
            print("\n" + "=" * 60)
            print("Answer:", answer)
            return

        # Interactive mode
        print("Measuring Devices RAG — type 'quit' to exit\n")
        print("Example questions:")
        print("  - What components does the Fluke 87V use?")
        print("  - Which devices use an FPGA?")
        print("  - What device should I use to analyze RF signals?")
        print("  - Tell me about the ADC component and who makes it.\n")

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break

            answer, _ = agent.run(query)
            print("\n" + "=" * 60)
            print(f"Answer: {answer}")
            print("=" * 60 + "\n")
    finally:
        kg_tool.close()


if __name__ == "__main__":
    main()
