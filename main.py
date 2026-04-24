"""
Agentic RAG system for measuring devices.

Prerequisites:
  1. Start containers:  podman-compose up -d
  2. Ollama running:    ollama serve  (then: ollama pull llama3.2)

Usage:
  python main.py                         # interactive mode
  python main.py -q "your question"      # single query mode
  python main.py --model llama3.1 -q … # override model
"""

import argparse
from agent import Agent
from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool

DEFAULT_MODEL = "llama3.2"


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
        help=f"Ollama model to use (default: {DEFAULT_MODEL}).",
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
            answer = agent.run(args.query)
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

            answer = agent.run(query)
            print("\n" + "=" * 60)
            print(f"Answer: {answer}")
            print("=" * 60 + "\n")
    finally:
        kg_tool.close()


if __name__ == "__main__":
    main()
