"""
ReAct Agent for the measuring devices RAG system, powered by LangGraph.

Connection settings are read from env vars with sensible defaults:
  VLLM_BASE_URL  (default: http://localhost:8080/v1)
"""

import os
import time

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openai import OpenAI


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "3"))

SYSTEM_PROMPT = """You are an expert assistant for measuring and test equipment.
Use the available tools to answer questions about devices, components, manufacturers, and use cases.
Always use a tool before giving a final answer unless the question needs no lookup.
Be concise and factual."""


class Agent:
    def __init__(self, tools: list, model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_steps: int = AGENT_MAX_STEPS):
        self.max_steps = max_steps
        self._verify_connection()

        llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="vllm",
            model=model,
            temperature=0.1,
        )
        self._graph = create_react_agent(llm, tools, state_modifier=SYSTEM_PROMPT)

    def run(self, user_query: str) -> tuple[str, list[str]]:
        """Run the agent and return (final_answer, steps)."""
        print(f"\n[Agent] Query: {user_query}")
        print("-" * 60)

        result = self._graph.invoke(
            {"messages": [("human", user_query)]},
            config={"recursion_limit": self.max_steps * 2 + 1},
        )

        messages = result["messages"]

        answer = next(
            (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            "I was unable to complete the task.",
        )

        steps = self._extract_steps(messages)
        return answer, steps

    def _extract_steps(self, messages: list) -> list[str]:
        steps = []
        for i, m in enumerate(messages):
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    result = next(
                        (msg.content for msg in messages[i:] if isinstance(msg, ToolMessage) and msg.tool_call_id == tc["id"]),
                        "No result",
                    )
                    steps.append(f"Tool: {tc['name']}\nInput: {tc['args']}\n\nResult:\n{result}")
        return steps

    def _verify_connection(self, retries: int = 60, delay: float = 10.0):
        print(f"[Agent] Waiting for vLLM at {VLLM_BASE_URL}...")
        client = OpenAI(base_url=VLLM_BASE_URL, api_key="vllm")
        for attempt in range(1, retries + 1):
            try:
                models = client.models.list()
                if models.data:
                    print(f"[Agent] vLLM ready (model: {models.data[0].id}).")
                    return
            except Exception:
                pass
            print(f"[Agent] vLLM not ready yet ({attempt}/{retries}), retrying in {int(delay)}s...")
            time.sleep(delay)
        raise RuntimeError(
            f"vLLM not reachable at {VLLM_BASE_URL} after {retries} attempts.\n"
            "Start the containers with:  podman-compose up -d"
        )
