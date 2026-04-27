"""
ReAct Agent for the measuring devices RAG system, powered by LangGraph.

Connection settings are read from env vars with sensible defaults:
  VLLM_BASE_URL  (default: http://localhost:8080/v1)
"""

import os
import time

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_react_agent
from openai import OpenAI


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "6"))

REWRITE_PROMPT = """You are a query optimizer for a measuring and test equipment knowledge base.

Rewrite the user query to be clearer and more specific:
- Expand model numbers to full device names where recognizable (e.g. "Fluke 87V" → "Fluke 87V Multimeter")
- Make implicit intent explicit (e.g. "what's inside it?" → "what components does [device] contain?")
- Resolve vague pronouns if the device or component is clear from the query
- Do not add speculation or information not present in the query

If the query is already clear and specific, return it unchanged.
Return ONLY the rewritten query — no explanation, no punctuation changes.

Query: {query}"""

SYSTEM_PROMPT = """You are an expert assistant for measuring and test equipment.

STRICT RULES — follow these without exception:
1. Always call at least one tool before answering.
2. Base your answer ONLY on what the tools returned. Do not add facts, names, or specifications that do not appear in the tool results.
3. If the tools return no results or the information is not found, say exactly that — do not guess or invent an answer.
4. If you are unsure, say so. Never fabricate device names, component names, manufacturers, or specifications.

Device and component names: all tools accept partial or shortened names (e.g. "Fluke 87V" matches "Fluke 87V Multimeter", "ADC" matches "ADC (Analog-to-Digital Converter)"). Never tell the user a device was not found without first trying a shorter or alternative form of the name.

Choose the right tool based on the question:
- get_device_by_name: use this FIRST whenever the user mentions a device by name (full or partial) — resolves the canonical name before calling other tools
- get_components_of_device: what parts or components are inside a specific device
- get_devices_using_component: which devices contain a specific component or part
- get_component_info: description, manufacturer, or category of a specific component
- get_components_by_category: find components by type — valid categories: signal_processing, protection, power, timing, rf, measurement, signal_generation
- list_all_devices: browse all available devices
- list_all_components: browse all available components
- search_devices: find devices by purpose, use case, or measurement type — use for intent queries like "device for RF signal analysis", NOT for name lookups
- hybrid_search: combine semantic search with component details for complex queries involving both purpose and internals

Be concise and factual. Only state what the data confirms."""


class Agent:
    def __init__(self, tools: list, model: str = "Qwen/Qwen2.5-3B-Instruct", max_steps: int = AGENT_MAX_STEPS):
        self.max_steps = max_steps
        self._verify_connection()
        self._memory = MemorySaver()

        self._llm = ChatOpenAI(
            base_url=VLLM_BASE_URL,
            api_key="vllm",
            model=model,
            temperature=0,
        )
        self._graph = create_react_agent(self._llm, tools, checkpointer=self._memory)

    def run(self, user_query: str, thread_id: str = "default") -> tuple[str, list[str]]:
        """Run the agent and return (final_answer, steps)."""
        query = self._rewrite_query(user_query)
        print(f"\n[Agent] Query: {query}")
        print("-" * 60)

        result = self._graph.invoke(
            {"messages": self._build_messages(query, thread_id)},
            config=self._get_config(thread_id),
        )

        messages = result["messages"]
        answer = next(
            (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            "I was unable to complete the task.",
        )
        return answer, self._extract_steps(messages)

    def stream_run(self, user_query: str, thread_id: str = "default"):
        """Yields ('rewrite', query), ('step', step_str), and ('token', token_str) events."""
        query = self._rewrite_query(user_query)
        if query != user_query:
            yield ("rewrite", query)

        for chunk, _ in self._graph.stream(
            {"messages": self._build_messages(query, thread_id)},
            config=self._get_config(thread_id),
            stream_mode="messages",
        ):
            if isinstance(chunk, ToolMessage):
                yield ("step", f"Tool: {getattr(chunk, 'name', 'unknown')}\n\nResult:\n{chunk.content}")
            elif isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
                yield ("token", chunk.content)

    def _build_messages(self, user_query: str, thread_id: str) -> list:
        state = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        messages = []
        if not state.values.get("messages"):
            messages.append(SystemMessage(content=SYSTEM_PROMPT))
        messages.append(HumanMessage(content=user_query))
        return messages

    def _get_config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.max_steps * 2 + 1,
        }

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

    def _rewrite_query(self, user_query: str) -> str:
        prompt = REWRITE_PROMPT.format(query=user_query)
        result = self._llm.invoke([HumanMessage(content=prompt)])
        rewritten = result.content.strip()
        if rewritten:
            print(f"[Agent] Rewritten query: {rewritten}")
        return rewritten or user_query

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
