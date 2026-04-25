"""
ReAct Agent for the measuring devices RAG system.

Loop: Thought → Action → Observation → (repeat or) Final Answer
The agent connects to a vLLM instance via the OpenAI-compatible HTTP API.

Connection settings are read from env vars with sensible defaults:
  VLLM_BASE_URL  (default: http://localhost:8080/v1)
"""

import json
import os
import re
import time

from openai import OpenAI


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8080/v1")

SYSTEM_PROMPT = """You are an expert assistant for measuring and test equipment.
You have access to two tools to answer questions:

## Tools

### knowledge_graph
Query structured data about device components (descriptions, manufacturers, which device uses which component).
Input must be JSON. Available actions:
- {"action": "get_components_of_device", "device_name": "<name>"}
- {"action": "get_devices_using_component", "component_name": "<name>"}
- {"action": "get_component_info", "component_name": "<name>"}
- {"action": "list_all_components"}
- {"action": "list_all_devices"}

### vector_store
Semantic search over device names and descriptions. Use for questions about device purpose or use case.
Input must be JSON. Available actions:
- {"action": "search", "query": "<natural language query>", "top_k": 3}
- {"action": "get_device_by_name", "name": "<partial name>"}

## Response format

Think step by step. Use this exact format:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <JSON input for the tool>
Observation: <tool result — filled in by the system>

Repeat Thought/Action/Observation as needed. When you have enough information:

Thought: I now have enough information to answer.
Final Answer: <your complete answer to the user>

Rules:
- Always use a tool before giving a Final Answer unless the question needs no lookup.
- Never fabricate tool outputs — wait for the Observation.
- Be concise and factual in the Final Answer.
"""


class Agent:
    def __init__(self, tools: list, model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_steps: int = 6):
        self.tools = {t.name: t for t in tools}
        self.model = model
        self.max_steps = max_steps
        self._client = OpenAI(base_url=VLLM_BASE_URL, api_key="vllm")
        self._verify_connection()

    def run(self, user_query: str) -> tuple[str, list[str]]:
        """
        Run the ReAct loop for the given query.
        Returns (answer, steps) where steps is a list of reasoning+observation
        strings — one entry per tool call round.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        steps: list[str] = []

        print(f"\n[Agent] Query: {user_query}")
        print("-" * 60)

        for step_num in range(self.max_steps):
            response = self._call_llm(messages)
            print(f"\n[Step {step_num + 1}]\n{response}")

            # Check for Final Answer
            if "Final Answer:" in response:
                pre_final = response.split("Final Answer:")[0].strip()
                if pre_final:
                    steps.append(pre_final)
                final = response.split("Final Answer:")[-1].strip()
                return final, steps

            # Parse Action + Action Input
            action, action_input = self._parse_action(response)
            if not action:
                return response.strip(), steps

            # Execute tool
            if action not in self.tools:
                observation = json.dumps(
                    {"error": f"Unknown tool '{action}'. Available: {list(self.tools.keys())}"}
                )
            else:
                print(f"\n[Tool] {action} ← {action_input}")
                observation = self.tools[action].run(action_input)
                print(f"[Observation] {observation[:300]}{'...' if len(observation) > 300 else ''}")

            steps.append(f"{response.strip()}\n\nObservation:\n{observation}")

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        return "I was unable to complete the task within the allowed number of steps.", steps

    def _verify_connection(self, retries: int = 60, delay: float = 10.0):
        """
        Wait for vLLM to be reachable AND have a model loaded.
        vLLM returns HTTP 503 while the model is still loading, which the
        OpenAI client surfaces as InternalServerError — distinct from a
        pure network failure (APIConnectionError). Both are retried here.
        """
        print(f"[Agent] Waiting for vLLM at {VLLM_BASE_URL}...")
        for attempt in range(1, retries + 1):
            try:
                models = self._client.models.list()
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

    def _call_llm(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        return response.choices[0].message.content

    def _parse_action(self, text: str) -> tuple[str | None, str | None]:
        action_match = re.search(r"Action:\s*(\w+)", text)
        input_match = re.search(r"Action Input:\s*(\{[^}]*\})", text, re.DOTALL)

        if not action_match or not input_match:
            return None, None

        return action_match.group(1).strip(), input_match.group(1).strip()
