import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agent import Agent
from tools.hybrid_search import HybridSearchTool
from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool

MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

st.set_page_config(
    page_title="Measuring Devices Assistant",
    layout="centered",
)


@st.cache_resource(show_spinner="Starting up — this may take a moment on first run...")
def load_agent() -> Agent:
    kg = KnowledgeGraphTool()
    vs = VectorStoreTool()
    hs = HybridSearchTool(kg, vs)
    tools = kg.get_tools() + vs.get_tools() + hs.get_tools()
    return Agent(tools=tools, model=MODEL)


st.title("Measuring Devices Assistant")
st.caption("Ask about devices, components, manufacturers, or use cases.")

agent = load_agent()

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Render chat history
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        if entry.get("rewritten_query"):
            st.caption(f"Interpreted as: *{entry['rewritten_query']}*")
        if entry["steps"]:
            with st.expander(f"Tools used ({len(entry['steps'])})"):
                for i, step in enumerate(entry["steps"], 1):
                    st.markdown(f"**Tool {i}**")
                    st.code(step, language=None)
        st.write(entry["answer"])

# Chat input
if query := st.chat_input("Ask about a measuring device or component..."):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        steps_placeholder = st.empty()
        answer_placeholder = st.empty()

        collected_steps = []
        answer_tokens = []
        rewritten_query = None
        rewrite_placeholder = st.empty()

        for event_type, data in agent.stream_run(query, st.session_state.thread_id):
            if event_type == "rewrite":
                rewritten_query = data
                rewrite_placeholder.caption(f"Interpreted as: *{data}*")
            elif event_type == "step":
                collected_steps.append(data)
                with steps_placeholder.container():
                    with st.expander(f"Tools used ({len(collected_steps)})"):
                        for i, s in enumerate(collected_steps, 1):
                            st.markdown(f"**Tool {i}**")
                            st.code(s, language=None)
            elif event_type == "token":
                answer_tokens.append(data)
                answer_placeholder.markdown("".join(answer_tokens))

        answer = "".join(answer_tokens) or "I was unable to complete the task."
        if not answer_tokens:
            answer_placeholder.write(answer)

    st.session_state.history.append({
        "query": query,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "steps": collected_steps,
    })
