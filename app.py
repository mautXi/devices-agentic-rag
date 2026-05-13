import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from agent import Agent
from data.sample_data import DEVICES
from tools.hybrid_search import HybridSearchTool
from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool

MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

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

with st.sidebar:
    st.header("Filter by device")
    device_names = [d["name"] for d in sorted(DEVICES, key=lambda d: d["name"])]
    selected_device = st.selectbox(
        "Device context",
        options=["All devices"] + device_names,
        label_visibility="collapsed",
    )

device_filter = selected_device if selected_device != "All devices" else None
if device_filter:
    st.sidebar.caption(f"Questions will be scoped to **{device_filter}**.")

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])
        if entry.get("device_filter"):
            st.caption(f"Device: *{entry['device_filter']}*")
    with st.chat_message("assistant"):
        if entry.get("rewritten_query"):
            st.caption(f"Interpreted as: *{entry['rewritten_query']}*")
        st.write(entry["answer"])

if query := st.chat_input("Ask about a measuring device or component..."):
    with st.chat_message("user"):
        st.write(query)
        if device_filter:
            st.caption(f"Device: *{device_filter}*")

    effective_query = f"Regarding the {device_filter}: {query}" if device_filter else query

    with st.chat_message("assistant"):
        rewrite_placeholder = st.empty()
        answer_placeholder = st.empty()

        answer_tokens = []
        rewritten_query = None

        for event_type, data in agent.stream_run(effective_query, st.session_state.thread_id):
            if event_type == "rewrite":
                rewritten_query = data
                rewrite_placeholder.caption(f"Interpreted as: *{data}*")
            elif event_type == "step":
                print(data)
            elif event_type == "token":
                answer_tokens.append(data)
                answer_placeholder.markdown("".join(answer_tokens))

        answer = "".join(answer_tokens) or "I was unable to complete the task."
        if not answer_tokens:
            answer_placeholder.write(answer)

    st.session_state.history.append({
        "query": query,
        "device_filter": device_filter,
        "rewritten_query": rewritten_query,
        "answer": answer,
    })
