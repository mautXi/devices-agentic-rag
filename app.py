import os
import streamlit as st

from agent import Agent
from tools.knowledge_graph import KnowledgeGraphTool
from tools.vector_store import VectorStoreTool

MODEL = os.getenv("VLLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

st.set_page_config(
    page_title="Measuring Devices Assistant",
    page_icon="⚡",
    layout="centered",
)


@st.cache_resource(show_spinner="Starting up — this may take a moment on first run...")
def load_agent() -> Agent:
    kg = KnowledgeGraphTool()
    vs = VectorStoreTool()
    tools = kg.get_tools() + vs.get_tools()
    return Agent(tools=tools, model=MODEL)


st.title("⚡ Measuring Devices Assistant")
st.caption("Ask about devices, components, manufacturers, or use cases.")

agent = load_agent()

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        if entry["steps"]:
            with st.expander(f"Reasoning ({len(entry['steps'])} step{'s' if len(entry['steps']) > 1 else ''})"):
                for i, step in enumerate(entry["steps"], 1):
                    st.markdown(f"**Step {i}**")
                    st.code(step, language=None)
        st.write(entry["answer"])

if query := st.chat_input("Ask about a measuring device or component..."):
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, steps = agent.run(query)

        if steps:
            with st.expander(f"Reasoning ({len(steps)} step{'s' if len(steps) > 1 else ''})"):
                for i, step in enumerate(steps, 1):
                    st.markdown(f"**Step {i}**")
                    st.code(step, language=None)

        st.write(answer)

    st.session_state.history.append({"query": query, "answer": answer, "steps": steps})
