import os
from typing import Any

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


st.set_page_config(page_title="Agentic Multimodal AI", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1180px; }
    .chat-shell { border: 1px solid #d8dee9; border-radius: 8px; padding: 1rem; background: #fbfcfe; }
    .trace-step { border-left: 3px solid #2f6feb; padding: .35rem .75rem; margin: .4rem 0; }
    .small-muted { color: #667085; font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Agentic Multimodal AI")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("Request")
    uploaded_files = st.file_uploader(
        "Files",
        accept_multiple_files=True,
        type=[
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tiff",
            "pdf",
            "mp3",
            "wav",
            "m4a",
            "ogg",
            "flac",
            "webm",
            "mp4",
            "txt",
            "md",
            "csv",
            "json",
            "py",
            "js",
            "ts",
            "tsx",
            "html",
            "css",
            "log",
        ],
    )
    message = st.text_area("Message", height=140, placeholder="Ask a question, request a summary, analyze sentiment, explain code, or reason across files.")
    submit = st.button("Run Agent", type="primary", use_container_width=True)

if submit:
    with st.spinner("The agent is extracting, planning, and reasoning..."):
        try:
            multipart_files = [
                ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
                for file in uploaded_files
            ]
            response = requests.post(
                f"{API_BASE_URL}/api/chat",
                data={"message": message},
                files=multipart_files,
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            st.session_state.history.insert(0, {"message": message, "response": payload})
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

for item_index, item in enumerate(st.session_state.history):
    payload: dict[str, Any] = item["response"]
    st.chat_message("user").write(item["message"] or "Analyze uploaded files.")
    with st.chat_message("assistant"):
        st.subheader("Final Answer")
        st.write(payload.get("final_answer", ""))

        cost = payload.get("cost_estimate", {})
        st.caption(
            f"Intent: {payload.get('intent', 'unknown')} | "
            f"Estimated tokens: {cost.get('input_tokens_estimate', 0)} in / {cost.get('output_tokens_estimate', 0)} out | "
            f"Estimated cost: ${cost.get('estimated_usd', 0):.6f}"
        )

        extracted_tab, trace_tab, graph_tab = st.tabs(["Extracted Content", "Tool Steps", "Execution Graph"])
        # with answer_tab:
        #     st.write(payload.get("final_answer", ""))

        with extracted_tab:
            artifacts = payload.get("extracted_text", [])
            if not artifacts:
                st.info("No uploaded files were processed.")
            for artifact_index, artifact in enumerate(artifacts):
                with st.expander(f"{artifact.get('file_name')} ({artifact.get('artifact_type')})", expanded=False):
                    st.text_area(
                        "Text",
                        artifact.get("text", ""),
                        height=220,
                        key=f"artifact_text_{item_index}_{artifact_index}",
                    )
                    if artifact.get("urls"):
                        st.write("URLs:", artifact["urls"])
                    if artifact.get("warnings"):
                        st.warning("\n".join(artifact["warnings"]))
                    st.json(artifact.get("metadata", {}))

        with trace_tab:
            for step in payload.get("tool_execution_trace", []):
                st.markdown(
                    f"""
                    <div class="trace-step">
                    <strong>{step.get('step')}. {step.get('tool')}</strong><br/>
                    <span class="small-muted">Input: {step.get('input_summary')}</span><br/>
                    <span>{step.get('output_summary')}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with graph_tab:
            graph = payload.get("tool_graph", {})
            dot_lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#eef5ff", color="#2f6feb"];']
            for node in graph.get("nodes", []):
                dot_lines.append(f'"{node["id"]}" [label="{node["label"]}"];')
            for edge in graph.get("edges", []):
                dot_lines.append(f'"{edge["source"]}" -> "{edge["target"]}";')
            dot_lines.append("}")
            st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

if not st.session_state.history:
    st.info("Upload one or more files, add a prompt, and run the agent.")
