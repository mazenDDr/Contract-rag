"""A small front end for the contract-rag API.

Run the API first, then:  API_URL=http://127.0.0.1:8000 APP_API_KEY=... streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os
import re

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
HEADERS = {"X-API-Key": key} if (key := os.environ.get("APP_API_KEY")) else {}
API_TIMEOUT = float(os.environ.get("API_TIMEOUT", "180"))  # seconds; a CPU-only model needs minutes

st.set_page_config(page_title="contract-rag", page_icon="⚖️", layout="centered")


@st.cache_data(ttl=300, show_spinner=False)
def load_contracts() -> list[dict]:
    response = httpx.get(f"{API_URL}/contracts", headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


def show_answer(data: dict) -> None:
    if data["abstained"]:
        st.info("The contract excerpts it found don't answer this. It said so instead of guessing.")
    st.markdown(re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", r"**[\1]**", data["answer"]))
    for c in data["citations"]:
        pages = (
            f"page {c['page_start']}"
            if c["page_start"] == c["page_end"]
            else f"pages {c['page_start']}–{c['page_end']}"
        )
        with st.expander(f"[{c['n']}] {pages}" + (f" · {c['section']}" if c["section"] else "")):
            st.write(c["text"])
            st.caption(c["chunk_id"])
    lat = data["latency_ms"]
    st.caption(
        f"search {lat.get('retrieval', 0):.0f} ms · answer {lat.get('generation', 0) / 1000:.1f} s · "
        f"{data['prompt_tokens'] + data['completion_tokens']:,} tokens · {data['model']} · "
        f"request {data['request_id']}"
    )


st.title("Ask a contract")
st.caption("Answers come only from the chosen contract, with a citation for every claim.")

try:
    contracts = load_contracts()
except httpx.HTTPError as exc:
    st.error(
        f"Can't reach the API at {API_URL} ({exc}). Start it with scripts/serve_api.py, and set APP_API_KEY."
    )
    st.stop()

contract = st.selectbox("Contract", contracts, format_func=lambda c: f"{c['title']} · {c['pages']} pages")
question = st.text_area(
    "Question",
    placeholder="Can either party terminate early without cause, and on what notice?",
    max_chars=500,
)
if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the contract and writing the answer…"):
        try:
            response = httpx.post(
                f"{API_URL}/ask",
                json={"doc_id": contract["doc_id"], "question": question},
                headers=HEADERS,
                timeout=API_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            st.error(f"The request failed: {exc}")
            st.stop()
    if response.status_code == 200:
        show_answer(response.json())
    else:
        st.error(f"{response.status_code}: {response.json().get('detail', response.text)}")
