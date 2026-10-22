import os
import time

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
ROUTES = {"policy": "university pages", "study": "EEG study graph", "literature": "published research",
          "safety:crisis": "safety referral", "safety:distress": "support referral"}
EXAMPLES = ["What CGPA puts me on academic probation?", "Can I use ChatGPT for my assignments?",
            "Does our EEG data replicate frontal alpha asymmetry in depression?",
            "What do published studies say about alpha power in depression?"]

st.set_page_config(page_title="NeuroGraph assistant", page_icon="🧠", layout="centered")


def api(method: str, path: str, **kw):
    r = requests.request(method, f"{API}{path}", timeout=300, **kw)
    r.raise_for_status()
    return r.json()


def wait_for_api() -> dict:
    for _ in range(120):  # first start downloads the model and builds the index
        try:
            h = api("GET", "/health")
            if h["status"] == "ok":
                return h
        except requests.RequestException:
            pass
        time.sleep(5)
    st.error("The assistant did not start. Check the Space logs.")
    st.stop()


def badge(res: dict) -> str:
    route = ROUTES.get(res["route"], res["route"])
    if res["route"].startswith("safety"):
        return f"🛟 {route}"
    check = "✅ checked against the sources" if res["verdict"] == "supported" else "⚠️ not fully verified"
    return f"🧭 {route} · {check} · {res['seconds']:.0f} s"


def vote(res_id: str, v: str) -> None:
    api("POST", "/feedback", json={"id": res_id, "vote": v})
    st.session_state.voted.add(res_id)


health = wait_for_api()
st.title("NeuroGraph assistant")
st.caption("Student questions (rules, deadlines, services) and the NeuroGraph EEG research database, "
           "answered from public sources and checked before they are shown.")
st.warning(health["disclaimer"], icon="⚠️")

with st.sidebar:
    st.subheader("How it works")
    st.markdown("Safety check → router → policy page search or knowledge-graph query → Qwen2.5-3B (QLoRA fine-tune for "
                "policy questions, base model for research) → verifier (numbers must be in the sources, each sentence "
                "checked by an NLI model, effect directions checked against the statistics).")
    st.markdown(f"**Model:** {health['llm']}  \n**Graph:** {health['graph']}  \n**Policy passages:** {health['chunks']}")
    st.subheader("Monitoring")
    s = api("GET", "/stats")
    st.markdown(f"answers {s['answers']} · verified {s['verified_share'] if s['verified_share'] is not None else '-'}  \n"
                f"median {s['seconds_median'] or '-'} s · p95 {s['seconds_p95'] or '-'} s  \n"
                f"feedback 👍 {s['feedback_up']} · 👎 {s['feedback_down']}")
    if s["routes"]:
        st.bar_chart(s["routes"], horizontal=True)
    st.caption("Questions are logged with personal details removed; messages about distress are not stored.")

st.session_state.setdefault("history", [])
st.session_state.setdefault("voted", set())
for i, (q, res) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(res["answer"])
        st.caption(badge(res))
        if res["id"] not in st.session_state.voted:
            c1, c2, _ = st.columns([1, 1, 8])
            c1.button("👍", key=f"up{i}", on_click=vote, args=(res["id"], "up"))
            c2.button("👎", key=f"down{i}", on_click=vote, args=(res["id"], "down"))
        else:
            st.caption("Thanks for the feedback.")

if not st.session_state.history:
    cols = st.columns(2)
    for k, ex in enumerate(EXAMPLES):
        if cols[k % 2].button(ex, key=f"ex{k}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Ask about university rules and services, or about the EEG study") or st.session_state.pop("pending", None)
if question:
    with st.spinner("Searching, answering and checking (up to 2 minutes on the free CPU)…"):
        res = api("POST", "/ask", json={"question": question})
    st.session_state.history.append((question, res))
    st.rerun()
