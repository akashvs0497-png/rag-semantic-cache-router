import os, sqlite3
import pandas as pd
import streamlit as st
from metrics import DB_PATH

st.set_page_config(page_title="RAG cache + router", layout="wide")
st.title("RAG: semantic cache + cost-aware routing")
st.caption("Costs are API-equivalent estimates from local token counts, using the illustrative prices in metrics.py.")


@st.cache_resource
def get_ask():
    from rag import ask  # loads the models on first use only
    return ask


with st.expander("Ask a question (live)"):
    with st.form("ask_form"):
        q = st.text_input("Question")
        submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        with st.spinner("Thinking..."):
            r = get_ask()(q.strip())
        st.write(r["answer"])
        st.caption(f"{r['source']} | model: {r['model']} | {r['latency']:.2f}s | est. ${r['cost']:.5f}")
        st.caption(r["note"])

st.button("Refresh")


def load():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM requests ORDER BY id", con)
    con.close()
    return df


df = load()
if df.empty:
    st.info("No requests logged yet. Ask a question above, or run test_logging.py.")
    st.stop()

df["when"] = pd.to_datetime(df["ts"], unit="s")
df["route"] = df.apply(lambda r: "cache" if r["source"] == "cache" else r["model"], axis=1)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Requests", len(df))
c2.metric("Cache hit rate", f"{(df['source'] == 'cache').mean():.0%}")
c3.metric("Est. total cost", f"${df['cost_usd'].sum():.4f}")
c4.metric("Avg latency", f"{df['latency'].mean():.2f}s")
c5.metric("p95 latency", f"{df['latency'].quantile(0.95):.2f}s")

left, right = st.columns(2)
with left:
    st.subheader("Requests by route")
    st.bar_chart(df["route"].value_counts())
    st.subheader("Cumulative est. cost ($)")
    st.line_chart(df.set_index("id")["cost_usd"].cumsum())
with right:
    st.subheader("Latency per request (s)")
    st.line_chart(df.set_index("id")["latency"])
    st.subheader("Avg latency by route (s)")
    st.bar_chart(df.groupby("route")["latency"].mean())

st.subheader("Recent requests")
cols = ["when", "query", "source", "model", "similarity", "latency", "tokens_in", "tokens_out", "cost_usd"]
st.dataframe(df[cols].iloc[::-1].head(50), use_container_width=True)
