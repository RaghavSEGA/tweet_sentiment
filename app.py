import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import io
import time
import os
import boto3
from botocore.exceptions import ClientError

from collector import fetch_tweets
from analyzer import analyze_tweets_batch
from storage import init_db, save_tweets, load_tweets, get_projects, save_project, load_project_config

init_db()

st.set_page_config(
    page_title="X Sentiment Tool",
    page_icon="🎮",
    layout="wide"
)

# ── Global dark theme ─────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
  --bg:      #0a0c1a;
  --surface: #0f1120;
  --border:  #232640;
  --blue:    #4080ff;
  --text:    #eef0fa;
  --muted:   #5a5f82;
  --green:   #22c55e;
  --red:     #ef4444;
  --yellow:  #f5c218;
}
html, body, .stApp { background: var(--bg) !important; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
.stButton>button { background: var(--blue) !important; color: #fff !important; border: none !important; border-radius: 6px !important; }
.stButton>button:hover { opacity: 0.85; }
h1,h2,h3,h4,h5,h6 { color: var(--text) !important; }
.stMarkdown p, .stMarkdown li { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)


# ── Secrets — loaded once from AWS Secrets Manager ────────────────────────────
@st.cache_resource
def _load_secrets():
    """
    Fetch API credentials from Secrets Manager.
    The ECS task role grants access — no hardcoded credentials needed.
    """
    client = boto3.client("secretsmanager", region_name="us-east-2")

    def _get(secret_name):
        try:
            resp = client.get_secret_value(SecretId=secret_name)
            return json.loads(resp["SecretString"])
        except ClientError as e:
            st.error(f"Could not load secret '{secret_name}': {e}")
            st.stop()

    twitter = _get("soa-tools/xsentiment/twitter")
    return twitter["bearer_token"]

TWITTER_BEARER = _load_secrets()

# Bedrock client — auth via ECS task role, no API key needed
BEDROCK_MODEL  = "anthropic.claude-sonnet-4-6"
BEDROCK_REGION = "us-east-2"


# ── Identity — injected by ALB OIDC, no login screen needed ──────────────────
def _get_owner() -> str:
    """
    Read the authenticated user's email from the ALB OIDC identity header.
    Falls back to an env var for local development.
    """
    # In production the ALB sets this header after OIDC auth
    oidc_identity = os.environ.get("OIDC_IDENTITY", "")
    if oidc_identity:
        return oidc_identity.lower()
    # Local dev fallback — override with your own email
    return os.environ.get("DEV_USER_EMAIL", "dev@segaamerica.com")

OWNER = _get_owner()


# ── Sidebar: Project Management ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
        f'color:#5a5f82;margin-bottom:.15rem;">SEGA America</div>'
        f'<div style="font-size:.75rem;font-weight:600;color:#b8bcd4;margin-bottom:.5rem;">'
        f'{OWNER}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:none;border-top:1px solid #232640;margin:.5rem 0 .75rem;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.85rem;font-weight:700;color:#eef0fa;margin-bottom:.25rem;">🎮 X Sentiment Tool</div>', unsafe_allow_html=True)
    st.caption("Collect, analyse, and visualise X sentiment for any game or topic.")
    st.divider()

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
**Follow the tabs left to right:**

1. **⚙️ Configure** — Set keywords, handles, and custom categories. Categories tell Claude how to group tweets beyond positive/negative/neutral.

2. **📥 Collect** — Pull tweets from the X API. Tweets are saved so you can collect incrementally without duplicates.

3. **🤖 Analyze** — Send tweets to Claude in batches. Each tweet gets a sentiment, category, score (−1.0 to +1.0), and a one-sentence reason.

4. **📊 Dashboard** — Explore results: sentiment breakdown, category distribution, trends over time, word clouds, and top/bottom tweets.

5. **💾 Export** — Download the full dataset as CSV or Excel.

---
**Projects** keep each game or campaign separate — own tweets, categories, and settings.
        """)

    st.divider()

    # Project selector — persisted in URL so it survives refresh
    projects = get_projects(OWNER)
    project_names = [p["name"] for p in projects] + ["+ New Project"]

    _qp = st.query_params.get("project", "")
    _default_idx = project_names.index(_qp) if _qp in project_names else 0

    selected_project = st.selectbox("Project", project_names, index=_default_idx)

    if selected_project == "+ New Project":
        new_name = st.text_input("Project name")
        if st.button("Create") and new_name:
            save_project({"name": new_name, "keywords": [], "handles": [], "categories": []}, OWNER)
            st.query_params["project"] = new_name
            st.rerun()
    else:
        st.session_state["active_project"] = selected_project
        st.query_params["project"] = selected_project


# ── Load active project ───────────────────────────────────────────────────────
active = st.session_state.get("active_project") or st.query_params.get("project", "")

if not active or active not in [p["name"] for p in get_projects(OWNER)]:
    st.info("👈 Create or select a project in the sidebar to get started.")
    st.stop()

config = load_project_config(active, OWNER)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_config, tab_collect, tab_analyze, tab_dashboard, tab_export, tab_report = st.tabs(
    ["⚙️ Configure", "📥 Collect", "🤖 Analyze", "📊 Dashboard", "💾 Export", "📝 Report & Chat"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CONFIGURE
# ══════════════════════════════════════════════════════════════════════════════
with tab_config:
    st.header(f"Configure: {active}")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Set up what to search for and how Claude should categorise results.

**Search Terms**
- **Keywords** — Any tweet containing this text will be collected. Use plain phrases (`Stranger Than Heaven`), hashtags (`#STHGame`), or both. One per line.
- **Handles** — Collect tweets that mention or reply to these accounts (e.g. `@STH_Game`). Include the `@`.

**Sentiment Categories**
Every tweet automatically gets a sentiment (`positive` / `negative` / `neutral`). Categories are a *second layer* on top of that — they let you group tweets by topic or theme so you can see not just *how people feel*, but *what they're talking about*.

Define as many categories as you need. Claude reads the name and description for each one and picks the best match per tweet. The more specific and example-rich your descriptions, the more accurate the results.

> 💡 **Tip:** You can update categories at any time. Re-running analysis will only process tweets that haven't been classified yet.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Search Terms")
        keywords_raw = st.text_area(
            "Keywords (one per line)",
            value="\n".join(config.get("keywords", [])),
            height=120,
            placeholder="Stranger Than Heaven\n#STHGame"
        )
        handles_raw = st.text_area(
            "X handles to track (one per line, with @)",
            value="\n".join(config.get("handles", [])),
            height=80,
            placeholder="@STH_Game"
        )

    with col2:
        st.subheader("🏷️ Sentiment Categories")
        st.markdown("""
Claude will assign every tweet **two** things:
1. A **sentiment** — always `positive`, `negative`, or `neutral` (automatic, no setup needed)
2. A **category** — one of the custom buckets you define below

Categories let you go deeper than just "good vs bad". Instead of a pile of negative tweets, you can see *which kind* of negative — is it backlash about the story? Frustration with gameplay? Controversy about a specific person? Each category you create becomes a filter and a bar in the dashboard.

**How to write good categories:**
- **Name** — Short label, shown in charts. Keep it under ~30 characters.
- **Description** — This is what Claude actually reads. Be specific. List example phrases, names, topics, or situations that belong here.

If a tweet doesn't clearly fit any category, Claude will assign it **Other**.
        """)

    st.divider()
    st.subheader("🏷️ Define Categories")

    existing_cats = config.get("categories", [])
    num_cats = st.number_input("Number of categories", min_value=1, max_value=20,
                                value=max(len(existing_cats), 1), step=1)

    new_cats = []
    for i in range(int(num_cats)):
        existing = existing_cats[i] if i < len(existing_cats) else {}
        c1, c2 = st.columns([1, 3])
        with c1:
            name = st.text_input(f"Name #{i+1}", value=existing.get("name", ""),
                                  key=f"cat_name_{i}", placeholder="e.g. Story Feedback")
        with c2:
            desc = st.text_input(f"Description #{i+1}", value=existing.get("description", ""),
                                  key=f"cat_desc_{i}",
                                  placeholder="e.g. Tweets about the plot, characters, writing quality")
        if name:
            new_cats.append({"name": name, "description": desc})

    if st.button("💾 Save Configuration", type="primary"):
        keywords  = [k.strip() for k in keywords_raw.splitlines() if k.strip()]
        handles   = [h.strip() for h in handles_raw.splitlines() if h.strip()]
        new_config = {
            "name":       active,
            "keywords":   keywords,
            "handles":    handles,
            "categories": new_cats,
        }
        save_project(new_config, OWNER)
        st.success("✅ Configuration saved.")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLLECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_collect:
    st.header("📥 Collect Tweets")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Fetches tweets from the X API v2 that match your configured keywords and handles.

**Settings**
- **Date range** — The window to search within. See the API tier note below for limits.
- **Max tweets** — Cap on how many tweets to retrieve per run. X returns up to 100 per page; the tool paginates automatically.
- **Include retweets** — Retweets add volume but are often noise. Recommended: leave unchecked unless you specifically want to track how content spreads.

**How collection works**
All your keywords and handles are combined into a single X API query using OR logic — so a tweet only needs to match *one* of your terms to be collected. Results are saved to the database. Re-running collection on the same date range will **not** create duplicates.

**X API date limits**
The free tier restricts searches to a rolling **7-day window**. If you select a start date of "7 days ago", the tool automatically adjusts `start_time` forward to the earliest second X will accept.

> ⚠️ If you get a 403 error, your X developer app may not have the correct permissions. Ensure it has **Read** access and that your Bearer Token is from an app with access to the v2 search endpoint.
        """)

    bearer = TWITTER_BEARER

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=7))
        max_results = st.slider("Max tweets per query", 10, 500, 100, step=10)
    with col2:
        end_date = st.date_input("To", value=datetime.now())
        include_retweets = st.checkbox("Include retweets", value=False)

    with st.expander("⏱️ X API limits — read this before fetching", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**🗓️ Date range (7-day rolling window)**

The free X API tier only allows searching tweets from the **last 7 days**. This tool automatically clamps your start date forward if it falls outside the allowed window.

To access older tweets you need the **Basic tier ($100/mo)**, which unlocks the full archive back to 2006.

---

**📄 Page size**

X returns tweets in pages of up to 100. Each page = one API request. If you set Max Tweets to 200, that's at least 2 requests.
            """)
        with col_b:
            st.markdown("""
**🚦 Rate limits (why it may stop early)**

The free tier allows **1 search request every 15 minutes**. This means:

| Max Tweets | Pages needed | Min time |
|---|---|---|
| 10–100 | 1 page | ~instant |
| 101–200 | 2 pages | 15+ min wait |
| 201–300 | 3 pages | 30+ min wait |

If the rate limit is hit mid-fetch, the tool will **stop and save whatever it collected so far**. You'll see a warning telling you how many tweets were saved. Simply fetch again after 15 minutes — duplicates are automatically skipped.

---

**💡 Tip:** Keep Max Tweets at 100 or below during testing to stay within one request per run.
            """)

    keywords = config.get("keywords", [])
    handles  = config.get("handles", [])

    if not keywords and not handles:
        st.info("Configure keywords/handles in the Configure tab first.")
    else:
        st.write("**Will search for:**")
        for kw in keywords:
            st.markdown(f"- keyword: `{kw}`")
        for h in handles:
            st.markdown(f"- mention of: `{h}`")

        if st.button("🚀 Fetch Tweets", type="primary"):
            st.session_state["last_fetch_time"] = datetime.now()
            progress_bar   = st.progress(0, text="Connecting to X API...")
            count_display  = st.empty()
            latest_tweet   = st.empty()
            status         = st.empty()
            api_response_box = st.empty()

            with st.expander("🔍 Debug — query sent to X", expanded=True):
                from collector import build_query, compute_date_range
                debug_query = build_query(keywords, handles, include_retweets)
                debug_start, debug_end = compute_date_range(start_date, end_date)
                st.code(f"Query:      {debug_query}\nStart time: {debug_start}\nEnd time:   {debug_end}", language="text")

            tweets    = []
            api_error = None
            try:
                generator = fetch_tweets(
                    bearer_token=bearer,
                    keywords=keywords,
                    handles=handles,
                    start_date=start_date,
                    end_date=end_date,
                    max_results=max_results,
                    include_retweets=include_retweets,
                )

                progress_bar.progress(0, text="⏳ Waiting for X API response...")

                for item in generator:
                    if "_error" in item:
                        api_error = item
                        break

                    if not tweets:
                        progress_bar.progress(0, text="✅ Connected — fetching tweets...")

                    tweets.append(item)
                    pct = min(len(tweets) / max_results, 1.0)
                    progress_bar.progress(pct, text=f"Fetching... {len(tweets)} / {max_results}")
                    count_display.metric("Tweets fetched so far", len(tweets))

                    ts      = item.get("created_at", "")[:16].replace("T", " ")
                    handle  = item.get("username") or item.get("author_id", "unknown")
                    preview = item["text"][:120] + ("…" if len(item["text"]) > 120 else "")
                    latest_tweet.info(f"**@{handle}** · {ts}\n\n{preview}")

                progress_bar.empty()
                count_display.empty()
                latest_tweet.empty()

                if api_error:
                    code = api_error["_error"]
                    msg  = api_error["_message"]
                    if code == "429":
                        reset     = api_error.get("_reset_time")
                        reset_str = reset.strftime("%H:%M UTC") if reset else "~15 min from now"
                        api_response_box.error(
                            f"**API response: 429 Too Many Requests**\n\n"
                            f"Rate limit hit. Estimated reset: **{reset_str}**\n\n"
                            f"Raw: `{msg}`"
                        )
                        st.session_state["rate_limit_hit"] = datetime.now()
                    elif code == "401":
                        api_response_box.error(f"**API response: 401 Unauthorized** — Bearer token is invalid or expired.\n\n`{msg}`")
                    elif code == "403":
                        api_response_box.error(f"**API response: 403 Forbidden** — Your app may not have search permissions.\n\n`{msg}`")
                    elif code == "5xx":
                        api_response_box.error(f"**API response: Server Error** — X is having issues, try again shortly.\n\n`{msg}`")
                    else:
                        api_response_box.error(f"**API error: {code}**\n\n`{msg}`")
                elif tweets:
                    api_response_box.success(f"**API response: 200 OK** — {len(tweets)} tweets returned.")
                else:
                    api_response_box.warning("**API response: 200 OK** — Query succeeded but returned no results. Try broadening your keywords.")

                if tweets:
                    save_tweets(active, OWNER, tweets)
                    if not api_error:
                        status.success(f"✅ Fetched and saved **{len(tweets)} tweets**.")
                    else:
                        status.warning(f"⚠️ Saved **{len(tweets)} tweets** collected before the error.")
                    st.dataframe(
                        pd.DataFrame(tweets)[["id", "created_at", "username", "text"]].head(20),
                        use_container_width=True,
                    )

            except Exception as e:
                progress_bar.empty()
                api_response_box.error(f"**Unexpected error:** `{e}`")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    st.header("🤖 Analyze with Claude")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Sends your collected tweets to Claude (via the Anthropic API) for sentiment classification.

**What Claude returns for each tweet**
| Field | Description |
|---|---|
| **Sentiment** | `positive`, `negative`, or `neutral` |
| **Score** | Float from `−1.0` (most negative) to `+1.0` (most positive) |
| **Category** | The best-matching category from your Configure tab, or *Other* |
| **Reasoning** | One sentence explaining the classification |

**Batch size**
Tweets are sent to Claude in groups. Larger batches are more efficient (fewer API calls, lower cost) but may be slightly less accurate on edge cases. A batch size of 20 is a good default.

**Incremental analysis**
Only **unanalysed** tweets are sent — so if you collect more tweets tomorrow and re-run, only the new ones will be processed. You won't be charged for tweets already classified.

**Cost estimate**
At roughly 50 tokens per tweet (input + output), 1,000 tweets costs approximately $0.15–$0.30 using Claude Sonnet.
        """)

    api_key = ANTHROPIC_KEY

    df_raw = load_tweets(active, OWNER)
    if df_raw.empty:
        st.info("No tweets collected yet. Go to the Collect tab first.")
        st.stop()

    categories = config.get("categories", [])
    if not categories:
        st.warning("No categories defined. Add some in the Configure tab.")

    unanalysed = df_raw[df_raw["sentiment"].isna()] if "sentiment" in df_raw.columns else df_raw
    total      = len(df_raw)
    analysed   = total - len(unanalysed)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total tweets",    total)
    col2.metric("Already analysed", analysed)
    col3.metric("Pending",          len(unanalysed))

    if unanalysed.empty:
        st.success("✅ All tweets have been analysed.")
    else:
        batch_size = st.slider("Batch size", 5, 50, 20, step=5,
                               help="Number of tweets sent to Claude per API call.")
        if st.button("🤖 Run Analysis", type="primary"):
            tweets_to_analyse = unanalysed.to_dict("records")
            batches = [tweets_to_analyse[i:i+batch_size]
                       for i in range(0, len(tweets_to_analyse), batch_size)]

            progress = st.progress(0, text="Starting analysis...")
            status   = st.empty()

            total_done = 0
            for idx, batch in enumerate(batches):
                try:
                    results = analyze_tweets_batch(batch, categories, api_key)
                    save_tweets(active, OWNER, results, update=True)
                    total_done += len(results)
                    pct = (idx + 1) / len(batches)
                    progress.progress(pct, text=f"Analysed {total_done} / {len(tweets_to_analyse)} tweets...")
                except Exception as e:
                    st.error(f"Error on batch {idx+1}: {e}")
                    break

            progress.empty()
            status.success(f"✅ Analysis complete — {total_done} tweets classified.")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.header("📊 Dashboard")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Visualise your analysed tweet data.

**Top / Bottom Tweets**
The 5 highest-scoring positive and 5 lowest-scoring negative tweets, surfaced automatically. Useful for pulling representative quotes for a report.

> 💡 **Tip:** Use the category filter to focus on a specific topic.
        """)

    df = load_tweets(active, OWNER)
    if df.empty or "sentiment" not in df.columns or df["sentiment"].isna().all():
        st.info("No analyzed data yet. Run analysis first.")
        st.stop()

    df = df.dropna(subset=["sentiment"])
    df["created_at"] = pd.to_datetime(df["created_at"])

    # ── Filters
    with st.expander("🔧 Filters"):
        fc1, fc2 = st.columns(2)
        with fc1:
            sentiment_filter = st.multiselect(
                "Sentiment", ["positive", "negative", "neutral"],
                default=["positive", "negative", "neutral"]
            )
        with fc2:
            all_cats   = df["category"].dropna().unique().tolist() if "category" in df.columns else []
            cat_filter = st.multiselect("Category", all_cats, default=all_cats)

    mask = df["sentiment"].isin(sentiment_filter)
    if cat_filter and "category" in df.columns:
        mask &= df["category"].isin(cat_filter)
    df_f = df[mask]

    # ── Top metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Tweets",  len(df_f))
    m2.metric("Positive",      (df_f["sentiment"] == "positive").sum())
    m3.metric("Negative",      (df_f["sentiment"] == "negative").sum())
    m4.metric("Neutral",       (df_f["sentiment"] == "neutral").sum())

    if "like_count" in df_f.columns and "retweet_count" in df_f.columns:
        df_f = df_f.copy()
        df_f["engagement"]  = df_f["like_count"].fillna(0) + df_f["retweet_count"].fillna(0) * 2
        total_engagement    = df_f["engagement"].sum()
        weighted_score      = (
            (df_f["score"].fillna(0) * df_f["engagement"]).sum() / total_engagement
            if total_engagement > 0
            else df_f["score"].fillna(0).mean()
        )
    else:
        weighted_score = df_f["score"].fillna(0).mean()

    score_label = "🟢 Positive" if weighted_score > 0.1 else "🔴 Negative" if weighted_score < -0.1 else "⚪ Neutral"
    m5.metric("Engagement-Weighted Score", f"{weighted_score:.2f}",
              help=f"{score_label} — weighted by likes + 2× retweets")

    st.divider()

    # ── Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Sentiment Breakdown")
        sent_counts = df_f["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        color_map = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#5a5f82"}
        fig_pie = px.pie(
            sent_counts, names="sentiment", values="count",
            color="sentiment", color_discrete_map=color_map,
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#eef0fa")
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        if "category" in df_f.columns:
            st.subheader("Category Distribution")
            cat_counts = df_f["category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig_bar = px.bar(cat_counts, x="count", y="category", orientation="h",
                             color_discrete_sequence=["#4080ff"])
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#eef0fa", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Timeline
    st.subheader("Sentiment Over Time")
    df_time = df_f.copy()
    df_time["date"] = df_time["created_at"].dt.date
    time_data = df_time.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig_line = px.line(time_data, x="date", y="count", color="sentiment",
                       color_discrete_map=color_map)
    fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#eef0fa")
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Word cloud
    st.subheader("Word Cloud")
    all_text = " ".join(df_f["text"].dropna().tolist())
    if all_text.strip():
        wc = WordCloud(width=800, height=300, background_color="#0a0c1a",
                       colormap="Blues", max_words=100).generate(all_text)
        fig_wc, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        fig_wc.patch.set_facecolor("#0a0c1a")
        st.pyplot(fig_wc)
        plt.close(fig_wc)

    st.divider()

    # ── Top / Bottom tweets
    st.subheader("🔝 Top Positive Tweets")
    pos = df_f[df_f["sentiment"] == "positive"].sort_values("score", ascending=False).head(5)
    for _, row in pos.iterrows():
        with st.container(border=True):
            handle = row.get("username") or row.get("author_id", "unknown")
            st.markdown(f"**@{handle}** · {row['created_at'].strftime('%b %d, %H:%M UTC')}")
            st.write(row["text"])
            likes = int(row.get("like_count", 0) or 0)
            rts   = int(row.get("retweet_count", 0) or 0)
            st.caption(f"Category: {row.get('category','—')} · Score: {row.get('score','—')} · ❤️ {likes} · 🔁 {rts}")

    st.subheader("🔻 Top Negative Tweets")
    neg = df_f[df_f["sentiment"] == "negative"].sort_values("score").head(5)
    for _, row in neg.iterrows():
        with st.container(border=True):
            handle = row.get("username") or row.get("author_id", "unknown")
            st.markdown(f"**@{handle}** · {row['created_at'].strftime('%b %d, %H:%M UTC')}")
            st.write(row["text"])
            likes = int(row.get("like_count", 0) or 0)
            rts   = int(row.get("retweet_count", 0) or 0)
            st.caption(f"Category: {row.get('category','—')} · Score: {row.get('score','—')} · ❤️ {likes} · 🔁 {rts}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("💾 Export Data")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Download the full tweet dataset for this project — including all raw tweet data and Claude's analysis results.

**Formats**
- **CSV** — Best for Excel, Google Sheets, or further data processing
- **Excel (.xlsx)** — Pre-formatted spreadsheet, useful for sharing reports directly
        """)

    df_exp = load_tweets(active, OWNER)
    if df_exp.empty:
        st.info("No data to export yet.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df_exp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                csv_data,
                file_name=f"{active}_tweets.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                df_exp.to_excel(writer, index=False, sheet_name="Tweets")
            st.download_button(
                "⬇️ Download Excel",
                excel_buf.getvalue(),
                file_name=f"{active}_tweets.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        st.dataframe(df_exp.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT & CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.header("📝 Report & Chat")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Generate an AI-written executive report and ask follow-up questions about your data.

**Generate Report** produces a structured analysis with:
- Executive summary
- Sentiment breakdown
- Category analysis
- Top praise and concerns
- Influencer signals
- Actionable recommendations

After generating, you can ask follow-up questions in the chat — Claude has full context of your dataset.
        """)

    df_chat = load_tweets(active, OWNER)
    if df_chat.empty or "sentiment" not in df_chat.columns or df_chat["sentiment"].isna().all():
        st.info("No analysed data yet. Run analysis first.")
        st.stop()

    def _build_context(df, cfg):
        project_name = active
        keywords     = cfg.get("keywords", [])
        handles      = cfg.get("handles", [])
        cat_defs     = cfg.get("categories", [])

        df_a = df.dropna(subset=["sentiment"])
        total    = len(df_a)
        analysed = total
        pos      = (df_a["sentiment"] == "positive").sum()
        neg      = (df_a["sentiment"] == "negative").sum()
        neu      = (df_a["sentiment"] == "neutral").sum()
        avg_score = df_a["score"].mean() if "score" in df_a.columns else None

        categories = df_a["category"].value_counts().to_dict() if "category" in df_a.columns else {}

        if "like_count" in df_a.columns and "retweet_count" in df_a.columns:
            df_a = df_a.copy()
            df_a["engagement"] = df_a["like_count"].fillna(0) + df_a["retweet_count"].fillna(0) * 2
            top_tweets    = df_a.sort_values("engagement", ascending=False).head(40).to_dict("records")
            sample_tweets = df_a.sample(min(60, len(df_a))).to_dict("records")
        else:
            top_tweets    = df_a.head(40).to_dict("records")
            sample_tweets = df_a.sample(min(60, len(df_a))).to_dict("records")

        lines = [
            f"PROJECT: {project_name}",
            f"SEARCH KEYWORDS: {', '.join(keywords)}",
            f"MONITORED HANDLES: {', '.join(handles)}",
            "",
            "CUSTOM CATEGORIES:",
        ]
        for c in cat_defs:
            lines.append(f"  - {c['name']}: {c['description']}")
        lines += [
            "",
            f"TOTAL TWEETS: {total}  |  ANALYSED: {analysed}",
            f"SENTIMENT BREAKDOWN — Positive: {pos} ({pos/analysed*100:.1f}%)  "
            f"Negative: {neg} ({neg/analysed*100:.1f}%)  "
            f"Neutral: {neu} ({neu/analysed*100:.1f}%)",
        ]
        if avg_score is not None:
            lines.append(f"AVERAGE SENTIMENT SCORE: {avg_score:.3f}  (range −1.0 to +1.0)")
        lines += ["", "CATEGORY COUNTS:"]
        for cat, cnt in categories.items():
            lines.append(f"  {cat}: {cnt}")
        lines += ["", "TOP 40 TWEETS BY ENGAGEMENT:"]
        for t in top_tweets:
            lines.append(
                f"  [@{t.get('username','?')}] [{t.get('sentiment','?').upper()}] "
                f"[{t.get('category','?')}] score={t.get('score','?')}  "
                f"❤️{t.get('like_count',0)} 🔁{t.get('retweet_count',0)}\n"
                f"  \"{t.get('text','')}\"\n"
                f"  → {t.get('reasoning','')}"
            )
        lines += ["", "RANDOM SAMPLE (60 additional tweets):"]
        for t in sample_tweets:
            lines.append(
                f"  [@{t.get('username','?')}] [{t.get('sentiment','?').upper()}] "
                f"[{t.get('category','?')}] score={t.get('score','?')}\n"
                f"  \"{t.get('text','')}\""
            )
        return "\n".join(lines)

    SYSTEM_PROMPT = (
        "You are a senior social media analyst embedded in SEGA America's insights team. "
        "You have access to a structured dataset of tweets collected and analysed for a specific "
        "game or project. All sentiment classifications and category labels were produced by an "
        "AI classifier — treat them as high-quality but not infallible. "
        "Answer questions analytically, cite specific tweets when relevant, and keep your tone "
        "professional but approachable. When generating a report, structure it with clear sections "
        "and use markdown formatting."
    )

    REPORT_PROMPT = (
        "Generate a comprehensive executive report for the following dataset. "
        "Structure it with these sections:\n"
        "1. **Executive Summary** — 3–5 sentence overview of overall sentiment and key findings\n"
        "2. **Sentiment Overview** — breakdown of positive/negative/neutral with notable trends\n"
        "3. **Category Analysis** — what each category reveals; which are driving positive vs negative sentiment\n"
        "4. **Top Praise** — 3–5 specific things players/fans love most, with example tweets\n"
        "5. **Top Concerns** — 3–5 specific issues or criticisms, with example tweets\n"
        "6. **Influencer & High-Engagement Signals** — what high-engagement tweets reveal\n"
        "7. **Recommendations** — 3–5 actionable suggestions for the team\n\n"
        "Be specific and grounded in the data. Quote actual tweets where helpful."
    )

    chat_key = f"chat_{active}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    context = _build_context(df_chat, config)

    col_btn, col_clear = st.columns([3, 1])
    with col_btn:
        gen_report = st.button("📝 Generate Report", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state[chat_key] = []
            st.rerun()

    if gen_report:
        with st.spinner("Claude is reading your data and writing the report…"):
            import anthropic
            client = anthropic.AnthropicBedrock(aws_region=BEDROCK_REGION)
            messages = [{
                "role": "user",
                "content": (
                    f"Here is the tweet dataset:\n\n<dataset>\n{context}\n</dataset>\n\n"
                    f"{REPORT_PROMPT}"
                )
            }]
            response = client.messages.create(
                model=BEDROCK_MODEL,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            report_text = response.content[0].text
            st.session_state[chat_key].append({
                "role": "user",
                "content": f"<dataset>\n{context}\n</dataset>\n\n{REPORT_PROMPT}"
            })
            st.session_state[chat_key].append({
                "role": "assistant",
                "content": report_text,
            })
            st.rerun()

    if st.session_state[chat_key]:
        for msg in st.session_state[chat_key]:
            role    = msg["role"]
            content = msg["content"]
            if role == "user" and content.startswith("<dataset>"):
                parts   = content.split("</dataset>\n\n", 1)
                display = parts[1] if len(parts) > 1 else content
                if display == REPORT_PROMPT:
                    display = "*(Generated report from full dataset)*"
            else:
                display = content
            with st.chat_message(role):
                st.markdown(display)

        last_assistant = next(
            (m["content"] for m in reversed(st.session_state[chat_key]) if m["role"] == "assistant"),
            None
        )
        if last_assistant:
            st.download_button(
                "⬇️ Download latest response as Markdown",
                last_assistant.encode("utf-8"),
                file_name=f"{active}_report.md",
                mime="text/markdown",
            )
    else:
        st.info("Click **Generate Report** for a full analysis, or type a question below to ask about your data directly.")

    st.divider()

    user_input = st.chat_input("Ask anything about your tweet data…")

    if user_input:
        if not st.session_state[chat_key]:
            st.session_state[chat_key].append({
                "role": "user",
                "content": f"Here is the tweet dataset for reference:\n\n<dataset>\n{context}\n</dataset>\n\nAcknowledge briefly that you have the data, then answer: {user_input}"
            })
        else:
            st.session_state[chat_key].append({
                "role": "user",
                "content": user_input,
            })

        with st.spinner("Thinking…"):
            import anthropic
            client = anthropic.AnthropicBedrock(aws_region=BEDROCK_REGION)
            response = client.messages.create(
                model=BEDROCK_MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=st.session_state[chat_key],
            )
            reply = response.content[0].text

        st.session_state[chat_key].append({
            "role": "assistant",
            "content": reply,
        })
        st.rerun()
