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
import hashlib
import hmac
import random
import base64

from collector import fetch_tweets
from analyzer import analyze_tweets_batch
from storage import init_db, save_tweets, load_tweets, get_projects, save_project, load_project_config

init_db()

st.set_page_config(
    page_title="Tweet Sentiment Tool",
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

# ── OTP Auth helpers ──────────────────────────────────────────────────────────
ALLOWED_DOMAIN     = "@segaamerica.com"
OTP_EXPIRY_SECS    = 600
COOKIE_EXPIRY_DAYS = 7
COOKIE_NAME        = "sentiment_tool_auth"

def _send_otp(email: str, code: str) -> bool:
    try:
        import boto3
        from botocore.exceptions import ClientError
        ses = boto3.client(
            "ses",
            region_name=st.secrets.get("AWS_SES_REGION", "us-east-1"),
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        ses.send_email(
            Source=st.secrets.get("EMAIL_FROM", "noreply@segaamerica.com"),
            Destination={"ToAddresses": [email]},
            Message={
                "Subject": {"Data": "Tweet Sentiment Tool — Your verification code", "Charset": "UTF-8"},
                "Body": {
                    "Text": {
                        "Data": f"Your verification code is: {code}\n\nExpires in 10 minutes.",
                        "Charset": "UTF-8",
                    },
                    "Html": {
                        "Data": f"""
                        <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;padding:32px 24px;">
                          <div style="font-size:22px;font-weight:900;letter-spacing:0.1em;color:#4080ff;margin-bottom:4px;">SEGA</div>
                          <div style="font-size:14px;color:#444;margin-bottom:28px;">Tweet Sentiment Tool</div>
                          <div style="font-size:14px;color:#222;margin-bottom:16px;">Your verification code is:</div>
                          <div style="font-size:42px;font-weight:900;letter-spacing:0.18em;color:#1a1a2e;
                                      background:#f0f4ff;border-radius:8px;padding:18px 24px;
                                      display:inline-block;margin-bottom:24px;">{code}</div>
                          <div style="font-size:12px;color:#888;">
                            This code expires in 10 minutes.<br>
                            If you didn't request this, you can safely ignore this email.
                          </div>
                        </div>""",
                        "Charset": "UTF-8",
                    },
                },
            },
        )
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def _sign_cookie(email: str) -> str:
    secret = st.secrets.get("COOKIE_SIGNING_KEY", "fallback-change-this")
    expiry = int(time.time()) + (COOKIE_EXPIRY_DAYS * 86400)
    payload = f"{email}|{expiry}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()

def _verify_cookie(token: str):
    try:
        secret = st.secrets.get("COOKIE_SIGNING_KEY", "fallback-change-this")
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        email, expiry_str, sig = decoded.rsplit("|", 2)
        payload = f"{email}|{expiry_str}"
        expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        if int(time.time()) > int(expiry_str):
            return None
        return email
    except Exception:
        return None

# ── Cookie manager ────────────────────────────────────────────────────────────
try:
    import extra_streamlit_components as stx
    _cookie_manager  = stx.CookieManager(key="auth_cookies")
    _existing_cookie = _cookie_manager.get(COOKIE_NAME)
except Exception:
    _cookie_manager  = None
    _existing_cookie = None

_cookie_email = _verify_cookie(_existing_cookie) if _existing_cookie else None

# ── Auth state init ───────────────────────────────────────────────────────────
for _k, _v in [
    ("auth_verified", False), ("auth_email", ""),
    ("otp_code", ""),  ("otp_email", ""), ("otp_expiry", 0),
    ("otp_sent", False), ("otp_attempts", 0),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

if _cookie_email and not st.session_state.auth_verified:
    st.session_state.auth_verified = True
    st.session_state.auth_email    = _cookie_email

# ── Login gate ────────────────────────────────────────────────────────────────
if not st.session_state.auth_verified:
    st.markdown("""
    <style>
    .auth-wrap {
        max-width:420px; margin:5rem auto; padding:2.5rem 2.5rem 2rem;
        background:var(--surface); border:1px solid var(--border);
        border-top:3px solid var(--blue); border-radius:0 0 10px 10px;
    }
    .auth-logo  { font-family:'Arial Black',sans-serif; font-size:1.6rem; font-weight:900;
                  letter-spacing:0.12em; color:var(--blue) !important; margin-bottom:0.2rem; }
    .auth-title { font-size:1rem; font-weight:700; color:var(--text) !important; margin-bottom:0.25rem; }
    .auth-sub   { font-size:0.8rem; color:var(--muted) !important; margin-bottom:1.5rem; }
    .auth-note  { font-size:0.72rem; color:var(--muted) !important; margin-top:1rem;
                  text-align:center; line-height:1.5; }
    </style>
    """, unsafe_allow_html=True)

    _lc, _mc, _rc = st.columns([1, 2, 1])
    with _mc:
        st.markdown("""
        <div class="auth-wrap">
          <div class="auth-logo">SEGA</div>
          <div class="auth-title">Tweet Sentiment Tool</div>
          <div class="auth-sub">Sign in with your SEGA America email</div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.otp_sent:
            _email_input = st.text_input(
                "Email address", placeholder="you@segaamerica.com",
                label_visibility="hidden", key="auth_email_input",
            )
            if st.button("Send verification code", use_container_width=True):
                if _email_input and not _email_input.strip().lower().endswith(ALLOWED_DOMAIN):
                    st.error(f"Access restricted to {ALLOWED_DOMAIN} addresses.")
                elif _email_input:
                    _code = str(random.randint(100000, 999999))
                    if _send_otp(_email_input.strip().lower(), _code):
                        st.session_state.otp_code     = _code
                        st.session_state.otp_email    = _email_input.strip().lower()
                        st.session_state.otp_expiry   = time.time() + OTP_EXPIRY_SECS
                        st.session_state.otp_sent     = True
                        st.session_state.otp_attempts = 0
                        st.rerun()
        else:
            st.info(f"Code sent to **{st.session_state.otp_email}** — check your inbox.")
            _code_input = st.text_input(
                "6-digit code", placeholder="123456",
                label_visibility="hidden", max_chars=6, key="auth_code_input",
            )
            if st.button("Verify code", use_container_width=True) and _code_input:
                if st.session_state.otp_attempts >= 5:
                    st.error("Too many attempts. Please request a new code.")
                    st.session_state.otp_sent = False
                elif time.time() > st.session_state.otp_expiry:
                    st.error("Code has expired. Please request a new one.")
                    st.session_state.otp_sent = False
                elif _code_input.strip() != st.session_state.otp_code:
                    st.session_state.otp_attempts += 1
                    _rem = 5 - st.session_state.otp_attempts
                    st.error(f"Incorrect code. {_rem} attempt{'s' if _rem != 1 else ''} remaining.")
                else:
                    st.session_state.auth_verified = True
                    st.session_state.auth_email    = st.session_state.otp_email
                    st.session_state.otp_code      = ""
                    if _cookie_manager:
                        _token = _sign_cookie(st.session_state.auth_email)
                        _cookie_manager.set(COOKIE_NAME, _token, expires_at=None, key="set_auth_cookie")
                    st.rerun()

            if st.button("← Use a different email", key="auth_back"):
                st.session_state.otp_sent = False
                st.session_state.otp_code = ""
                st.rerun()

        st.markdown(
            '<div class="auth-note">Restricted to @segaamerica.com addresses only.<br>'
            'Codes expire after 10 minutes.</div>',
            unsafe_allow_html=True,
        )

    st.stop()

# Shorthand for the signed-in user's email — passed to every storage call
OWNER = st.session_state.auth_email

# ── Load API keys from st.secrets ────────────────────────────────────────────
try:
    TWITTER_BEARER = st.secrets["TWITTER_BEARER_TOKEN"]
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError as e:
    st.error(f"Missing secret: {e}. Add it to `.streamlit/secrets.toml`.")
    st.code("""
# .streamlit/secrets.toml
TWITTER_BEARER_TOKEN = "your-twitter-bearer-token"
ANTHROPIC_API_KEY    = "your-anthropic-api-key"
""", language="toml")
    st.stop()

# ── Sidebar: Project Management ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:.6rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;'
        f'color:#5a5f82;margin-bottom:.15rem;">SEGA America</div>'
        f'<div style="font-size:.75rem;font-weight:600;color:#b8bcd4;margin-bottom:.5rem;">'
        f'{st.session_state.auth_email}</div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign out", key="sign_out_btn"):
        if _cookie_manager:
            _cookie_manager.delete(COOKIE_NAME, key="delete_auth_cookie")
        for _k in ["auth_verified","auth_email","otp_sent","otp_code","otp_email","otp_expiry","otp_attempts"]:
            st.session_state[_k] = False if _k == "auth_verified" else ""
        st.rerun()
    st.markdown("<hr style='border:none;border-top:1px solid #232640;margin:.5rem 0 .75rem;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.85rem;font-weight:700;color:#eef0fa;margin-bottom:.25rem;">🎮 Tweet Sentiment Tool</div>', unsafe_allow_html=True)
    st.caption("Collect, analyse, and visualise Twitter sentiment for any game or topic.")
    st.divider()

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown("""
**Follow the tabs left to right:**

1. **⚙️ Configure** — Set keywords, handles, and custom categories. Categories tell Claude how to group tweets beyond positive/negative/neutral.

2. **📥 Collect** — Pull tweets from the Twitter API. Tweets are saved locally so you can collect incrementally without duplicates.

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

    # Read the project from the URL if present and valid
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
            "Twitter handles to track (one per line, with @)",
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
- **Name** — Short label, shown in charts. Keep it under ~30 characters. Example: `Biscus/Scandal-Related`
- **Description** — This is what Claude actually reads. Be specific. List example phrases, names, topics, or situations that belong here. Vague descriptions lead to miscategorisation.

> **Example of a weak description:** *"Negative tweets"*
> **Example of a strong description:** *"Tweets referencing Kagawa, the 'Sex Pest' allegations, retcons to the Biscus storyline, or controversy about the game's writing — regardless of whether the tweet is positive or negative in tone"*

If a tweet doesn't clearly fit any category, Claude will assign it **Other**.
        """)
        st.divider()

        categories = config.get("categories", [])

        # Dynamic category editor
        if "categories_state" not in st.session_state or st.session_state.get("cat_project") != active:
            st.session_state["categories_state"] = categories.copy()
            st.session_state["cat_project"] = active

        cats = st.session_state["categories_state"]

        if cats:
            hc1, hc2, hc3 = st.columns([2, 4, 1])
            hc1.caption("**Category name**")
            hc2.caption("**Description — what belongs here? Give Claude examples.**")
            hc3.caption("")

        for i, cat in enumerate(cats):
            c1, c2, c3 = st.columns([2, 4, 1])
            with c1:
                cats[i]["name"] = st.text_input(f"Name##{i}", value=cat["name"],
                                                 placeholder="e.g. Excitement/Hype",
                                                 label_visibility="collapsed")
            with c2:
                cats[i]["description"] = st.text_input(f"Desc##{i}", value=cat["description"],
                                                        placeholder="e.g. Tweets expressing anticipation, excitement, hype — 'can't wait', 'looks amazing', day-one purchase mentions",
                                                        label_visibility="collapsed")
            with c3:
                if st.button("✕", key=f"del_{i}"):
                    cats.pop(i)
                    st.rerun()

        if st.button("+ Add Category"):
            cats.append({"name": "", "description": ""})
            st.rerun()

    st.divider()
    if st.button("💾 Save Configuration", type="primary"):
        updated = {
            "name": active,
            "keywords": [k.strip() for k in keywords_raw.splitlines() if k.strip()],
            "handles": [h.strip() for h in handles_raw.splitlines() if h.strip()],
            "categories": [c for c in cats if c["name"]],
        }
        save_project(updated, OWNER)
        st.session_state["categories_state"] = updated["categories"]
        config = updated
        st.success("Configuration saved!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COLLECT
# ══════════════════════════════════════════════════════════════════════════════
with tab_collect:
    st.header("📥 Collect Tweets")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Fetches tweets from the Twitter API v2 that match your configured keywords and handles.

**Settings**
- **Date range** — The window to search within. See the API tier note below for limits.
- **Max tweets** — Cap on how many tweets to retrieve per run. Twitter returns up to 100 per page; the tool paginates automatically.
- **Include retweets** — Retweets add volume but are often noise. Recommended: leave unchecked unless you specifically want to track how content spreads.

**How collection works**
All your keywords and handles are combined into a single Twitter API query using OR logic — so a tweet only needs to match *one* of your terms to be collected. Results are saved to a local SQLite database. Re-running collection on the same date range will **not** create duplicates.

**Twitter API date limits**
The free tier restricts searches to a rolling **7-day window** measured to the exact second — not midnight of 7 days ago, but the precise moment 7 days prior to now. If you select a start date of "7 days ago", the tool automatically adjusts `start_time` forward to the earliest second Twitter will accept, so you won't see a 400 error. Dates earlier than 7 days ago are simply unavailable on the free tier; upgrade to Basic ($100/mo) for full archive access back to 2006.

> ⚠️ If you get a 403 error, your Twitter developer app may not have the correct permissions. Ensure it has **Read** access and that your Bearer Token is from an app with access to the v2 search endpoint.
        """)

    bearer = TWITTER_BEARER

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=7))
        max_results = st.slider("Max tweets per query", 10, 500, 100, step=10)
    with col2:
        end_date = st.date_input("To", value=datetime.now())
        include_retweets = st.checkbox("Include retweets", value=False)

    with st.expander("⏱️ Twitter API limits — read this before fetching", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
**🗓️ Date range (7-day rolling window)**

The free Twitter API tier only allows searching tweets from the **last 7 days**, measured to the exact second — not midnight of 7 days ago. This tool automatically clamps your start date forward if it falls outside the allowed window, so you won't see a 400 error.

To access older tweets you need the **Basic tier ($100/mo)**, which unlocks the full archive back to 2006.

---

**📄 Page size**

Twitter returns tweets in pages of up to 100. Each page = one API request. If you set Max Tweets to 200, that's at least 2 requests.
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

If the rate limit is hit mid-fetch, the tool will **stop and save whatever it collected so far** rather than hanging. You'll see a warning telling you how many tweets were saved.

**You can simply fetch again after 15 minutes** — duplicates are automatically skipped, so your existing tweets are safe and the next run will pick up where it left off.

---

**💡 Tip:** Keep Max Tweets at 100 or below during testing to stay within one request per run.
            """)

    keywords = config.get("keywords", [])
    handles = config.get("handles", [])

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
            progress_bar = st.progress(0, text="Connecting to Twitter API...")
            count_display = st.empty()
            latest_tweet = st.empty()
            status = st.empty()
            api_response_box = st.empty()

            # Show debug info so we can verify the query being sent
            with st.expander("🔍 Debug — query sent to Twitter", expanded=True):
                from collector import build_query, compute_date_range
                debug_query = build_query(keywords, handles, include_retweets)
                debug_start, debug_end = compute_date_range(start_date, end_date)
                st.code(f"Query:      {debug_query}\nStart time: {debug_start}\nEnd time:   {debug_end}", language="text")

            tweets = []
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

                progress_bar.progress(0, text="⏳ Waiting for Twitter API response...")

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

                    ts = item.get("created_at", "")[:16].replace("T", " ")
                    handle = item.get("username") or item.get("author_id", "unknown")
                    preview = item["text"][:120] + ("…" if len(item["text"]) > 120 else "")
                    latest_tweet.info(f"**@{handle}** · {ts}\n\n{preview}")

                progress_bar.empty()
                count_display.empty()
                latest_tweet.empty()

                # Show API result
                if api_error:
                    code = api_error["_error"]
                    msg = api_error["_message"]
                    if code == "429":
                        reset = api_error.get("_reset_time")
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
                        api_response_box.error(f"**API response: Server Error** — Twitter is having issues, try again shortly.\n\n`{msg}`")
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

        # Last fetch time + rate limit countdown
        if "last_fetch_time" in st.session_state:
            last = st.session_state["last_fetch_time"]
            elapsed_min = int((datetime.now() - last).total_seconds() / 60)
            st.caption(f"Last fetch attempt: {last.strftime('%H:%M:%S')} ({elapsed_min} min ago)")

        if "rate_limit_hit" in st.session_state:
            rl_elapsed = (datetime.now() - st.session_state["rate_limit_hit"]).total_seconds()
            rl_remaining = max(0, 15 * 60 - rl_elapsed)
            if rl_remaining > 0:
                mins = int(rl_remaining // 60)
                secs = int(rl_remaining % 60)
                st.warning(f"🚦 Rate limit active — estimated reset in **{mins}m {secs}s** (refresh page to update)")
            else:
                st.success("🟢 Rate limit window has passed — safe to fetch again.")


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
        st.stop()

    unanalyzed = df_raw[df_raw["sentiment"].isna()] if "sentiment" in df_raw.columns else df_raw
    total = len(df_raw)
    done = total - len(unanalyzed)

    st.metric("Total tweets", total)
    st.metric("Already analyzed", done)
    st.metric("Pending", len(unanalyzed))

    batch_size = st.slider("Batch size (tweets per Claude call)", 5, 50, 20)

    cat_preview = "\n".join([f"- **{c['name']}**: {c['description']}" for c in categories])
    with st.expander("Preview categories sent to Claude"):
        st.markdown(cat_preview)

    if len(unanalyzed) > 0:
        if st.button("▶️ Run Analysis", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            results = []
            failed_batches = []

            batches = [unanalyzed.iloc[i:i+batch_size] for i in range(0, len(unanalyzed), batch_size)]
            for idx, batch in enumerate(batches):
                status.text(f"Analyzing batch {idx+1}/{len(batches)}...")
                try:
                    analyzed = analyze_tweets_batch(
                        tweets=batch.to_dict("records"),
                        categories=categories,
                        api_key=api_key,
                    )
                    results.extend(analyzed)
                except ConnectionError as e:
                    st.error(f"Batch {idx+1} failed after retries: {e}")
                    failed_batches.append(idx + 1)
                except Exception as e:
                    st.error(f"Batch {idx+1} failed: {e}")
                    failed_batches.append(idx + 1)
                progress.progress((idx + 1) / len(batches))

            if results:
                save_tweets(active, OWNER, results, update=True)
                if failed_batches:
                    status.warning(f"⚠️ Partial complete — {len(results)} tweets saved. Batches {failed_batches} failed. Re-run to retry failed batches.")
                else:
                    status.text("✅ Analysis complete!")
                    st.balloons()
    else:
        st.success("All tweets have been analyzed!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.header("📊 Sentiment Dashboard")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Visualises the results of Claude's analysis for the active project.

**Filters** (top of page) — Narrow the view by sentiment and/or category. Filters apply to all charts and tweet cards simultaneously.

**Charts**
- **Sentiment Breakdown** — Pie chart of positive / negative / neutral share across all filtered tweets.
- **Category Distribution** — Bar chart showing how many tweets fall into each of your custom categories.
- **Sentiment Over Time** — Line chart showing how sentiment volume changes day by day. Useful for spotting spikes around announcements or controversies.
- **Word Clouds** — The most frequent words in positive vs negative tweets. Common stop words are filtered out automatically.

**Top / Bottom Tweets**
The 5 highest-scoring positive and 5 lowest-scoring negative tweets, surfaced automatically. Useful for pulling representative quotes for a report.

> 💡 **Tip:** Use the category filter to focus on a specific topic — e.g. filter to *Biscus/Scandal-Related* only to see how that conversation breaks down in isolation.
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
            all_cats = df["category"].dropna().unique().tolist() if "category" in df.columns else []
            cat_filter = st.multiselect("Category", all_cats, default=all_cats)

    mask = df["sentiment"].isin(sentiment_filter)
    if cat_filter and "category" in df.columns:
        mask &= df["category"].isin(cat_filter)
    df_f = df[mask]

    # ── Top metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Tweets", len(df_f))
    m2.metric("Positive", (df_f["sentiment"] == "positive").sum())
    m3.metric("Negative", (df_f["sentiment"] == "negative").sum())
    m4.metric("Neutral", (df_f["sentiment"] == "neutral").sum())

    # Engagement-weighted sentiment score
    if "like_count" in df_f.columns and "retweet_count" in df_f.columns:
        df_f["engagement"] = df_f["like_count"].fillna(0) + df_f["retweet_count"].fillna(0) * 2
        total_engagement = df_f["engagement"].sum()
        if total_engagement > 0:
            weighted_score = (df_f["score"].fillna(0) * df_f["engagement"]).sum() / total_engagement
        else:
            weighted_score = df_f["score"].fillna(0).mean()
    else:
        weighted_score = df_f["score"].fillna(0).mean()
    score_label = "🟢 Positive" if weighted_score > 0.1 else "🔴 Negative" if weighted_score < -0.1 else "⚪ Neutral"
    m5.metric("Engagement-Weighted Score", f"{weighted_score:.2f}", help=f"{score_label} — weighted by likes + 2× retweets")

    st.divider()

    # ── Row 1: Sentiment breakdown + Category heatmap
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment Breakdown")
        sent_counts = df_f["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        color_map = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#94a3b8"}
        fig_pie = px.pie(sent_counts, names="sentiment", values="count",
                         color="sentiment", color_discrete_map=color_map)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("Sentiment by Category")
        if "category" in df_f.columns:
            heat_df = df_f.groupby(["category", "sentiment"]).size().reset_index(name="count")
            heat_pivot = heat_df.pivot(index="category", columns="sentiment", values="count").fillna(0)
            # Ensure all sentiment columns exist
            for s in ["positive", "negative", "neutral"]:
                if s not in heat_pivot.columns:
                    heat_pivot[s] = 0
            heat_pivot = heat_pivot[["positive", "neutral", "negative"]]
            fig_heat = px.imshow(
                heat_pivot,
                color_continuous_scale=[[0, "#ef4444"], [0.5, "#f8fafc"], [1, "#22c55e"]],
                text_auto=True,
                aspect="auto",
                labels={"color": "Tweet count"},
            )
            fig_heat.update_coloraxes(colorbar_title="Count")
            st.plotly_chart(fig_heat, use_container_width=True)

    # ── Row 2: Sentiment over time + Category trend over time
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Sentiment Over Time")
        df_f["date"] = df_f["created_at"].dt.date
        time_df = df_f.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig_line = px.line(time_df, x="date", y="count", color="sentiment",
                           color_discrete_map=color_map)
        st.plotly_chart(fig_line, use_container_width=True)

    with c4:
        st.subheader("Category Volume Over Time")
        if "category" in df_f.columns:
            cat_time_df = df_f.groupby(["date", "category"]).size().reset_index(name="count")
            fig_cat_time = px.line(cat_time_df, x="date", y="count", color="category")
            fig_cat_time.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.4))
            st.plotly_chart(fig_cat_time, use_container_width=True)

    # ── Row 3: Engagement-weighted sentiment by category + Hourly heatmap
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Engagement-Weighted Score by Category")
        if "category" in df_f.columns and "engagement" in df_f.columns:
            eng_df = df_f.groupby("category").apply(
                lambda g: (g["score"].fillna(0) * g["engagement"]).sum() / g["engagement"].sum()
                if g["engagement"].sum() > 0 else g["score"].fillna(0).mean()
            ).reset_index(name="weighted_score")
            eng_df = eng_df.sort_values("weighted_score")
            eng_df["color"] = eng_df["weighted_score"].apply(
                lambda x: "#22c55e" if x > 0.1 else "#ef4444" if x < -0.1 else "#94a3b8"
            )
            fig_eng = px.bar(eng_df, x="weighted_score", y="category", orientation="h",
                             color="weighted_score",
                             color_continuous_scale=[[0, "#ef4444"], [0.5, "#94a3b8"], [1, "#22c55e"]],
                             range_color=[-1, 1])
            fig_eng.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_eng.update_coloraxes(showscale=False)
            st.plotly_chart(fig_eng, use_container_width=True)

    with c6:
        st.subheader("Hourly Activity Heatmap")
        df_f["hour"] = df_f["created_at"].dt.hour
        df_f["day"] = df_f["created_at"].dt.day_name()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hourly_df = df_f.groupby(["day", "hour"]).size().reset_index(name="count")
        hourly_pivot = hourly_df.pivot(index="day", columns="hour", values="count").fillna(0)
        # Reindex to ensure all days and hours are present
        hourly_pivot = hourly_pivot.reindex(
            [d for d in day_order if d in hourly_pivot.index]
        )
        hourly_pivot = hourly_pivot.reindex(columns=range(24), fill_value=0)
        fig_hourly = px.imshow(
            hourly_pivot,
            color_continuous_scale="Blues",
            labels={"x": "Hour (UTC)", "y": "Day", "color": "Tweets"},
            aspect="auto",
            text_auto=True,
        )
        fig_hourly.update_xaxes(tickvals=list(range(0, 24, 3)),
                                 ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])
        st.plotly_chart(fig_hourly, use_container_width=True)

    # ── Word cloud
    st.subheader("Word Cloud")
    wc_col1, wc_col2 = st.columns(2)
    for col, sent_label in zip([wc_col1, wc_col2], ["positive", "negative"]):
        with col:
            st.caption(f"{sent_label.capitalize()} tweets")
            texts = " ".join(df_f[df_f["sentiment"] == sent_label]["text"].fillna("").tolist())
            if texts.strip():
                wc = WordCloud(width=600, height=300, background_color="white",
                               colormap="Greens" if sent_label == "positive" else "Reds").generate(texts)
                fig_wc, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info(f"No {sent_label} tweets to display.")

    # ── Top tweets (fixed to use @username)
    st.subheader("🔝 Top Positive Tweets")
    pos = df_f[df_f["sentiment"] == "positive"].sort_values("score", ascending=False).head(5)
    for _, row in pos.iterrows():
        with st.container(border=True):
            handle = row.get("username") or row.get("author_id", "unknown")
            st.markdown(f"**@{handle}** · {row['created_at'].strftime('%b %d, %H:%M UTC')}")
            st.write(row["text"])
            likes = int(row.get("like_count", 0) or 0)
            rts = int(row.get("retweet_count", 0) or 0)
            st.caption(f"Category: {row.get('category', '—')} · Score: {row.get('score', '—')} · ❤️ {likes} · 🔁 {rts}")

    st.subheader("🔻 Top Negative Tweets")
    neg = df_f[df_f["sentiment"] == "negative"].sort_values("score").head(5)
    for _, row in neg.iterrows():
        with st.container(border=True):
            handle = row.get("username") or row.get("author_id", "unknown")
            st.markdown(f"**@{handle}** · {row['created_at'].strftime('%b %d, %H:%M UTC')}")
            st.write(row["text"])
            likes = int(row.get("like_count", 0) or 0)
            rts = int(row.get("retweet_count", 0) or 0)
            st.caption(f"Category: {row.get('category', '—')} · Score: {row.get('score', '—')} · ❤️ {likes} · 🔁 {rts}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.header("💾 Export Data")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
Download the full tweet dataset for this project — including all raw tweet data and Claude's analysis results.

**Columns in the export**
| Column | Description |
|---|---|
| `id` | Twitter tweet ID |
| `created_at` | Tweet timestamp (UTC) |
| `username` / `name` | Author handle and display name |
| `text` | Full tweet text |
| `sentiment` | `positive`, `negative`, or `neutral` |
| `score` | Float −1.0 to +1.0 |
| `category` | Your custom category label |
| `reasoning` | Claude's one-sentence explanation |
| `like_count`, `retweet_count`, etc. | Engagement metrics from Twitter |

**Formats**
- **CSV** — Best for Excel, Google Sheets, or further data processing
- **Excel (.xlsx)** — Pre-formatted spreadsheet, useful for sharing reports directly

> 💡 **Tip:** To create a filtered export, use the Dashboard filters first to identify the subset you want, then come back here — or filter the downloaded CSV/Excel manually using your preferred tool.
        """)

    df_exp = load_tweets(active, OWNER)
    if df_exp.empty:
        st.info("No data to export yet.")
        st.stop()

    st.dataframe(df_exp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        csv = df_exp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, f"{active}_tweets.csv", "text/csv")
    with col2:
        excel_buf = io.BytesIO()
        df_exp.to_excel(excel_buf, index=False)
        st.download_button("⬇️ Download Excel", excel_buf.getvalue(),
                           f"{active}_tweets.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT & CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.header(f"Report & Chat: {active}")
    with st.expander("ℹ️ About this tab", expanded=False):
        st.markdown("""
**Generate Report** — Claude reads all your analysed tweets and writes a structured
executive report covering sentiment trends, top issues, standout praise, and
category breakdowns.

**Chat** — Ask follow-up questions about the data. Claude has the full dataset
in context so you can drill into specifics: *"What are the top complaints about
the tutorial?"*, *"Summarise negative tweets from influencers"*, etc.

> The dataset is passed in full on every message, so answers are grounded in
> your actual tweets — not generic knowledge.
        """)

    df_chat = load_tweets(active, OWNER)

    if df_chat.empty or df_chat["sentiment"].isna().all():
        st.info("No analysed tweets yet. Run the Analyze tab first.")
        st.stop()

    # ── Build a compact dataset summary to pass as context ──────────────────
    def _build_context(df: "pd.DataFrame", config: dict) -> str:
        total      = len(df)
        analysed   = df["sentiment"].notna().sum()
        pos        = (df["sentiment"] == "positive").sum()
        neg        = (df["sentiment"] == "negative").sum()
        neu        = (df["sentiment"] == "neutral").sum()
        avg_score  = df["score"].mean() if "score" in df.columns else None
        categories = df["category"].value_counts().to_dict() if "category" in df.columns else {}

        # Top 40 tweets by engagement (like + retweet) for richness
        df_sorted = df.copy()
        df_sorted["engagement"] = df_sorted.get("like_count", 0) + df_sorted.get("retweet_count", 0)
        top_tweets = df_sorted.nlargest(40, "engagement")[
            ["username", "text", "sentiment", "category", "score", "reasoning",
             "like_count", "retweet_count"]
        ].to_dict("records")

        # Random sample of 60 more for breadth
        rest = df_sorted.drop(df_sorted.nlargest(40, "engagement").index)
        sample = rest.sample(min(60, len(rest)), random_state=42) if len(rest) else rest
        sample_tweets = sample[
            ["username", "text", "sentiment", "category", "score", "reasoning"]
        ].to_dict("records")

        project_name = active
        keywords     = config.get("keywords", [])
        handles      = config.get("handles", [])
        cat_defs     = config.get("categories", [])

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

    # ── Session state for chat ───────────────────────────────────────────────
    chat_key = f"chat_{active}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []   # list of {"role": ..., "content": ...}

    context = _build_context(df_chat, config)

    # ── Generate Report button ───────────────────────────────────────────────
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
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Here is the tweet dataset:\n\n<dataset>\n{context}\n</dataset>\n\n"
                        f"{REPORT_PROMPT}"
                    )
                }
            ]
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            report_text = response.content[0].text
            # Save both turns to chat history so follow-ups have full context
            st.session_state[chat_key].append({
                "role": "user",
                "content": (
                    f"<dataset>\n{context}\n</dataset>\n\n{REPORT_PROMPT}"
                )
            })
            st.session_state[chat_key].append({
                "role": "assistant",
                "content": report_text,
            })
            st.rerun()

    # ── Render chat history ──────────────────────────────────────────────────
    if st.session_state[chat_key]:
        for msg in st.session_state[chat_key]:
            role = msg["role"]
            # Don't render the raw dataset blob — show a tidy label instead
            content = msg["content"]
            if role == "user" and content.startswith("<dataset>"):
                # Strip the dataset preamble for display; show only the question part
                parts = content.split("</dataset>\n\n", 1)
                display = parts[1] if len(parts) > 1 else content
                if display == REPORT_PROMPT:
                    display = "*(Generated report from full dataset)*"
            else:
                display = content

            with st.chat_message(role):
                st.markdown(display)

        # ── Download report button (shown when there's content) ─────────────
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

    # ── Chat input ───────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything about your tweet data…")

    if user_input:
        # If no prior history, inject the dataset as the first user message silently
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
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
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