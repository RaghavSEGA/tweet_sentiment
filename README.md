# Tweet Sentiment Dashboard

A general-purpose tweet sentiment analysis tool using Twitter API v2 and Claude.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## First-time setup

1. Copy the secrets template:
   ```bash
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Edit `.streamlit/secrets.toml` and fill in your keys:
   ```toml
   TWITTER_BEARER_TOKEN = "your-twitter-bearer-token"   # developer.twitter.com
   ANTHROPIC_API_KEY    = "your-anthropic-api-key"      # console.anthropic.com
   ```
3. Run the app and create a new **Project** in the sidebar

> **Never commit `secrets.toml` to git** — it's already in `.gitignore`.
> When deploying to Streamlit Cloud, add the keys via the Secrets UI instead.

## Workflow

### 1. Configure
- Add keywords and/or Twitter handles to track
- Define custom sentiment categories with descriptions
- Example categories for STH:
  - `Biscus/Scandal-Related` — *Mentions of Kagawa, Sex Pest, retcon, controversy*
  - `Excitement/Anticipation` — *Hype, looking forward to, can't wait*
  - `Concern/Skepticism` — *RGG last chance, worried, disappointed*
  - `Yokoyama/Staff` — *Comments about studio staff or director*
  - `Gameplay/Mechanics` — *Combat, systems, features discussion*
  - `General Positive` — *Praise without specific topic*
  - `General Negative` — *Criticism without specific topic*

### 2. Collect
- Set a date range and max tweet count
- Click **Fetch Tweets** — data is stored locally in SQLite

### 3. Analyze
- Review your categories
- Click **Run Analysis** — Claude classifies each tweet in batches
- Progress is saved; you can resume if interrupted

### 4. Dashboard
- View sentiment breakdown, category distribution, time trends
- Word clouds for positive vs negative tweets
- Top positive and negative tweets surfaced automatically

### 5. Export
- Download full dataset as CSV or Excel for further analysis

## Notes

- Twitter API v2 free tier only allows searching the **last 7 days** of tweets
- Basic tier ($100/mo) unlocks full archive search
- Tweets are deduplicated and stored locally — re-fetching won't create duplicates
- Analysis can be run incrementally on new tweets without re-analyzing old ones
