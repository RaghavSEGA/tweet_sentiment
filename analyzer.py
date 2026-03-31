import anthropic
import json
import re
import time


SYSTEM_PROMPT = """You are a social media sentiment analyst specializing in gaming communities.
You will be given a batch of tweets and a list of custom categories defined by the user.
For each tweet, return a JSON array where each element has:
  - "id": the tweet id (string)
  - "sentiment": one of "positive", "negative", or "neutral"
  - "score": float from -1.0 (most negative) to 1.0 (most positive)
  - "category": the best matching category name from the provided list, or "Other" if none fit
  - "reasoning": one concise sentence explaining your classification

Respond ONLY with a valid JSON array. No markdown, no preamble, no trailing text."""


def build_user_prompt(tweets: list[dict], categories: list[dict]) -> str:
    cat_block = "\n".join(
        [f'- "{c["name"]}": {c["description"]}' for c in categories]
    )
    tweets_block = "\n\n".join(
        [f'[{t["id"]}] {t["text"]}' for t in tweets]
    )
    return f"""## Categories
{cat_block}

## Tweets to classify
{tweets_block}

Return a JSON array with one object per tweet."""


def analyze_tweets_batch(
    tweets: list[dict],
    categories: list[dict],
    api_key: str,
    max_retries: int = 3,
) -> list[dict]:
    """
    Send a batch of tweets to Claude for sentiment + category classification.
    Retries up to max_retries times on connection errors with exponential backoff.
    """
    if not tweets:
        return []

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_user_prompt(tweets, categories)

    last_error = None
    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = message.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            results = json.loads(raw)

            result_map = {str(r["id"]): r for r in results}
            updated = []
            for tweet in tweets:
                tid = str(tweet["id"])
                if tid in result_map:
                    r = result_map[tid]
                    tweet["sentiment"] = r.get("sentiment", "neutral")
                    tweet["score"] = r.get("score", 0.0)
                    tweet["category"] = r.get("category", "Other")
                    tweet["reasoning"] = r.get("reasoning", "")
                updated.append(tweet)

            return updated

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            last_error = e
            wait = 2 ** attempt  # 1s, 2s, 4s
            time.sleep(wait)
        except anthropic.RateLimitError as e:
            last_error = e
            time.sleep(60)  # Anthropic rate limits reset quickly
        except Exception as e:
            raise  # Don't retry on unexpected errors

    raise ConnectionError(f"Failed after {max_retries} attempts: {last_error}")