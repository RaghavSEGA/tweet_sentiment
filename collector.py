import tweepy
from datetime import datetime, timedelta, timezone


def build_query(keywords: list[str], handles: list[str], include_retweets: bool = False) -> str:
    parts = []
    for kw in keywords:
        # Use unquoted for single words, quoted for multi-word phrases
        words = kw.strip().split()
        if len(words) == 1:
            parts.append(words[0])
        else:
            parts.append(f'"{kw.strip()}"')
    for handle in handles:
        h = handle.lstrip("@")
        parts.append(f"(@{h} OR to:{h})")
    if not parts:
        return ""
    query = " OR ".join(parts)
    if not include_retweets:
        query += " -is:retweet"
    return query


def compute_date_range(start_date, end_date):
    now_utc = datetime.now(timezone.utc)
    earliest_allowed = now_utc - timedelta(days=7) + timedelta(hours=1)
    start_dt = max(
        datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc),
        earliest_allowed,
    )
    end_dt = min(
        datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc),
        now_utc - timedelta(seconds=30),
    )
    return start_dt, end_dt


def fetch_tweets(
    bearer_token: str,
    keywords: list[str],
    handles: list[str],
    start_date,
    end_date,
    max_results: int = 100,
    include_retweets: bool = False,
):
    """
    Generator that yields tweets one at a time as they are fetched from Twitter API v2.
    """
    query = build_query(keywords, handles, include_retweets)
    if not query:
        return

    start_dt, end_dt = compute_date_range(start_date, end_date)

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

    tweet_fields = ["created_at", "author_id", "public_metrics", "lang", "geo"]
    user_fields = ["username", "name"]
    expansions = ["author_id"]

    seen_ids = set()
    collected = 0
    users_map = {}

    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        start_time=start_dt,
        end_time=end_dt,
        tweet_fields=tweet_fields,
        user_fields=user_fields,
        expansions=expansions,
        max_results=min(max_results, 100),
    )

    try:
        for response in paginator:
            if response.includes and "users" in response.includes:
                for user in response.includes["users"]:
                    users_map[user.id] = {"username": user.username, "name": user.name}

            if not response.data:
                continue

            for tweet in response.data:
                if tweet.id in seen_ids:
                    continue
                seen_ids.add(tweet.id)

                user_info = users_map.get(tweet.author_id, {})
                metrics = tweet.public_metrics or {}

                yield {
                    "id": str(tweet.id),
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "author_id": str(tweet.author_id),
                    "username": user_info.get("username", ""),
                    "name": user_info.get("name", ""),
                    "text": tweet.text,
                    "lang": tweet.lang,
                    "retweet_count": metrics.get("retweet_count", 0),
                    "like_count": metrics.get("like_count", 0),
                    "reply_count": metrics.get("reply_count", 0),
                    "quote_count": metrics.get("quote_count", 0),
                    "sentiment": None,
                    "category": None,
                    "score": None,
                    "reasoning": None,
                }

                collected += 1
                if collected >= max_results:
                    return

            if collected >= max_results:
                return

    except tweepy.errors.TooManyRequests as e:
        yield {"_error": "429", "_message": str(e), "_reset_time": datetime.now(timezone.utc) + timedelta(minutes=15)}
    except tweepy.errors.TwitterServerError as e:
        yield {"_error": "5xx", "_message": str(e)}
    except tweepy.errors.Unauthorized as e:
        yield {"_error": "401", "_message": str(e)}
    except tweepy.errors.Forbidden as e:
        yield {"_error": "403", "_message": str(e)}
    except Exception as e:
        yield {"_error": "unknown", "_message": str(e)}