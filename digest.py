"""
AI Releases Daily Digest Agent
Runs via GitHub Actions — no always-on server needed.

Requires two GitHub secrets:
  GROQ_API_KEY        — free from console.groq.com (no credit card, unlimited free tier)
  SLACK_WEBHOOK_URL   — incoming webhook URL from your Slack app
"""

import os, json, re, sys, time
from datetime import datetime, timezone
import requests
import groq

# ── Config ───────────────────────────────────────────────────────────

CATEGORIES = {
    "frontier":  "🧠 Frontier models",
    "agents":    "🤖 Agents & autonomy",
    "infra":     "⚙️  AI infrastructure",
    "applied":   "💼 Applied AI",
    "oss":       "🔓 Open source",
    "research":  "📄 Research papers",
}

DIGEST_PROMPT = f"""
You are an expert AI releases digest agent. Today is {datetime.now(timezone.utc).strftime('%A, %d %B %Y')}.

Search your knowledge for the most notable AI announcements, releases, and papers published in the last 24 hours.
Cover all six categories below. Aim for 3–4 items per category (15–20 total).

Categories:
- frontier  : frontier model releases/updates/evals (OpenAI, Anthropic, Google, Meta, Mistral, xAI, etc.)
- agents    : coding agents, browser agents, multi-agent systems, agent frameworks
- infra     : GPUs, inference optimisation, fine-tuning, MLOps, chips, serving
- applied   : enterprise AI products, vertical AI applications, notable product launches
- oss       : open-source model releases, Hugging Face drops, new weights, datasets
- research  : notable arXiv papers, safety research, new benchmarks

Return ONLY a valid JSON array — no markdown, no backticks, no explanation before or after.
Each item must match this exact schema:
{{
  "title":    "<concise headline, max 12 words>",
  "cat":      "<one of: frontier | agents | infra | applied | oss | research>",
  "source":   "<e.g. arxiv | huggingface | openai blog | hacker news | github | x.com | techcrunch>",
  "summary":  "<2 sentences: what it is + why it matters>",
  "url":      "<direct link to announcement or paper>",
  "signal":   <1 | 2 | 3>
}}
signal = 1, 2, or 3. 3 means must-read / highly significant.
"""

# ── Groq call (free, unlimited, no quota) ────────────────────────────────

# Make the model configurable so deprecations don't break the workflow.
# Default to a safe Mixtral name; override with the GROQ_MODEL env var or a repo secret.
MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")


def validate_item(item: any) -> dict | None:
    """
    Validate and sanitize a single digest item.
    Returns the item if valid, None if invalid.
    """
    # Must be a dictionary
    if not isinstance(item, dict):
        print(f"⚠️  Skipping invalid item (not a dict): {repr(item)[:100]}", file=sys.stderr)
        return None
    
    # Validate required fields
    required_fields = ["title", "cat", "source", "summary", "url", "signal"]
    for field in required_fields:
        if field not in item:
            print(f"⚠️  Skipping item missing field '{field}': {item.get('title', 'unknown')}", file=sys.stderr)
            return None
    
    # Ensure all fields are strings except signal
    for field in ["title", "cat", "source", "summary", "url"]:
        if not isinstance(item.get(field), str):
            print(f"⚠️  Skipping item with non-string '{field}': {item.get('title', 'unknown')}", file=sys.stderr)
            return None
    
    # Validate and sanitize signal (must be int 1-3)
    signal = item.get("signal")
    if not isinstance(signal, int):
        try:
            signal = int(signal)
        except (ValueError, TypeError):
            print(f"⚠️  Invalid signal for '{item.get('title', 'unknown')}': {signal}. Using default 1.", file=sys.stderr)
            signal = 1
    
    signal = max(1, min(3, signal))  # Clamp to 1-3
    item["signal"] = signal
    
    # Validate category
    if item["cat"] not in CATEGORIES:
        print(f"⚠️  Invalid category '{item['cat']}' for '{item.get('title', 'unknown')}'. Using 'applied'.", file=sys.stderr)
        item["cat"] = "applied"
    
    return item


def fetch_digest(max_retries: int = 3) -> list[dict]:
    """
    Fetch AI digest from Groq API (free, unlimited tier).
    Model is configurable via the GROQ_MODEL environment variable.
    Returns validated list of items.
    """
    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])

    for attempt in range(max_retries):
        try:
            print(f"📡 Calling Groq model {MODEL} (attempt {attempt + 1}/{max_retries})...")
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI digest agent. Return ONLY valid JSON array, nothing else. No markdown, no code fences."
                    },
                    {
                        "role": "user",
                        "content": DIGEST_PROMPT
                    }
                ],
                temperature=0.7,
                max_tokens=4096,
            )

            raw = response.choices[0].message.content

            # Extract JSON array from response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array found in response:\n{raw[:500]}")

            # Parse JSON
            parsed = json.loads(match.group())
            
            # Ensure it's a list
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
            
            # Validate and filter items
            validated_items = []
            for item in parsed:
                validated = validate_item(item)
                if validated:
                    validated_items.append(validated)
            
            if not validated_items:
                raise ValueError("No valid items found after validation")
            
            print(f"✅  Fetched {len(validated_items)} valid items from Groq")
            return validated_items

        except groq.BadRequestError as e:
            # Specific handling for decommissioned / invalid model errors to give an actionable message.
            msg = str(e).lower()
            if "decommission" in msg or "no longer supported" in msg:
                err_text = (
                    f"Groq model '{MODEL}' appears to be decommissioned or unsupported.\n"
                    "Set the GROQ_MODEL environment variable (or add a repository secret named GROQ_MODEL) to a supported model.\n"
                    "See: https://console.groq.com/docs/deprecations for recommended replacements."
                )
                print(f"❌  {err_text}", file=sys.stderr)
                raise RuntimeError(err_text) from e
            # Re-raise for other BadRequest situations
            print(f"❌  BadRequestError from Groq: {e}", file=sys.stderr)
            raise

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️  Rate limited. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌  Rate limited after {max_retries} attempts.", file=sys.stderr)
                    raise
            else:
                # Non-rate-limit errors should fail immediately
                print(f"❌  Error: {e}", file=sys.stderr)
                raise


# ── Slack formatter ────────────────────────────────────────────────────────

def signal_dots(n: int) -> str:
    """Convert signal level (1-3) to visual dots."""
    n = max(1, min(3, int(n)))  # Clamp to 1-3
    return "●" * n + "○" * (3 - n)


def build_slack_payload(items: list[dict]) -> dict:
    """Build Slack message payload from digest items."""
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    top   = [i for i in items if i.get("signal") == 3]

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🗞  AI Releases Digest — {today}"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*{len(items)} items* across "
                        + ", ".join(CATEGORIES.values())
                        + f"  ·  {len(top)} must-reads ●●●"
                    ),
                }
            ],
        },
        {"type": "divider"},
    ]

    # Group items by category
    by_cat: dict[str, list] = {k: [] for k in CATEGORIES}
    for item in items:
        cat = item.get("cat", "applied")
        if cat in by_cat:
            by_cat[cat].append(item)

    # Render each category
    for cat_key, cat_label in CATEGORIES.items():
        cat_items = by_cat.get(cat_key, [])
        if not cat_items:
            continue

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{cat_label}*"},
        })

        for item in cat_items:
            # Extract and sanitize fields
            title   = str(item.get("title", "Untitled"))
            summary = str(item.get("summary", ""))
            url     = str(item.get("url", ""))
            source  = str(item.get("source", ""))
            signal  = item.get("signal", 1)
            
            # Ensure signal is valid
            if not isinstance(signal, int):
                try:
                    signal = int(signal)
                except (ValueError, TypeError):
                    signal = 1
            signal = max(1, min(3, signal))
            
            dots = signal_dots(signal)

            # Build link text
            link_text = f"<{url}|{title}>" if url else title
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"{dots}  {link_text}\n"
                        f"_{summary}_\n"
                        f"› _{source}_"
                    ),
                },
            })

        blocks.append({"type": "divider"})

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Powered by Groq {MODEL} (free tier) · runs daily at 18:00 Budapest time",
            }
        ],
    })

    return {"blocks": blocks}


# ── Slack delivery ─────────────────────────────────────────────────────────

def send_to_slack(payload: dict) -> None:
    """Send formatted digest to Slack via webhook."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("⚠️  SLACK_WEBHOOK_URL not set — skipping Slack delivery")
        return

    resp = requests.post(webhook, json=payload, timeout=10)
    if resp.status_code == 200:
        print("✅  Slack message sent")
    else:
        print(f"❌  Slack error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀  Starting AI Releases Digest Agent")
    try:
        items   = fetch_digest()
        payload = build_slack_payload(items)
        send_to_slack(payload)
        print("🎉  Done")
    except Exception as e:
        print(f"❌  Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
