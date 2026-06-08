"""
AI Releases Daily Digest Agent
Runs via GitHub Actions — no always-on server needed.

Requires two GitHub secrets:
  GEMINI_API_KEY      — free from aistudio.google.com (no credit card)
  SLACK_WEBHOOK_URL   — incoming webhook URL from your Slack app
"""

import os, json, re, sys, time
import requests
from datetime import datetime, timezone
from google import genai
from google.genai import types
from google.genai.errors import ClientError

# ── Config ───────────────────────────────────────────────────────────[...]

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

Search the web for the most notable AI announcements, releases, and papers published in the last 24 hours.
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
signal = 3 means must-read / highly significant.
"""

# ── Gemini call (with Google Search grounding + retry logic) ────────────────────────────────

def fetch_digest(max_retries: int = 3) -> list[dict]:
    """
    Fetch AI digest from Gemini 2.5 Pro with exponential backoff retry logic.
    Handles rate limiting (429) gracefully.
    """
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    for attempt in range(max_retries):
        try:
            print(f"📡 Calling Gemini 2.5 Pro (attempt {attempt + 1}/{max_retries})...")
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=DIGEST_PROMPT,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )

            raw = response.text

            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON array found in response:\n{raw[:500]}")

            items = json.loads(match.group())
            print(f"✅  Fetched {len(items)} items from Gemini 2.5 Pro")
            return items
            
        except ClientError as e:
            if e.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"⚠️  Rate limited (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌  Rate limited after {max_retries} attempts. Check your Gemini API quota:", file=sys.stderr)
                    print(f"    Visit: https://ai.google.dev/gemini-api/docs/rate-limits", file=sys.stderr)
                    print(f"    Or upgrade at: https://aistudio.google.com/app/apikey", file=sys.stderr)
                    raise
            else:
                # Non-rate-limit errors should fail immediately
                raise


# ── Slack formatter ─────────────────────────────────────────────────────────[...]

def signal_dots(n: int) -> str:
    return "●" * n + "○" * (3 - n)

def build_slack_payload(items: list[dict]) -> dict:
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

    by_cat: dict[str, list] = {k: [] for k in CATEGORIES}
    for item in items:
        cat = item.get("cat", "applied")
        if cat in by_cat:
            by_cat[cat].append(item)

    for cat_key, cat_label in CATEGORIES.items():
        cat_items = by_cat.get(cat_key, [])
        if not cat_items:
            continue

        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*{cat_label}*"},
        })

        for item in cat_items:
            title   = item.get("title", "Untitled")
            summary = item.get("summary", "")
            url     = item.get("url", "")
            source  = item.get("source", "")
            signal  = item.get("signal", 1)
            dots    = signal_dots(signal)

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
                "text": "Powered by Gemini 2.5 Pro + Google Search · runs daily at 18:00 Budapest time",
            }
        ],
    })

    return {"blocks": blocks}


# ── Slack delivery ─────────────────────────────────────────────────────────[...]

def send_to_slack(payload: dict) -> None:
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


# ── Main ────────────────────────────────────────────────────────────[...]

if __name__ == "__main__":
    print("🚀  Starting AI Releases Digest Agent")
    items   = fetch_digest()
    payload = build_slack_payload(items)
    send_to_slack(payload)
    print("🎉  Done")
