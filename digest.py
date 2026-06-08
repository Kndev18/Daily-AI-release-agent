import os
import json
import requests
import anthropic
from datetime import datetime

DIGEST_PROMPT = """
You are an AI news curator. Today is {date}.

Search the web and find today's most important AI releases, announcements, and updates across these 6 bundles:

1. **Foundation Models** – New LLMs, multimodal models, reasoning models
2. **Developer Tools** – APIs, SDKs, frameworks, libraries for AI development
3. **AI Agents & Automation** – Agent frameworks, autonomous systems, workflow tools
4. **Image / Video / Audio** – Generative media models and tools
5. **AI Products & Apps** – Consumer and enterprise AI product launches
6. **Research & Papers** – Notable papers, benchmarks, evals

For each bundle, list the top 2-3 releases with:
- Name and one-line description
- Why it matters
- Link if available

Format the output as a clean Slack message using *bold* for titles and bullet points.
Start with a header: *AI Releases Digest – {date}*
End with a footer: _Powered by Claude + GitHub Actions_
"""

def run_digest():
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    slack_url = os.environ["SLACK_WEBHOOK_URL"]
    
    today = datetime.utcnow().strftime("%A, %B %d, %Y")
    prompt = DIGEST_PROMPT.format(date=today)
    
    print(f"Running digest for {today}...")
    
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    digest_text = message.content[0].text
    print("Digest generated. Posting to Slack...")
    
    payload = {
        "text": digest_text
    }
    
    response = requests.post(
        slack_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        print("Successfully posted to Slack!")
    else:
        print(f"Failed to post to Slack: {response.status_code} {response.text}")
        raise Exception("Slack post failed")

if __name__ == "__main__":
    run_digest()
