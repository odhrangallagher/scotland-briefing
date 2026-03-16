"""
Scotland Morning Briefing — daily Scottish politics email briefing.
Run once and exits. Reads RSS feeds, generates AI briefing via Anthropic API,
sends HTML email via Gmail SMTP.
"""

import os
import re
import json
import smtplib
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from difflib import SequenceMatcher

import feedparser
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RSS_FEEDS = [
    "https://news.google.com/rss/search?q=Scotland+politics&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=Scottish+Parliament&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=Scottish+Government&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=SNP&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=Scottish+independence&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=Holyrood&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=John+Swinney&hl=en-GB&gl=GB&ceid=GB:en",
    "https://www.heraldscotland.com/news/politics/rss/",
    "https://www.scotsman.com/topic/scottish-politics/rss",
    "https://feeds.bbci.co.uk/news/scotland/rss.xml",
    "https://news.google.com/rss/search?q=Scottish+Labour&hl=en-GB&gl=GB&ceid=GB:en",
    "https://news.google.com/rss/search?q=Scottish+Conservative&hl=en-GB&gl=GB&ceid=GB:en",
]

CUTOFF_HOURS = 18
SIMILARITY_THRESHOLD = 0.72   # headlines closer than this are considered duplicates
ANTHROPIC_MODEL = "claude-opus-4-6"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM_PROMPT = """You are a senior political editor briefing a journalist at The Scottish Brief on
the key Scottish political stories of the day. Based on the article list provided,
write a morning briefing with the following structure:

1. THE LEAD STORY: 2-3 sentences on the single most important development.
2. KEY STORIES: 3-5 bullet points covering the other significant stories.
   Each bullet: one sentence summary + why it matters.
3. HOLYROOD WATCH: Any upcoming votes, committee sessions, or Holyrood
   business worth knowing about today.
4. ONE TO WATCH: A slow-burn story that might develop.

Tone: sharp, direct, no waffle. Written for a tabloid journalist who needs
to know what matters and why fast. Focus on Scotland — filter out anything
that is purely Westminster or England with no Scottish angle.

Also read the file scotland_politics_rag.md if provided — it contains domain
knowledge about current stories and key figures to prioritise."""


# ---------------------------------------------------------------------------
# 1. RSS feed ingestion
# ---------------------------------------------------------------------------

def fetch_feed(url: str) -> list[dict]:
    """Fetch a single RSS feed and return a list of article dicts."""
    try:
        feed = feedparser.parse(url)
        articles = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=CUTOFF_HOURS)

        for entry in feed.entries:
            # Parse published date — fall back to now if missing
            published = None
            for attr in ("published_parsed", "updated_parsed"):
                if hasattr(entry, attr) and getattr(entry, attr):
                    import calendar
                    ts = calendar.timegm(getattr(entry, attr))
                    published = datetime.fromtimestamp(ts, tz=timezone.utc)
                    break
            if published is None:
                published = datetime.now(timezone.utc)

            if published < cutoff:
                continue

            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            source = feed.feed.get("title", url)

            # Google News wraps the real source in the title: "Headline - Source"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                if len(parts) > 1:
                    source = parts[1].strip()

            if title and link:
                articles.append({
                    "title": title,
                    "url": link,
                    "source": source,
                    "published": published.isoformat(),
                })
        return articles
    except Exception as exc:
        log.warning("Failed to fetch %s: %s", url, exc)
        return []


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower())


def _similar(a: str, b: str) -> bool:
    ratio = SequenceMatcher(None, _normalise(a), _normalise(b)).ratio()
    return ratio >= SIMILARITY_THRESHOLD


def deduplicate(articles: list[dict]) -> list[dict]:
    """Remove articles with near-identical headlines, keeping earliest."""
    seen: list[str] = []
    unique: list[dict] = []
    for article in articles:
        title = article["title"]
        if any(_similar(title, s) for s in seen):
            continue
        seen.append(title)
        unique.append(article)
    return unique


def collect_articles() -> list[dict]:
    all_articles: list[dict] = []
    for url in RSS_FEEDS:
        log.info("Fetching %s", url)
        articles = fetch_feed(url)
        log.info("  → %d recent articles", len(articles))
        all_articles.extend(articles)

    # Primary dedup on exact URL
    seen_urls: set[str] = set()
    url_deduped: list[dict] = []
    for a in all_articles:
        if a["url"] not in seen_urls:
            seen_urls.add(a["url"])
            url_deduped.append(a)

    unique = deduplicate(url_deduped)
    log.info("Total unique articles after dedup: %d", len(unique))
    return unique


# ---------------------------------------------------------------------------
# 2. AI briefing generation
# ---------------------------------------------------------------------------

def load_rag() -> str:
    rag_path = Path(__file__).parent / "scotland_politics_rag.md"
    if rag_path.exists():
        log.info("Loading RAG context from %s", rag_path)
        return rag_path.read_text(encoding="utf-8")
    return ""


def build_user_message(articles: list[dict], rag: str) -> str:
    today = datetime.now().strftime("%A %d %B %Y")
    lines = [f"Today is {today}.", "", "ARTICLES:", ""]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. {a['title']}")
        lines.append(f"   Source: {a['source']}")
        lines.append(f"   URL: {a['url']}")
        lines.append(f"   Published: {a['published']}")
        lines.append("")

    if rag:
        lines += ["", "---", "DOMAIN KNOWLEDGE (scotland_politics_rag.md):", "", rag]

    return "\n".join(lines)


def generate_briefing(articles: list[dict]) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    rag = load_rag()
    user_message = build_user_message(articles, rag)

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message}],
    }

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    log.info("Sending %d articles to Anthropic API (model: %s)", len(articles), ANTHROPIC_MODEL)
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    briefing_text = data["content"][0]["text"]
    log.info("Briefing generated (%d chars)", len(briefing_text))
    return briefing_text


# ---------------------------------------------------------------------------
# 3. Email delivery
# ---------------------------------------------------------------------------

def briefing_to_html(briefing_text: str, articles: list[dict], date_str: str) -> str:
    """Convert plain-text briefing + article list to a styled HTML email."""

    def escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Convert briefing markdown-ish text to HTML paragraphs / lists
    def format_body(text: str) -> str:
        html_lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                html_lines.append("<br>")
            elif stripped.startswith("# ") or stripped.startswith("1. ") or stripped.startswith("2. ") or stripped.startswith("3. ") or stripped.startswith("4. "):
                # Section headers
                html_lines.append(f'<h2 style="color:#1a1a2e;border-bottom:2px solid #c8102e;padding-bottom:4px;margin-top:24px">{escape(stripped)}</h2>')
            elif stripped.startswith("- ") or stripped.startswith("• "):
                content = stripped[2:].strip()
                # Bold the text before the first colon if present
                if ": " in content:
                    head, rest = content.split(": ", 1)
                    html_lines.append(f'<li style="margin-bottom:8px"><strong>{escape(head)}</strong>: {escape(rest)}</li>')
                else:
                    html_lines.append(f'<li style="margin-bottom:8px">{escape(content)}</li>')
            else:
                html_lines.append(f'<p style="margin:0 0 10px">{escape(stripped)}</p>')

        # Wrap consecutive <li> in <ul>
        result = []
        in_ul = False
        for tag in html_lines:
            if tag.startswith("<li"):
                if not in_ul:
                    result.append('<ul style="padding-left:20px;margin:8px 0">')
                    in_ul = True
            else:
                if in_ul:
                    result.append("</ul>")
                    in_ul = False
            result.append(tag)
        if in_ul:
            result.append("</ul>")
        return "\n".join(result)

    body_html = format_body(briefing_text)

    # Source articles list
    sources_html_parts = []
    for a in articles:
        sources_html_parts.append(
            f'<li style="margin-bottom:6px">'
            f'<a href="{a["url"]}" style="color:#c8102e;text-decoration:none">{escape(a["title"])}</a>'
            f' <span style="color:#888;font-size:12px">— {escape(a["source"])}</span>'
            f'</li>'
        )
    sources_html = "\n".join(sources_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scotland Morning Briefing — {date_str}</title>
</head>
<body style="margin:0;padding:0;background:#f4f4f4;font-family:Georgia,serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f4;padding:20px 0">
    <tr><td align="center">
      <table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:6px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.12)">

        <!-- Header -->
        <tr>
          <td style="background:#1a1a2e;padding:28px 36px">
            <p style="margin:0;color:#c8102e;font-family:Arial,sans-serif;font-size:11px;letter-spacing:2px;text-transform:uppercase">The Scottish Brief</p>
            <h1 style="margin:6px 0 0;color:#ffffff;font-family:Arial,sans-serif;font-size:26px;font-weight:700">Scotland Morning Briefing</h1>
            <p style="margin:6px 0 0;color:#aaaacc;font-family:Arial,sans-serif;font-size:13px">{date_str}</p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:32px 36px;color:#1a1a1a;font-size:15px;line-height:1.7">
            {body_html}
          </td>
        </tr>

        <!-- Divider -->
        <tr>
          <td style="padding:0 36px">
            <hr style="border:none;border-top:1px solid #e0e0e0">
          </td>
        </tr>

        <!-- Source articles -->
        <tr>
          <td style="padding:24px 36px">
            <h3 style="font-family:Arial,sans-serif;font-size:13px;letter-spacing:1px;text-transform:uppercase;color:#888;margin:0 0 14px">Source Articles</h3>
            <ul style="padding-left:18px;margin:0;font-size:13px;line-height:1.6">
              {sources_html}
            </ul>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#1a1a2e;padding:16px 36px;text-align:center">
            <p style="margin:0;color:#888;font-family:Arial,sans-serif;font-size:11px">
              Generated by Scotland Morning Briefing · {date_str}
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_email(briefing_text: str, articles: list[dict]) -> None:
    gmail_address = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    recipient = os.environ["RECIPIENT_EMAIL"]

    date_str = datetime.now().strftime("%A %d %B %Y")
    subject = f"Scotland Morning Briefing \u2014 {datetime.now().strftime('%d %B %Y')}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = recipient

    # Plain text fallback
    plain = f"Scotland Morning Briefing — {date_str}\n\n{briefing_text}\n\nSource articles:\n"
    for a in articles:
        plain += f"  • {a['title']} ({a['source']})\n    {a['url']}\n"
    msg.attach(MIMEText(plain, "plain", "utf-8"))

    # HTML version
    html = briefing_to_html(briefing_text, articles, date_str)
    msg.attach(MIMEText(html, "html", "utf-8"))

    log.info("Connecting to Gmail SMTP...")
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(gmail_address, app_password)
        server.sendmail(gmail_address, recipient, msg.as_string())
    log.info("Email sent to %s", recipient)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Scotland Morning Briefing starting ===")

    # Validate env vars early
    required = ["ANTHROPIC_API_KEY", "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "RECIPIENT_EMAIL"]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    articles = collect_articles()
    if not articles:
        raise SystemExit("No articles found in the last 18 hours — nothing to send.")

    briefing_text = generate_briefing(articles)
    send_email(briefing_text, articles)

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
