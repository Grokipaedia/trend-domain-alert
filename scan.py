#!/usr/bin/env python3
"""
Trend-to-Domain Alert — scan.py

Watches a curated list of RSS feeds for newly-coined terms (acronyms,
program names, policy names) and checks whether the matching domain
is available, alerting immediately via Telegram if so.

Run on a schedule via GitHub Actions (see .github/workflows/scan.yml).
Requires three secrets: ANTHROPIC_API_KEY, DYNADOT_API_KEY, TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID.
"""

import os
import re
import json
import time
import hashlib
import feedparser
import requests
from anthropic import Anthropic

FEEDS_FILE = "feeds.json"
SEEN_FILE = "seen.json"
TERM_HISTORY_FILE = "term_history.json"
STATE_MAX_AGE_DAYS = 14  # forget seen items older than this, keep seen.json small
TERM_HISTORY_MAX_AGE_DAYS = 90  # keep term momentum history longer than raw seen items

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DYNADOT_API_KEY = os.environ.get("DYNADOT_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TLDS_TO_CHECK = [".com", ".io", ".ai"]


def normalize_term(term):
    """Canonical form of a term for history tracking, so 'ICNEV' and
    'icnev' and ' ICNEV ' are all recognized as the same term across runs."""
    return re.sub(r"\s+", " ", term.strip()).lower()


def update_term_history(history, term, now):
    """
    Record a mention of this term and return its momentum info.
    Returns: {"mention_count": int, "first_seen": float, "is_new": bool}
    """
    key = normalize_term(term)
    if key not in history:
        history[key] = {"first_seen": now, "last_seen": now, "mention_count": 1}
        is_new = True
    else:
        history[key]["last_seen"] = now
        history[key]["mention_count"] += 1
        is_new = False
    return {
        "mention_count": history[key]["mention_count"],
        "first_seen": history[key]["first_seen"],
        "is_new": is_new,
    }


def compute_score(mention_count, available_count, total_checked):
    """
    Transparent, documented scoring -- every point traces to a real,
    countable input. No LLM-guessed confidence numbers.

    - Momentum: up to 50 points, +10 per mention seen (caps at 5 mentions).
      A term seen once is unproven; one seen across multiple scan runs
      has demonstrated it isn't a one-off mention.
    - Availability breadth: up to 50 points, scaled by what fraction of
      checked TLDs are actually available. All available = full 50;
      none available = 0, since someone likely already claimed the space.
    """
    momentum_points = min(mention_count, 5) * 10
    availability_points = (available_count / total_checked * 50) if total_checked else 0
    total = round(momentum_points + availability_points)
    return {
        "total": min(total, 100),
        "momentum_points": momentum_points,
        "availability_points": round(availability_points, 1),
    }


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def item_id(entry):
    """Stable hash for a feed entry, used for de-duplication."""
    key = entry.get("link") or entry.get("id") or entry.get("title", "")
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def fetch_new_items(feeds, seen):
    new_items = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  [WARN] Failed to fetch {feed_url}: {e}")
            continue
        for entry in parsed.entries:
            iid = item_id(entry)
            if iid in seen:
                continue
            new_items.append({
                "id": iid,
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", "")),
                "link": entry.get("link", ""),
                "source": feed_url,
                "seen_at": time.time(),
            })
    return new_items


def extract_new_term(client, item):
    """
    Ask Claude whether this item describes a genuinely newly-coined term
    (acronym, program name, policy name). Returns a dict with 'found' bool
    and, if found, 'term' and 'reason'. Fails closed (found=False) on any
    parsing problem, rather than guessing.
    """
    text = f"Title: {item['title']}\n\nSummary: {item['summary']}"[:3000]

    prompt = f"""You are screening a news item to see if it introduces a
genuinely NEW term: a newly-coined acronym, program name, policy name, or
initiative name that did not exist before this announcement. This is NOT
about existing well-known terms (like "EV" or "AI") being mentioned again.

Only flag it if a specific, novel, nameable term is being introduced for
the first time in this text -- something a domain investor might want to
register the matching domain for, because it could become the standard
shorthand for this new thing.

News item:
{text}

Respond with ONLY a JSON object, no other text, in this exact shape:
{{"found": true, "term": "THE FULL TERM", "acronym": "ITS ACRONYM IF ONE IS USED IN THE TEXT, ELSE NULL", "reason": "one sentence on why it might matter"}}
or
{{"found": false}}
"""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if the model added them
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        if parsed.get("found") and parsed.get("term"):
            return parsed
    except Exception as e:
        print(f"  [WARN] Extraction failed for '{item['title'][:60]}': {e}")
    return {"found": False}


def check_blocklist(client, term, reason):
    """
    Screens a candidate term for categories that should never be suggested
    as a registration opportunity, regardless of how strong the momentum or
    availability score looks. This is the inverse safety design from
    extract_new_term: there, uncertainty defaults to "not a new term" (fail
    closed, avoid false alarms). Here, uncertainty defaults to "block it"
    (fail SAFE, avoid suggesting something legally or ethically risky) --
    a missed opportunity costs nothing; a suggested trademark or tragedy
    domain costs real reputation and possibly real legal exposure.

    Returns: {"blocked": bool, "category": str or None, "explanation": str}
    """
    prompt = f"""You are screening a candidate domain-name term for red flags
before it's suggested as a registration opportunity. Block the term if it
falls into ANY of these categories:

1. TRADEMARK: matches or is confusingly similar to an existing company name,
   product name, or registered trademark.
2. PRIVATE_INDIVIDUAL: refers to or is derived from a private (non-public-
   figure) person's real name, drawn from a news story about them.
3. DISASTER: relates to a disaster, tragedy, death, or crisis event --
   suggesting a domain from this would read as exploiting the event.
4. FLASH_NEWS: describes a single viral moment or one-off event name, not
   an actual reusable term/acronym/product name likely to see repeated
   future use.

Term: "{term}"
Context: {reason}

Respond with ONLY a JSON object:
{{"blocked": true, "category": "TRADEMARK", "explanation": "one sentence"}}
or
{{"blocked": false}}
"""
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = json.loads(raw)
        if "blocked" in parsed:
            return parsed
    except Exception as e:
        print(f"  [WARN] Blocklist check itself failed for '{term}': {e} -- blocking as a precaution")
    # Any failure -- malformed response, API error, anything -- blocks by default
    return {"blocked": True, "category": "CHECK_FAILED", "explanation": "the safety check itself could not be completed"}


def term_to_domain_candidates(term):
    """
    Turn an extracted term into candidate bare-domain strings.
    For multi-word phrases, also generates the acronym (first letters of
    each word) since that's frequently the more valuable, more likely-to-
    become-standard-shorthand domain -- e.g. "Intelligent Connected New
    Energy Vehicles" should also yield "icnev", not just the 38-character
    full phrase.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "", term)
    if not slug:
        return []

    candidates = [slug.lower()]

    words = re.findall(r"[A-Za-z0-9]+", term)
    if len(words) >= 3:
        acronym = "".join(w[0] for w in words if w[0].isalpha()).lower()
        if acronym and acronym not in candidates:
            candidates.append(acronym)

    return candidates


def check_domain_available(domain_base, tld):
    """
    Check availability via Dynadot's API.
    Docs: https://www.dynadot.com/domain/api3.html (search command)
    Returns True/False, or None if the check itself failed.
    """
    if not DYNADOT_API_KEY:
        print("  [WARN] No DYNADOT_API_KEY set -- skipping real availability check")
        return None

    full_domain = domain_base + tld
    url = "https://api.dynadot.com/api3.json"
    params = {
        "key": DYNADOT_API_KEY,
        "command": "search",
        "domain0": full_domain,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Dynadot's response shape: SearchResponse -> SearchResults -> [{Available: "yes"/"no", ...}]
        results = data.get("SearchResponse", {}).get("SearchResults", [])
        if results and results[0].get("Available", "").lower() == "yes":
            return True
        return False
    except Exception as e:
        print(f"  [WARN] Dynadot check failed for {full_domain}: {e}")
        return None


def send_telegram_alert(term, reason, available_domains, source_item, momentum, score):
    momentum_badge = "\U0001F195 First sighting" if momentum["is_new"] else f"\U0001F525 Rising \u2014 seen {momentum['mention_count']}x"

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  [WARN] Telegram not configured -- printing alert instead:")
        print(f"    NEW TERM: {term}\n    WHY: {reason}\n    MOMENTUM: {momentum_badge}\n"
              f"    SCORE: {score['total']}/100 (momentum {score['momentum_points']} + availability {score['availability_points']})\n"
              f"    AVAILABLE: {available_domains}\n    SOURCE: {source_item['link']}")
        return

    domains_str = "\n".join(f"  \u2022 {d} \u2014 AVAILABLE" for d in available_domains)
    message = (
        f"\U0001F6A8 New term spotted: *{term}*\n\n"
        f"{reason}\n\n"
        f"{momentum_badge}\n"
        f"*Score: {score['total']}/100* "
        f"(momentum {score['momentum_points']}pt + availability {score['availability_points']}pt)\n\n"
        f"*Domains available:*\n{domains_str}\n\n"
        f"Source: {source_item['title']}\n{source_item['link']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": False,
        }, timeout=10)
    except Exception as e:
        print(f"  [WARN] Telegram send failed: {e}")


def main():
    feeds = load_json(FEEDS_FILE, [])
    if not feeds:
        print("No feeds configured in feeds.json -- nothing to do.")
        return

    seen = load_json(SEEN_FILE, {})
    # prune old seen entries so the file doesn't grow forever
    cutoff = time.time() - STATE_MAX_AGE_DAYS * 86400
    seen = {k: v for k, v in seen.items() if v > cutoff}

    print(f"Scanning {len(feeds)} feed(s)...")
    new_items = fetch_new_items(feeds, seen)
    print(f"Found {len(new_items)} new item(s) not seen before.")

    if not new_items:
        save_json(SEEN_FILE, seen)
        return

    client = None
    if ANTHROPIC_API_KEY:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        print("[WARN] No ANTHROPIC_API_KEY set -- cannot run extraction, exiting.")
        return

    term_history = load_json(TERM_HISTORY_FILE, {})
    history_cutoff = time.time() - TERM_HISTORY_MAX_AGE_DAYS * 86400
    term_history = {k: v for k, v in term_history.items() if v.get("last_seen", 0) > history_cutoff}

    alerts_sent = 0
    for item in new_items:
        seen[item["id"]] = item["seen_at"]

        result = extract_new_term(client, item)
        if not result.get("found"):
            continue

        term = result["term"]
        reason = result.get("reason", "")
        print(f"  Candidate term: '{term}' -- {reason}")

        block_result = check_blocklist(client, term, reason)
        if block_result.get("blocked"):
            category = block_result.get("category", "UNKNOWN")
            explanation = block_result.get("explanation", "")
            print(f"    BLOCKED [{category}]: {explanation}")
            continue

        momentum = update_term_history(term_history, term, time.time())

        candidates = term_to_domain_candidates(term)
        acronym = result.get("acronym")
        if acronym:
            acronym_slug = re.sub(r"[^a-zA-Z0-9]+", "", acronym).lower()
            if acronym_slug and acronym_slug not in candidates:
                candidates.append(acronym_slug)

        available = []
        total_checked = 0
        for base in candidates:
            for tld in TLDS_TO_CHECK:
                avail = check_domain_available(base, tld)
                if avail is not None:  # only count checks that actually succeeded
                    total_checked += 1
                    if avail:
                        available.append(base + tld)

        score = compute_score(momentum["mention_count"], len(available), total_checked)
        print(f"    Momentum: {momentum['mention_count']}x seen | Score: {score['total']}/100"
              f" (momentum {score['momentum_points']} + availability {score['availability_points']})")

        if available:
            send_telegram_alert(term, reason, available, item, momentum, score)
            alerts_sent += 1
        else:
            print(f"    No available domains found for '{term}' (or check unavailable).")

    save_json(SEEN_FILE, seen)
    save_json(TERM_HISTORY_FILE, term_history)
    print(f"Done. {alerts_sent} alert(s) sent.")


if __name__ == "__main__":
    main()
