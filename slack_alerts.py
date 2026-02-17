import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

PRIORITY_EMOJI = {
    "New Follower": "🟢",
    "Liked Post": "⭐",
    "Retweeted": "🔁",
    "Replied": "💬",
    "Lost Follower": "🔴",
}


def send_slack_message(blocks: list, text: str = "GI Researcher Alert") -> bool:
    """Send a Slack message via webhook. Returns True on success."""
    if not SLACK_WEBHOOK_URL:
        print("No SLACK_WEBHOOK_URL set — skipping Slack alert.")
        return False
    try:
        resp = requests.post(
            SLACK_WEBHOOK_URL,
            json={"text": text, "blocks": blocks},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"Slack error: {e}")
        return False


def alert_new_follower(researcher: dict) -> bool:
    """Alert when a target researcher starts following GI."""
    name = researcher["name"]
    handle = researcher["twitter"]
    focus = researcher.get("focus", "")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🟢 New Researcher Follower"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Researcher:*\n{name}"},
                {"type": "mrkdwn", "text": f"*Handle:*\n<https://twitter.com/{handle}|@{handle}>"},
                {"type": "mrkdwn", "text": f"*Focus:*\n{focus}"},
                {"type": "mrkdwn", "text": f"*Detected:*\n{datetime.now().strftime('%b %d, %Y %H:%M')}"},
            ],
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Priority: *HIGH* — reply or DM within 24h to capitalise on the signal."}
            ],
        },
    ]
    return send_slack_message(blocks, text=f"🟢 {name} (@{handle}) is now following GI!")


def alert_lost_follower(researcher: dict) -> bool:
    """Alert when a tracked researcher unfollows GI."""
    name = researcher["name"]
    handle = researcher["twitter"]
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "🔴 Researcher Unfollowed"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Researcher:*\n{name}"},
                {"type": "mrkdwn", "text": f"*Handle:*\n<https://twitter.com/{handle}|@{handle}>"},
                {"type": "mrkdwn", "text": f"*Detected:*\n{datetime.now().strftime('%b %d, %Y %H:%M')}"},
            ],
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "Priority: *MEDIUM* — review recent GI posts for context."}],
        },
    ]
    return send_slack_message(blocks, text=f"🔴 {name} (@{handle}) unfollowed GI.")


def alert_daily_summary(results: dict) -> bool:
    """Send a daily digest of current researcher coverage."""
    following = results.get("following_you", 0)
    total = results.get("total_tracked", 0)
    coverage = results.get("coverage_pct", 0.0)
    matches = results.get("matches", [])

    followers_text = "\n".join(
        f"• {r['name']} (@{r['twitter']})" for r in matches
    ) or "_None yet_"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "📊 Daily GI Researcher Summary"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Researchers tracked:*\n{total}"},
                {"type": "mrkdwn", "text": f"*Following GI:*\n{following}"},
                {"type": "mrkdwn", "text": f"*Coverage score:*\n{coverage}%"},
                {"type": "mrkdwn", "text": f"*As of:*\n{datetime.now().strftime('%b %d, %Y %H:%M')}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Who follows GI:*\n{followers_text}"},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": "GI Researcher Intelligence Dashboard"}],
        },
    ]
    return send_slack_message(blocks, text=f"Daily summary: {following}/{total} researchers following GI ({coverage}%)")


def detect_and_alert_changes(old_results_path: str, new_results: dict) -> dict:
    """
    Compare previous results.json against new results.
    Returns {"new_followers": [...], "lost_followers": [...], "alerts_sent": int}
    """
    summary = {"new_followers": [], "lost_followers": [], "alerts_sent": 0}

    try:
        with open(old_results_path) as f:
            old = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # First run — no comparison possible
        return summary

    old_handles = {r["twitter"].lower() for r in old.get("matches", [])}
    new_handles = {r["twitter"].lower() for r in new_results.get("matches", [])}

    # Researchers who newly started following
    new_follower_handles = new_handles - old_handles
    for researcher in new_results.get("matches", []):
        if researcher["twitter"].lower() in new_follower_handles:
            summary["new_followers"].append(researcher)
            if alert_new_follower(researcher):
                summary["alerts_sent"] += 1

    # Researchers who unfollowed
    lost_handles = old_handles - new_handles
    for researcher in old.get("matches", []):
        if researcher["twitter"].lower() in lost_handles:
            summary["lost_followers"].append(researcher)
            if alert_lost_follower(researcher):
                summary["alerts_sent"] += 1

    return summary


def send_test_alert() -> bool:
    """Send a test message to verify the webhook is working."""
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "✅ GI Slack Alerts Connected"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Your GI Researcher Tracker is connected to Slack.\nYou'll receive alerts when target researchers follow, unfollow, or engage with GI's account.",
            },
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"Test sent: {datetime.now().strftime('%b %d, %Y %H:%M')}"}
            ],
        },
    ]
    return send_slack_message(blocks, text="✅ GI Researcher Tracker connected to Slack!")
