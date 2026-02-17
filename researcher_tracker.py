import tweepy
import json
import os
import shutil
from datetime import datetime
from dotenv import load_dotenv
from researchers import RESEARCHERS
from slack_alerts import detect_and_alert_changes, alert_daily_summary

load_dotenv()

client = tweepy.Client(
    bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
    consumer_key=os.getenv("TWITTER_API_KEY"),
    consumer_secret=os.getenv("TWITTER_API_SECRET"),
    access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
    access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
    wait_on_rate_limit=True
)

YOUR_USERNAME = os.getenv("YOUR_TWITTER_USERNAME")

def get_user_id(username):
    try:
        user = client.get_user(username=username)
        return user.data.id if user.data else None
    except Exception as e:
        print("Error: " + str(e))
        return None

def get_my_followers():
    print("Fetching followers...")
    try:
        my_id = get_user_id(YOUR_USERNAME)
        if not my_id:
            return []
        followers = []
        for follower in tweepy.Paginator(
            client.get_users_followers,
            id=my_id,
            max_results=1000,
            limit=5
        ).flatten(limit=1000):
            followers.append({"id": follower.id, "username": follower.username, "name": follower.name})
        print("Found " + str(len(followers)) + " followers")
        return followers
    except Exception as e:
        print("Error: " + str(e))
        return []

def match_researchers(followers):
    print("Matching researchers...")
    follower_usernames = {f["username"].lower() for f in followers}
    matches = []
    not_following = []
    for researcher in RESEARCHERS:
        handle = researcher["twitter"].lower()
        if handle in follower_usernames:
            matches.append(researcher)
            print("  MATCH: " + researcher["name"])
        else:
            not_following.append(researcher)
            print("  Missing: " + researcher["name"])
    return matches, not_following

def save_results(matches, not_following):
    results = {
        "last_updated": datetime.now().isoformat(),
        "your_account": YOUR_USERNAME,
        "total_tracked": len(RESEARCHERS),
        "following_you": len(matches),
        "not_following": len(not_following),
        "coverage_pct": round(len(matches) / len(RESEARCHERS) * 100, 1),
        "matches": matches,
        "not_following_list": not_following
    }
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to results.json")
    return results

def print_summary(results):
    print("=" * 40)
    print("GI RESEARCHER TRACKER RESULTS")
    print("=" * 40)
    print("Account: " + str(results["your_account"]))
    print("Tracked: " + str(results["total_tracked"]))
    print("Following you: " + str(results["following_you"]))
    print("Coverage: " + str(results["coverage_pct"]) + "%")
    if results["matches"]:
        print("WHO FOLLOWS YOU:")
        for r in results["matches"]:
            print("  - " + r["name"] + " (@" + r["twitter"] + ")")
    print("=" * 40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GI Researcher Tracker")
    parser.add_argument("--daily-summary", action="store_true", help="Send a daily Slack summary after the run")
    args = parser.parse_args()

    print("Starting GI Researcher Tracker...")

    # Back up current results before overwriting so we can diff for alerts
    old_results_path = "results_prev.json"
    if os.path.exists("results.json"):
        shutil.copy("results.json", old_results_path)

    followers = get_my_followers()
    matches, not_following = match_researchers(followers)
    results = save_results(matches, not_following)
    print_summary(results)

    # Detect changes and fire Slack alerts
    changes = detect_and_alert_changes(old_results_path, results)
    if changes["new_followers"]:
        print(f"Slack: alerted {len(changes['new_followers'])} new follower(s)")
    if changes["lost_followers"]:
        print(f"Slack: alerted {len(changes['lost_followers'])} lost follower(s)")
    if changes["alerts_sent"] == 0 and not (changes["new_followers"] or changes["lost_followers"]):
        print("Slack: no changes detected")

    if args.daily_summary:
        ok = alert_daily_summary(results)
        print("Slack: daily summary sent" if ok else "Slack: daily summary failed")