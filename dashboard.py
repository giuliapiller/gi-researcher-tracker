import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import os
import re
import time
import requests
import html
import xml.etree.ElementTree as ET
import math
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
from researchers import RESEARCHERS

load_dotenv()

# Global HTML render helper - ensures unsafe_allow_html=True everywhere
def render_html(content):
    st.markdown(content, unsafe_allow_html=True)

# --- Researcher Briefs API Functions ---

def _is_ml_ai_researcher(author):
    """Check if author's papers contain ML/AI keywords, indicating they're an AI researcher."""
    ml_ai_keywords = [
        'neural', 'learning', 'model', 'deep', 'ai', 'artificial intelligence',
        'robot', 'vision', 'language', 'transformer', 'network', 'reinforcement',
        'machine learning', 'computer vision', 'nlp', 'gpt', 'llm', 'embedding',
        'training', 'inference', 'dataset', 'benchmark', 'agent', 'cognitive',
        'recognition', 'generation', 'diffusion', 'attention', 'representation'
    ]

    papers = author.get("papers") or []
    if not papers:
        return False

    # Check top 5 most recent papers for ML/AI keywords
    recent_papers = sorted(papers, key=lambda p: p.get("year") or 0, reverse=True)[:5]

    for paper in recent_papers:
        title = (paper.get("title") or "").lower()
        for kw in ml_ai_keywords:
            if kw in title:
                return True

    return False

def _institution_matches(author, institution):
    """Check if author's affiliations match any institution keywords."""
    if not institution:
        return False

    affiliations = author.get("affiliations") or []
    affiliations_text = " ".join(affiliations).lower()

    # Split institution on / and , to get keywords (e.g., "Stanford / World Labs" -> ["stanford", "world labs"])
    institution_keywords = [kw.strip().lower() for kw in institution.replace("/", ",").split(",")]

    for kw in institution_keywords:
        if kw and len(kw) > 2 and kw in affiliations_text:
            return True

    return False

def fetch_semantic_scholar_papers(researcher_name, institution):
    """Fetch recent papers from Semantic Scholar API with strict author validation."""
    try:
        # Search for author by full name
        search_url = f"https://api.semanticscholar.org/graph/v1/author/search?query={requests.utils.quote(researcher_name)}&fields=name,affiliations,papers.title,papers.year,papers.abstract,papers.paperId"
        resp = requests.get(search_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("data"):
            return []

        # Loop through top 5 candidates and validate each one
        best_author = None
        candidates = data["data"][:5]

        for author in candidates:
            # Validation: institution match OR ML/AI keywords in papers
            if _institution_matches(author, institution):
                best_author = author
                break
            elif _is_ml_ai_researcher(author):
                best_author = author
                break

        # If no candidate passes validation, return empty rather than wrong data
        if not best_author:
            return []

        # Extract papers from the validated author
        author_papers = best_author.get("papers") or []

        # Sort by year descending, take top 5
        author_papers = sorted(author_papers, key=lambda p: p.get("year") or 0, reverse=True)[:5]

        papers = []
        for p in author_papers:
            title = p.get("title", "")
            if not title:
                continue
            paper_id = p.get("paperId", "")
            papers.append({
                "title": title.strip(),
                "summary": (p.get("abstract") or "").strip().replace("\n", " "),
                "link": f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else "",
                "published": str(p.get("year", "")) if p.get("year") else "",
            })

        return papers
    except Exception as e:
        return []

def fetch_github_repos(github_username):
    """Fetch recent public repos from GitHub for a researcher."""
    if not github_username:
        return []
    url = f"https://api.github.com/users/{github_username}/repos?sort=updated&per_page=5"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        repos = []
        for repo in resp.json():
            repos.append({
                "name": repo.get("name", ""),
                "description": repo.get("description") or "No description",
                "url": repo.get("html_url", ""),
                "updated_at": repo.get("updated_at", "")[:10] if repo.get("updated_at") else "",
            })
        return repos
    except Exception:
        return []

def extract_key_terms(text):
    """Extract key technical terms from paper titles/abstracts."""
    text_lower = text.lower()
    # Technical terms relevant to AI/ML research and GI's domains
    term_patterns = [
        (r'\b(world model|world-model)s?\b', 'world models'),
        (r'\b(reinforcement learning|rl)\b', 'reinforcement learning'),
        (r'\b(game|games|gaming|gameplay)\b', 'game AI'),
        (r'\b(video|vision|visual|image|images)\b', 'computer vision'),
        (r'\b(spatial|3d|three-dimensional|geometry|geometric)\b', 'spatial understanding'),
        (r'\b(temporal|time series|sequence|sequential)\b', 'temporal modeling'),
        (r'\b(transformer|attention|self-attention)\b', 'transformers'),
        (r'\b(language model|llm|gpt|large language)\b', 'language models'),
        (r'\b(tokeniz|token|byte-level|bpe)\b', 'tokenization'),
        (r'\b(robot|robotics|manipulation|embodied)\b', 'robotics'),
        (r'\b(simulation|simulator|simulated)\b', 'simulation'),
        (r'\b(predict|prediction|forecasting)\b', 'prediction'),
        (r'\b(representation|embedding|latent)\b', 'representation learning'),
        (r'\b(policy|planning|decision)\b', 'decision-making'),
        (r'\b(multimodal|multi-modal|vision-language)\b', 'multimodal AI'),
        (r'\b(diffusion|generative|generation)\b', 'generative models'),
        (r'\b(meta-learning|few-shot|adaptation)\b', 'meta-learning'),
        (r'\b(benchmark|evaluation|dataset)\b', 'benchmarking'),
        (r'\b(scaling|scale|large-scale)\b', 'scaling'),
        (r'\b(agent|agents|agentic)\b', 'AI agents'),
        (r'\b(reasoning|abstract|abstraction)\b', 'reasoning'),
        (r'\b(ethics|bias|fairness|harm)\b', 'AI ethics'),
    ]
    found_terms = set()
    for pattern, term in term_patterns:
        if re.search(pattern, text_lower):
            found_terms.add(term)
    return list(found_terms)

def summarize_abstract(abstract, max_len=180):
    """Create a one-sentence summary from an abstract."""
    if not abstract:
        return "No abstract available."
    # Take first sentence or truncate
    sentences = re.split(r'(?<=[.!?])\s+', abstract)
    first_sentence = sentences[0] if sentences else abstract
    if len(first_sentence) <= max_len:
        return first_sentence
    # Truncate at word boundary
    truncated = first_sentence[:max_len].rsplit(' ', 1)[0]
    return truncated + "..."

def generate_gi_angle(researcher, papers):
    """Generate GI engagement angle based on actual paper content."""
    if not papers:
        return f"No recent ArXiv papers found for {researcher['name']}. Consider monitoring their Twitter (@{researcher['twitter']}) for research updates that may connect to GI's spatial-temporal AI and world model work."

    # Combine all paper titles and abstracts for analysis
    all_text = " ".join([p["title"] + " " + p["summary"] for p in papers])
    key_terms = extract_key_terms(all_text)

    # Get the most recent paper title for specific reference
    recent_title = papers[0]["title"]

    # GI's core domains for matching
    gi_domains = {
        "world models": "GI's core thesis around learning world models from gameplay data",
        "game AI": "Medal's gaming telemetry and player behavior modeling",
        "spatial understanding": "GI's spatial-temporal AI pipeline for game environments",
        "temporal modeling": "Medal's sequential gameplay data and event prediction",
        "reinforcement learning": "potential RL applications on Medal's reward-rich gaming data",
        "simulation": "game environments as low-cost simulation testbeds",
        "prediction": "GI's predictive modeling on player behavior streams",
        "representation learning": "learning representations from Medal's multimodal gaming data",
        "computer vision": "Medal's video understanding and highlight detection systems",
        "transformers": "GI's sequence modeling infrastructure for gameplay events",
        "AI agents": "autonomous agents trained on gaming interaction data",
        "robotics": "sim-to-real transfer using game physics as pre-training",
        "multimodal AI": "Medal's combined video, audio, and event stream data",
        "generative models": "content generation from gaming priors",
        "language models": "GI's work on gaming-native language understanding",
    }

    # Find overlapping terms
    relevant_connections = []
    for term in key_terms:
        if term in gi_domains:
            relevant_connections.append((term, gi_domains[term]))

    if not relevant_connections:
        # Fallback: map their focus area to the closest GI domain
        focus_lower = researcher['focus'].lower()
        focus_to_gi = [
            (["vision", "image", "visual", "video"], "computer vision", "Medal's video understanding and highlight detection"),
            (["nlp", "language", "text"], "NLP", "GI's gaming-native language models"),
            (["rl", "reinforcement", "policy"], "reinforcement learning", "Medal's reward-rich gameplay data for RL research"),
            (["robot", "embodied", "manipulation"], "robotics", "sim-to-real transfer using game physics"),
            (["world model", "simulation"], "world models", "GI's core world model research agenda"),
            (["game", "play"], "game AI", "Medal's gaming telemetry and behavior modeling"),
            (["cognitive", "reasoning", "abstract"], "reasoning", "GI's work on abstraction in gameplay understanding"),
            (["ethics", "fairness", "bias"], "AI ethics", "responsible AI practices in gaming systems"),
            (["generative", "diffusion"], "generative models", "content generation from gaming priors"),
            (["foundation", "large", "scale"], "foundation models", "GI's large-scale gameplay pre-training"),
        ]
        closest_domain = None
        closest_gi_connection = None
        for keywords, domain, gi_connection in focus_to_gi:
            if any(kw in focus_lower for kw in keywords):
                closest_domain = domain
                closest_gi_connection = gi_connection
                break

        if closest_domain:
            return f"{researcher['name']}'s focus on {researcher['focus']} places them in the {closest_domain} space. Their recent paper \"{recent_title}\" doesn't directly overlap with GI's current technical stack, but their expertise is closest to {closest_gi_connection}. Consider tracking their work for methodological insights that could transfer to gaming contexts."
        else:
            return f"{researcher['name']} works on {researcher['focus']}. Their recent paper \"{recent_title}\" is outside GI's core domains (world models, spatial-temporal AI, gaming data). However, foundational ML advances often find unexpected applications—monitor for work that could inform GI's representation learning or evaluation methods."

    # Build specific prose based on actual connections
    primary_term, primary_connection = relevant_connections[0]

    prose = f"{researcher['name']}'s recent paper \"{recent_title}\" directly touches on {primary_term}, which connects to {primary_connection}. "

    if len(relevant_connections) > 1:
        secondary_terms = [t[0] for t in relevant_connections[1:3]]
        prose += f"Their broader work also spans {', '.join(secondary_terms)}, suggesting multiple potential collaboration vectors. "

    # Add specific tactical angle
    if "world models" in key_terms or "simulation" in key_terms:
        prose += "This positions them as a high-priority contact for GI's world model research agenda."
    elif "game AI" in key_terms or "reinforcement learning" in key_terms:
        prose += "Medal's dense reward signals from competitive gaming could provide unique training data for their research direction."
    elif "spatial understanding" in key_terms or "temporal modeling" in key_terms:
        prose += "GI's spatial-temporal pipelines could benefit from their methodological advances."
    else:
        prose += "A technical exchange could yield insights applicable to GI's gaming-first AI development."

    return prose

def generate_opening_move(researcher, papers):
    """Generate a specific non-promotional opening move based on recent papers."""
    if not papers:
        return f"Monitor @{researcher['twitter']} for new research announcements. When they post about a new paper or project, engage with a substantive technical question rather than promotional content."

    recent = papers[0]
    title = recent["title"]
    summary = recent["summary"]
    key_terms = extract_key_terms(title + " " + summary)

    # Extract a concrete detail from the abstract for specificity
    # Look for numbers, metrics, or specific claims
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|x|X)?\b', summary)

    # Generate specific opening based on content
    if "world models" in key_terms or "simulation" in key_terms:
        move = f"Respond to their paper \"{title}\" with a question about how their world model approach handles distribution shift between training environments—reference GI's observation that gaming environments have unusually consistent physics but high behavioral variance."
    elif "game AI" in key_terms:
        move = f"Reply to \"{title}\" asking whether they've considered competitive multiplayer games as a benchmark domain, noting the unique properties of adversarial human opponents vs. scripted NPCs."
    elif "reinforcement learning" in key_terms:
        move = f"Comment on \"{title}\" with a question about sample efficiency in their approach—gaming domains offer dense reward signals that could stress-test their method in interesting ways."
    elif "temporal modeling" in key_terms or "sequence" in key_terms:
        move = f"Engage with \"{title}\" by asking about handling variable-length sequences with irregular timestamps—a core challenge in gameplay event modeling."
    elif "computer vision" in key_terms or "video" in key_terms:
        move = f"Reference \"{title}\" when asking about their approach to fast-motion scenarios—gaming video has unique temporal characteristics (high APM, rapid state changes) worth discussing."
    elif "ethics" in key_terms or "bias" in key_terms:
        move = f"Engage thoughtfully with \"{title}\" by asking how their framework might apply to gaming AI systems that influence player behavior and attention."
    elif "benchmark" in key_terms:
        move = f"Comment on \"{title}\" suggesting gaming environments as a potential benchmark domain—offer specific observations about what makes game data distinctive rather than pitching GI directly."
    else:
        # Personalized fallback based on researcher's focus area
        focus_area = researcher.get('focus', 'AI research').split(',')[0].strip()
        move = f"Follow {researcher['name']}'s work on {focus_area}. When they post about {focus_area.lower()}, engage with a specific technical question connecting it to GI's gaming data pipeline."

    # Add tactical note
    move += f" Post this as a reply on Twitter (@{researcher['twitter']}) or as a comment if they share on LinkedIn."

    return move

def load_researcher_briefs():
    """Load all researcher briefs with rate limiting."""
    briefs = []
    progress_bar = st.progress(0, text="Loading researcher data...")

    for i, researcher in enumerate(RESEARCHERS):
        progress_bar.progress((i + 1) / len(RESEARCHERS), text=f"Fetching data for {researcher['name']}...")

        # Fetch papers from Semantic Scholar
        papers = fetch_semantic_scholar_papers(researcher["name"], researcher.get("institution", ""))
        time.sleep(1)  # Rate limit for Semantic Scholar

        # Fetch GitHub repos
        repos = fetch_github_repos(researcher.get("github"))
        if researcher.get("github"):
            time.sleep(0.5)  # Rate limit for GitHub

        # Generate analysis
        gi_angle = generate_gi_angle(researcher, papers)
        opening_move = generate_opening_move(researcher, papers)

        briefs.append({
            "researcher": researcher,
            "papers": papers,
            "repos": repos,
            "gi_angle": gi_angle,
            "opening_move": opening_move,
        })

    progress_bar.empty()
    return briefs

# --- GI in the Wild: Community Monitoring Functions ---

# Direct mention queries (exact phrases for GI-specific content)
DIRECT_MENTION_QUERIES = [
    '"General Intuition" AI',
    '"General Intuition" startup',
    '"Pim de Witte" AI',
    '"Pim de Witte" Medal',
    '"Pim de Witte" startup',
    '"Medal AI"',
    'gen_intuition',
]

# Adjacent conversation queries (topics GI should be part of)
ADJACENT_QUERIES = [
    '"gaming AI" "world model"',
    '"video game" "training data" AI',
    '"game AI" foundation model',
    '"spatial AI" gaming',
    'world models video games',
]

# Allowed subreddits for Reddit search
ALLOWED_SUBREDDITS = [
    "MachineLearning", "artificial", "LocalLLaMA", "singularity",
    "ChatGPT", "OpenAI", "Futurology", "technology", "programming"
]

# Relevance filter keywords - must contain at least one
RELEVANCE_KEYWORDS = [
    'ai', 'artificial intelligence', 'machine learning', 'ml', 'startup',
    'model', 'data', 'tech', 'gaming', 'neural', 'training', 'deep learning',
    'foundation model', 'world model', 'reinforcement', 'video game'
]

def is_relevant(text):
    """Check if text contains at least one relevance keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in RELEVANCE_KEYWORDS)

def is_direct_gi_mention(text):
    """Check if text is a direct mention of GI, Pim, or Medal in context."""
    text_lower = text.lower()
    # Must mention GI/Pim/Medal AND be relevant (not just a name match)
    gi_terms = ['general intuition', 'pim de witte', 'medal ai', 'gen_intuition', '@gen_intuition']
    has_gi_term = any(term in text_lower for term in gi_terms)
    return has_gi_term and is_relevant(text)

def detect_sentiment(text):
    """Simple keyword-based sentiment detection."""
    text_lower = text.lower()

    positive_keywords = [
        'impressive', 'interesting', 'cool', 'amazing', 'great', 'awesome',
        'brilliant', 'innovative', 'exciting', 'promising', 'solid', 'love',
        'fantastic', 'game-changer', 'breakthrough', 'clever', 'bullish'
    ]
    skeptical_keywords = [
        'hype', 'overfunded', 'unclear', 'skeptical', 'doubt', 'suspicious',
        'overrated', 'vaporware', 'scam', 'buzzword', 'questionable', 'meh',
        'disappointing', 'overhyped', 'nothing new', 'just another', 'bearish'
    ]

    positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
    skeptical_count = sum(1 for kw in skeptical_keywords if kw in text_lower)

    if positive_count > skeptical_count:
        return "positive"
    elif skeptical_count > positive_count:
        return "skeptical"
    else:
        return "neutral"

def is_featured_mention(item):
    """Check if this is a high-quality featured mention (OpenAI/Medal story, high upvotes)."""
    text_lower = item.get("text", "").lower()
    upvotes = item.get("upvotes", 0)

    # High-signal indicators
    has_openai = 'openai' in text_lower or '$500m' in text_lower or '500 million' in text_lower
    has_medal = 'medal' in text_lower
    has_pim = 'pim de witte' in text_lower
    has_gi = 'general intuition' in text_lower
    high_upvotes = upvotes >= 50

    return (has_openai and has_medal) or (has_pim and high_upvotes) or (has_gi and high_upvotes)

def search_hacker_news(queries, is_direct=True):
    """Search Hacker News via Algolia API."""
    results = []
    seen_ids = set()

    for query in queries:
        try:
            url = f"https://hn.algolia.com/api/v1/search?query={requests.utils.quote(query)}&tags=(story,comment)&hitsPerPage=15"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for hit in data.get("hits", []):
                obj_id = hit.get("objectID")
                if obj_id in seen_ids:
                    continue
                seen_ids.add(obj_id)

                # Get text content
                title = hit.get("title") or ""
                comment_text = hit.get("comment_text") or ""
                text = title if title else comment_text

                # Clean HTML from comment text
                if comment_text:
                    text = re.sub(r'<[^>]+>', '', comment_text)

                if not text:
                    continue

                # Apply relevance filter
                if not is_relevant(text):
                    continue

                result = {
                    "source": "HN",
                    "text": text[:500] + ("..." if len(text) > 500 else ""),
                    "author": hit.get("author", "unknown"),
                    "date": hit.get("created_at", "")[:10],
                    "url": f"https://news.ycombinator.com/item?id={obj_id}",
                    "upvotes": hit.get("points") or hit.get("num_comments") or 0,
                    "sentiment": detect_sentiment(text),
                    "query": query,
                    "is_direct": is_direct and is_direct_gi_mention(text),
                }
                results.append(result)

            time.sleep(0.5)  # Rate limit
        except Exception:
            continue

    return results

def search_reddit(queries, is_direct=True):
    """Search Reddit using public JSON API with subreddit filtering."""
    results = []
    seen_ids = set()

    headers = {
        "User-Agent": "GI-Research-Monitor/1.0"
    }

    # Build subreddit restriction string
    subreddit_filter = " OR ".join([f"subreddit:{sr}" for sr in ALLOWED_SUBREDDITS])

    for query in queries:
        try:
            # Search with subreddit restriction
            full_query = f"{query} ({subreddit_filter})"
            url = f"https://www.reddit.com/search.json?q={requests.utils.quote(full_query)}&sort=relevance&t=month&limit=15"
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            for post in data.get("data", {}).get("children", []):
                post_data = post.get("data", {})
                post_id = post_data.get("id")

                if post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                subreddit = post_data.get("subreddit", "")

                # Filter to allowed subreddits only
                if subreddit not in ALLOWED_SUBREDDITS:
                    continue

                title = post_data.get("title", "")
                selftext = post_data.get("selftext", "")[:300]
                text = f"{title} {selftext}".strip()

                if not text:
                    continue

                # Apply relevance filter
                if not is_relevant(text):
                    continue

                # Convert Unix timestamp to date
                created = post_data.get("created_utc", 0)
                date_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else ""

                result = {
                    "source": "Reddit",
                    "subreddit": subreddit,
                    "text": text[:500] + ("..." if len(text) > 500 else ""),
                    "author": post_data.get("author", "unknown"),
                    "date": date_str,
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "upvotes": post_data.get("score", 0),
                    "sentiment": detect_sentiment(text),
                    "query": query,
                    "is_direct": is_direct and is_direct_gi_mention(text),
                }
                results.append(result)

            time.sleep(1)  # Rate limit (Reddit is stricter)
        except Exception:
            continue

    return results

def is_recent(item, max_years=2):
    """Check if result is within the last N years."""
    date_str = item.get("date", "")
    if not date_str:
        return False
    try:
        result_date = datetime.strptime(date_str[:10], "%Y-%m-%d")
        cutoff = datetime.now().replace(year=datetime.now().year - max_years)
        return result_date >= cutoff
    except (ValueError, TypeError):
        return False

def load_community_mentions():
    """Load mentions from HN and Reddit, split into direct and adjacent."""
    progress = st.progress(0, text="Searching for direct GI mentions...")

    # Search for direct GI mentions
    hn_direct = search_hacker_news(DIRECT_MENTION_QUERIES, is_direct=True)
    progress.progress(0.25, text="Searching Reddit for direct mentions...")

    reddit_direct = search_reddit(DIRECT_MENTION_QUERIES, is_direct=True)
    progress.progress(0.5, text="Searching for adjacent conversations...")

    # Search for adjacent conversations
    hn_adjacent = search_hacker_news(ADJACENT_QUERIES, is_direct=False)
    progress.progress(0.75, text="Searching Reddit for adjacent topics...")

    reddit_adjacent = search_reddit(ADJACENT_QUERIES, is_direct=False)
    progress.progress(0.9, text="Processing results...")

    # Combine and categorize
    all_direct = hn_direct + reddit_direct
    all_adjacent = hn_adjacent + reddit_adjacent

    # Filter to recent results only (within last 2 years)
    all_direct = [r for r in all_direct if is_recent(r)]
    all_adjacent = [r for r in all_adjacent if is_recent(r)]

    # Strict GI keyword filter for direct mentions
    def contains_gi_keywords(item):
        """Check if item contains GI-specific keywords."""
        text = (item.get("text", "") + " " + item.get("title", "")).lower()
        gi_keywords = ["general intuition", "pim de witte", "medal ai", "gen_intuition", "@gen_intuition"]
        return any(kw in text for kw in gi_keywords)

    # Filter direct mentions to only those that truly mention GI
    direct_mentions = [r for r in all_direct if r.get("is_direct", False) and contains_gi_keywords(r)]

    # Find featured mentions (high-quality signals)
    featured = [r for r in direct_mentions if is_featured_mention(r)]
    non_featured_direct = [r for r in direct_mentions if not is_featured_mention(r)]

    # Sort by upvotes then date
    featured.sort(key=lambda x: (x.get("upvotes", 0), x.get("date", "")), reverse=True)
    non_featured_direct.sort(key=lambda x: (x.get("date", ""), x.get("upvotes", 0)), reverse=True)
    all_adjacent.sort(key=lambda x: (x.get("upvotes", 0), x.get("date", "")), reverse=True)

    # Calculate sentiment for direct mentions
    if direct_mentions:
        sentiments = [r["sentiment"] for r in direct_mentions]
        positive = sentiments.count("positive")
        skeptical = sentiments.count("skeptical")
        neutral = sentiments.count("neutral")

        if positive > skeptical and positive > neutral:
            overall = "positive"
        elif skeptical > positive and skeptical > neutral:
            overall = "skeptical"
        else:
            overall = "neutral"
    else:
        overall = "neutral"

    progress.empty()

    # Check if we have limited results
    total_direct = len(featured) + len(non_featured_direct)
    limited_direct = total_direct > 0 and total_direct < 3
    limited_adjacent = len(all_adjacent) > 0 and len(all_adjacent) < 3

    return {
        "featured": featured[:3],
        "direct": non_featured_direct[:10],
        "adjacent": all_adjacent[:15],
        "direct_count": len(direct_mentions),
        "adjacent_count": len(all_adjacent),
        "overall_sentiment": overall,
        "limited_direct": limited_direct,
        "limited_adjacent": limited_adjacent,
    }

def render_mention_card(item, is_featured=False):
    """Render a community mention card with HTML styling."""
    sentiment_colors = {"positive": "#00ff88", "neutral": "#888888", "skeptical": "#ff6b6b"}
    sentiment_icons = {"positive": "🟢", "neutral": "⚪", "skeptical": "🔴"}

    source = item["source"]
    source_color = "#ff6600" if source == "HN" else "#ff4500"
    source_label = "HN" if source == "HN" else f"r/{item.get('subreddit', 'Reddit')}"

    sentiment = item["sentiment"]
    sentiment_icon = sentiment_icons.get(sentiment, "⚪")
    sentiment_color = sentiment_colors.get(sentiment, "#888")

    upvotes = item.get("upvotes", 0)
    upvote_display = f"▲ {upvotes}" if upvotes > 0 else ""

    text_preview = html.escape(item["text"])
    author = html.escape(item["author"])
    date = item.get("date", "")
    url = html.escape(item.get("url", "#"))

    border_color = "#00ff88" if is_featured else "#2d3250"
    bg_color = "#1a2a1a" if is_featured else "#1e2130"
    featured_badge = '<span style="background: #00ff88; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-right: 8px;">⭐ FEATURED</span>' if is_featured else ""

    card_html = f"""
    <div style="background: {bg_color}; border-radius: 8px; padding: 16px; margin-bottom: 12px; border: 2px solid {border_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div>
                {featured_badge}
                <span style="background: {source_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{source_label}</span>
                <span style="color: {sentiment_color}; margin-left: 8px;">{sentiment_icon} {sentiment}</span>
            </div>
            <div style="color: #888; font-size: 12px;">
                <span style="color: #00d4ff; font-weight: bold;">{upvote_display}</span>
                <span style="margin-left: 10px;">{date}</span>
            </div>
        </div>
        <div style="color: #e0e0e0; font-size: 14px; line-height: 1.5; margin-bottom: 10px;">
            {text_preview}
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #666; font-size: 12px;">by {author}</span>
            <a href="{url}" target="_blank" style="color: #00d4ff; font-size: 12px; text-decoration: none;">View conversation →</a>
        </div>
    </div>
    """
    render_html(card_html)

# Mock data simulating what GI/Pim's account would show
MOCK_RESULTS = {
    "last_updated": datetime.now().isoformat(),
    "your_account": "gen_intuition",
    "total_tracked": 20,
    "following_you": 7,
    "not_following": 13,
    "coverage_pct": 35.0,
    "matches": [
        {"name": "Andrej Karpathy", "twitter": "karpathy", "focus": "Deep Learning, Autonomous Driving"},
        {"name": "Jim Fan", "twitter": "drjimfan", "focus": "Embodied AI, Foundation Models"},
        {"name": "Pieter Abbeel", "twitter": "pabbeel", "focus": "Robotics, Reinforcement Learning"},
        {"name": "Fei-Fei Li", "twitter": "drfeifei", "focus": "Computer Vision, Spatial AI"},
        {"name": "Sergey Levine", "twitter": "svlevine", "focus": "Robotics, Deep RL"},
        {"name": "Chelsea Finn", "twitter": "chelseabfinn", "focus": "Meta-Learning, Robotics"},
        {"name": "Oriol Vinyals", "twitter": "oriolvinyals", "focus": "Deep Learning, Game AI"},
    ],
    "not_following_list": [
        {"name": "Yann LeCun", "twitter": "ylecun", "focus": "Deep Learning, Computer Vision"},
        {"name": "Demis Hassabis", "twitter": "demishassabis", "focus": "AGI, World Models"},
        {"name": "Jeff Dean", "twitter": "jeffdean", "focus": "Large Scale ML"},
        {"name": "Ilya Sutskever", "twitter": "ilyasut", "focus": "Deep Learning, AGI"},
        {"name": "Joscha Bach", "twitter": "joscha_bach", "focus": "Cognitive AI, World Models"},
        {"name": "Emad Mostaque", "twitter": "emostaque", "focus": "Generative AI"},
        {"name": "Francois Chollet", "twitter": "fchollet", "focus": "Deep Learning, ARC"},
        {"name": "Lex Fridman", "twitter": "lexfridman", "focus": "AI Research, Autonomous Vehicles"},
        {"name": "Gary Marcus", "twitter": "garymarcus", "focus": "Cognitive Science, AI Critique"},
        {"name": "Kyunghyun Cho", "twitter": "kchonyc", "focus": "NLP, Generative Models"},
        {"name": "David Silver", "twitter": "davidsilvermind", "focus": "Reinforcement Learning"},
        {"name": "Abeba Birhane", "twitter": "Abebab", "focus": "AI Ethics, Cognitive Science"},
        {"name": "Percy Liang", "twitter": "percyliang", "focus": "Foundation Models, NLP"},
    ],
    "recent_alerts": [
        {"date": "2026-02-17", "type": "New Follower", "researcher": "Andrej Karpathy", "twitter": "karpathy", "priority": "HIGH"},
        {"date": "2026-02-16", "type": "Liked Post", "researcher": "Jim Fan", "twitter": "drjimfan", "priority": "HIGH"},
        {"date": "2026-02-15", "type": "Retweeted", "researcher": "Pieter Abbeel", "twitter": "pabbeel", "priority": "MEDIUM"},
        {"date": "2026-02-14", "type": "New Follower", "researcher": "Chelsea Finn", "twitter": "chelseabfinn", "priority": "HIGH"},
        {"date": "2026-02-13", "type": "Replied", "researcher": "Sergey Levine", "twitter": "svlevine", "priority": "MEDIUM"},
    ]
}

# Page config
st.set_page_config(
    page_title="GI Researcher Tracker",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
render_html("""
<style>
    .main { background-color: #0a0a0f; }
    .stApp { background-color: #0a0a0f; }
    [data-testid="stAppViewContainer"] { background-color: #0a0a0f; }
    [data-testid="stHeader"] { background-color: #0a0a0f; }
    .brief-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border: 1px solid #2d3250;
    }
    .brief-header {
        border-bottom: 1px solid #2d3250;
        padding-bottom: 12px;
        margin-bottom: 16px;
    }
    .brief-section {
        background: #161922;
        border-radius: 8px;
        padding: 14px;
        margin: 12px 0;
    }
    .brief-section-title {
        color: #00d4ff;
        font-weight: bold;
        margin-bottom: 8px;
        font-size: 14px;
    }
    .paper-item {
        padding: 8px 0;
        border-bottom: 1px solid #2d3250;
    }
    .paper-item:last-child {
        border-bottom: none;
    }
    .repo-item {
        padding: 6px 0;
    }
    .gi-angle {
        background: #1f3320;
        border-left: 6px solid #00ff88;
        padding: 18px;
        border-radius: 6px;
        margin: 16px 0;
        line-height: 1.7;
    }
    .gi-angle .section-title {
        color: #00ff88;
        font-weight: bold;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .gi-angle .section-subtitle {
        color: #888;
        font-size: 12px;
        margin-bottom: 12px;
        font-style: italic;
    }
    .gi-angle .section-body {
        color: #f0f0f0;
        font-size: 14px;
    }
    .opening-move {
        background: #332e10;
        border-left: 6px solid #ffcc00;
        padding: 18px;
        border-radius: 6px;
        margin: 16px 0;
        line-height: 1.7;
    }
    .opening-move .section-title {
        color: #ffcc00;
        font-weight: bold;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .opening-move .section-subtitle {
        color: #888;
        font-size: 12px;
        margin-bottom: 12px;
        font-style: italic;
    }
    .opening-move .section-body {
        color: #f0f0f0;
        font-size: 14px;
    }
</style>
""")

# Load data
try:
    with open("results.json") as f:
        data = json.load(f)
except:
    data = MOCK_RESULTS

# Header
render_html('<h1 style="color: #ffffff; font-size: 42px; margin-bottom: 0;">General Intuition</h1>')
render_html('<p style="color: #cccccc; font-size: 16px; margin-top: 4px;">Research Community Intelligence</p>')

# Create tabs
tab_dashboard, tab_briefs, tab_wild, tab_weekly = st.tabs(["Dashboard", "Researcher Briefs", "GI in the Wild", "Weekly Brief"])

# ============ DASHBOARD TAB ============
with tab_dashboard:
    # Mock engagement data for orbit map
    ORBIT_DATA = {
        "following_engaged": [
            {"name": "Demis Hassabis", "twitter": "demishassabis", "institution": "Google DeepMind", "followers": 180000, "last_interaction": "Liked post 2 days ago"},
            {"name": "Jim Fan", "twitter": "drjimfan", "institution": "NVIDIA", "followers": 95000, "last_interaction": "Replied to thread yesterday"},
            {"name": "Andrej Karpathy", "twitter": "karpathy", "institution": "Independent", "followers": 850000, "last_interaction": "Retweeted 3 days ago"},
        ],
        "following_only": [
            {"name": "Yann LeCun", "twitter": "ylecun", "institution": "Meta AI / NYU", "followers": 720000, "last_interaction": "Followed 2 weeks ago"},
            {"name": "Chelsea Finn", "twitter": "chelseabfinn", "institution": "Stanford", "followers": 45000, "last_interaction": "Followed 1 month ago"},
            {"name": "Pieter Abbeel", "twitter": "pabbeel", "institution": "UC Berkeley / Covariant", "followers": 68000, "last_interaction": "Followed 3 weeks ago"},
            {"name": "Percy Liang", "twitter": "percyliang", "institution": "Stanford", "followers": 32000, "last_interaction": "Followed 1 week ago"},
        ],
        "not_following": [
            {"name": "Ilya Sutskever", "twitter": "ilyasut", "institution": "SSI", "followers": 520000},
            {"name": "Jeff Dean", "twitter": "jeffdean", "institution": "Google", "followers": 410000},
            {"name": "Fei-Fei Li", "twitter": "drfeifei", "institution": "Stanford / World Labs", "followers": 280000},
            {"name": "Sergey Levine", "twitter": "svlevine", "institution": "UC Berkeley", "followers": 52000},
            {"name": "Oriol Vinyals", "twitter": "oriolvinyals", "institution": "Google DeepMind", "followers": 85000},
            {"name": "David Silver", "twitter": "davidsilvermind", "institution": "Google DeepMind", "followers": 42000},
            {"name": "Joscha Bach", "twitter": "joscha_bach", "institution": "Independent", "followers": 95000},
            {"name": "Francois Chollet", "twitter": "fchollet", "institution": "Google", "followers": 620000},
            {"name": "Gary Marcus", "twitter": "garymarcus", "institution": "NYU", "followers": 180000},
            {"name": "Kyunghyun Cho", "twitter": "kchonyc", "institution": "NYU / Genentech", "followers": 38000},
            {"name": "Lex Fridman", "twitter": "lexfridman", "institution": "MIT", "followers": 3200000},
            {"name": "Emad Mostaque", "twitter": "emostaque", "institution": "Stability AI", "followers": 290000},
            {"name": "Abeba Birhane", "twitter": "Abebab", "institution": "Trinity College Dublin", "followers": 45000},
        ]
    }

    # This week's signals (mock data)
    SIGNALS = [
        {"name": "Andrej Karpathy", "twitter": "karpathy", "action": "Followed GI", "date": "Feb 17"},
        {"name": "Jim Fan", "twitter": "drjimfan", "action": "Liked post about world models", "date": "Feb 16"},
        {"name": "Pieter Abbeel", "twitter": "pabbeel", "action": "Replied to Pim's thread", "date": "Feb 15"},
    ]

    # ========== ELEMENT 1: Cultural Gravity Score ==========
    # Calculate score from mock data
    follow_pct = 7 / 20  # 35%
    engagement_rate = 0.23  # 23% researcher-initiated
    mention_score = 12  # out of 100

    cultural_gravity = int(
        0.40 * (follow_pct * 100) +
        0.40 * (engagement_rate * 100) +
        0.20 * mention_score
    )

    # Simple score display with thin progress bar
    render_html(f"""
    <style>
        .tooltip-container {{
            position: relative;
            display: inline-block;
            cursor: help;
        }}
        .tooltip-container .tooltip-text {{
            visibility: hidden;
            width: 280px;
            background-color: #1e2130;
            color: #ccc;
            text-align: left;
            border-radius: 6px;
            padding: 10px 12px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -140px;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 12px;
            line-height: 1.4;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .tooltip-container:hover .tooltip-text {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
    <div style="text-align: center; padding: 30px 0 20px 0;">
        <div style="font-size: 14px; color: #cccccc; margin-bottom: 8px;">
            Cultural Gravity Score
            <span class="tooltip-container" style="margin-left: 4px;">
                <span style="color: #666; font-size: 12px;">ℹ</span>
                <span class="tooltip-text">Weighted score: researcher following rate (40%), researcher-initiated engagement (40%), community mentions (20%)</span>
            </span>
        </div>
        <div style="font-size: 80px; font-weight: bold; color: #ffffff; line-height: 1;">{cultural_gravity}</div>
        <div style="width: 200px; margin: 16px auto 0 auto;">
            <div style="background: #1e1e1e; height: 4px; border-radius: 2px; overflow: hidden;">
                <div style="background: #00d4ff; height: 100%; width: {cultural_gravity}%;"></div>
            </div>
        </div>
        <div style="font-size: 12px; color: #aaaaaa; margin-top: 12px;">
            Following rate 35%  ·  Initiated engagement 23%  ·  Community mentions 12
        </div>
        <div style="font-size: 14px; color: #00ff88; margin-top: 16px;">↑ 4 points from last week</div>
        <div style="font-size: 14px; color: #aaa; margin-top: 12px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
            3 researchers moved closer this week. Karpathy's GitHub activity suggests active interest in training data approaches.
        </div>
    </div>
    """)

    # ========== ELEMENT 2: Researcher Orbit Map ==========

    # Build orbit data with proper spacing
    orbit_points = []
    CENTER_EXCLUSION = 0.15  # Exclusion zone around GI center text

    def add_researchers(researchers, ring_radius, color, status):
        n = len(researchers)
        angle_step = 2 * math.pi / max(n, 1)
        for i, r in enumerate(researchers):
            angle = angle_step * i + (ring_radius * 0.3)  # offset per ring to avoid alignment
            # Ensure minimum radius to avoid center exclusion zone
            radius = max(ring_radius, CENTER_EXCLUSION + 0.1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            # Normalize follower count to dot size (12-35 range)
            followers = r.get("followers", 50000)
            size = 12 + min(23, followers / 100000 * 23)
            orbit_points.append({
                "x": x, "y": y,
                "name": r["name"],
                "twitter": r["twitter"],
                "institution": r.get("institution", ""),
                "last_interaction": r.get("last_interaction", "Not yet engaged"),
                "color": color,
                "size": size,
                "status": status
            })

    # Add researchers to their orbits (no labels, hover only)
    add_researchers(ORBIT_DATA["following_engaged"], 0.32, "#ffcc00", "Following + Engaged")
    add_researchers(ORBIT_DATA["following_only"], 0.58, "#00d4ff", "Following")
    add_researchers(ORBIT_DATA["not_following"], 0.88, "#444444", "Not yet reached")

    # Create the orbit map
    fig = go.Figure()

    # Draw concentric rings with increased opacity (0.15)
    for r in [0.32, 0.58, 0.88]:
        theta = [i * 2 * math.pi / 100 for i in range(101)]
        ring_x = [r * math.cos(t) for t in theta]
        ring_y = [r * math.sin(t) for t in theta]
        fig.add_trace(go.Scatter(
            x=ring_x, y=ring_y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.15)", width=1),
            hoverinfo="skip",
            showlegend=False
        ))

    # Add researcher dots first (so GI text renders on top)
    for p in orbit_points:
        fig.add_trace(go.Scatter(
            x=[p["x"]], y=[p["y"]],
            mode="markers",
            marker=dict(size=p["size"], color=p["color"], opacity=1.0),
            hovertemplate=(
                f"<b>{p['name']}</b><br>"
                f"{p['institution']}<br>"
                f"<i>{p['last_interaction']}</i>"
                "<extra></extra>"
            ),
            showlegend=False
        ))

    # Add GI at center (rendered last so it's on top)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode="text",
        text=["GI"],
        textfont=dict(size=28, color="#ffffff", family="Arial Black"),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-1.2, 1.2], fixedrange=True
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[-1.2, 1.2], scaleanchor="x", fixedrange=True
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        hoverlabel=dict(
            bgcolor="#1e2130",
            font_size=13,
            font_family="system-ui"
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Legend
    render_html("""
    <div style="display: flex; justify-content: center; gap: 32px; margin: 0 0 30px 0;">
        <span style="color: #ffcc00; font-size: 12px;">● Following + Engaged</span>
        <span style="color: #00d4ff; font-size: 12px;">● Following</span>
        <span style="color: #444; font-size: 12px;">● Not yet reached</span>
    </div>
    """)

    # ========== ELEMENT 3: This Week's Signal ==========
    render_html("""
    <div style="font-size: 14px; color: #cccccc; margin: 20px 0 16px 0;">This Week's Signal</div>
    """)

    signal_cols = st.columns(3)
    for i, signal in enumerate(SIGNALS):
        with signal_cols[i]:
            render_html(f"""
            <div style="background: #1a1a2e; padding: 16px; border-radius: 8px;">
                <div style="font-weight: bold; color: #fff; font-size: 14px;">{signal['name']}</div>
                <div style="color: #00d4ff; font-size: 12px; margin-bottom: 8px;">@{signal['twitter']}</div>
                <div style="color: #aaa; font-size: 13px;">{signal['action']}</div>
                <div style="color: #666; font-size: 11px; margin-top: 6px;">{signal['date']}</div>
            </div>
            """)

    # ========== ELEMENT 4: What Resonates ==========
    render_html("""
    <div style="font-size: 14px; color: #cccccc; margin: 30px 0 16px 0;">What Resonates</div>
    """)

    # Content performance data with specific colors
    content_data = [
        {"type": "Technical/research posts", "pct": 68, "color": "#00d4ff"},
        {"type": "Field commentary", "pct": 41, "color": "#7b8cde"},
        {"type": "Company announcements", "pct": 9, "color": "#444444"},
    ]

    render_html("""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
    """)

    for item in content_data:
        render_html(f"""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <span style="color: #aaa; font-size: 13px; width: 180px; flex-shrink: 0;">{item['type']}</span>
            <div style="flex: 1; height: 8px; background: #1e1e1e; border-radius: 4px; margin: 0 12px; overflow: hidden;">
                <div style="width: {item['pct']}%; height: 100%; background: {item['color']}; border-radius: 4px;"></div>
            </div>
            <span style="color: #fff; font-size: 13px; font-weight: bold; width: 40px; text-align: right;">{item['pct']}%</span>
        </div>
        """)

    render_html("""
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #2a2a2a;">
            <strong style="color: #fff; font-size: 14px;">23 researchers follow World Labs but not GI — this is the gap to close.</strong>
        </div>
    </div>
    """)

# ============ RESEARCHER BRIEFS TAB ============
with tab_briefs:
    render_html("### Researcher Briefs")
    render_html("<p style='color: #cccccc; font-style: italic;'>Deep-dive intelligence on each tracked researcher with GI-specific engagement strategies</p>")

    # Refresh button
    col_refresh, col_spacer = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh Data", key="refresh_briefs"):
            if "researcher_briefs" in st.session_state:
                del st.session_state["researcher_briefs"]
            st.rerun()

    render_html("---")

    # Load or retrieve cached briefs
    if "researcher_briefs" not in st.session_state:
        st.session_state["researcher_briefs"] = load_researcher_briefs()

    briefs = st.session_state["researcher_briefs"]

    # Render each researcher brief
    for brief in briefs:
        researcher = brief["researcher"]
        papers = brief["papers"]
        repos = brief["repos"]
        gi_angle = brief["gi_angle"]
        opening_move = brief["opening_move"]

        # Determine last paper date for staleness indicator
        if papers:
            last_paper_year = papers[0].get("published", "")[:4]
            last_paper_text = f"📄 last paper: {last_paper_year}"
            last_paper_color = "#888"
        else:
            last_paper_text = "📄 no papers found"
            last_paper_color = "#ff6b6b"

        # Build GitHub link if available
        github_user = researcher.get('github')
        github_html = f' · <a href="https://github.com/{github_user}" target="_blank" style="color:#888">GitHub</a>' if github_user else ""

        # Render header card - all fields properly escaped
        render_html(f"""
<div class="brief-card">
<div class="brief-header">
<span style="font-size: 22px; font-weight: bold; color: #00d4ff">{html.escape(researcher['name'])}</span>
<span style="color: {last_paper_color}; font-size: 12px; margin-left: 10px">{last_paper_text}</span>
<br>
<span style="color: #888">{html.escape(researcher.get('institution', 'Independent'))}</span> ·
<span style="color: #666">@{html.escape(researcher['twitter'])}</span>{github_html}
<br>
<span style="color: #ffcc00; font-size: 13px"><strong>Focus:</strong> {html.escape(researcher['focus'])}</span>
</div>
</div>
        """)

        # Recent Papers
        with st.expander(f"📄 Recent Papers ({len(papers)} found)", expanded=True):
            if papers:
                for paper in papers:
                    summary_short = summarize_abstract(paper["summary"])
                    title_safe = html.escape(paper['title'])
                    summary_safe = html.escape(summary_short)
                    link_safe = html.escape(paper['link'])
                    render_html(f"""
                    <div class="paper-item">
                        <a href="{link_safe}" style="color: #00d4ff; text-decoration: none; font-weight: 500">{title_safe}</a>
                        <span style="color: #666; font-size: 12px"> — {paper['published']}</span><br>
                        <span style="color: #999; font-size: 13px">{summary_safe}</span>
                    </div>
                    """)
            else:
                render_html("*No recent ArXiv papers found for this author query.*")

        # GitHub Repos (if available)
        if repos:
            with st.expander(f"💻 Recent GitHub Activity ({len(repos)} repos)", expanded=False):
                for repo in repos:
                    repo_name_safe = html.escape(repo['name'])
                    repo_url_safe = html.escape(repo['url'])
                    repo_desc = repo['description'][:100] + ('...' if len(repo['description']) > 100 else '')
                    repo_desc_safe = html.escape(repo_desc)
                    render_html(f"""
                    <div class="repo-item">
                        <a href="{repo_url_safe}" style="color: #00d4ff; text-decoration: none">{repo_name_safe}</a>
                        <span style="color: #666; font-size: 12px"> — updated {repo['updated_at']}</span><br>
                        <span style="color: #999; font-size: 12px">{repo_desc_safe}</span>
                    </div>
                    """)

        # GI Engagement Angle (escape HTML in generated prose)
        gi_angle_safe = html.escape(gi_angle)
        render_html(f"""
        <div class="gi-angle">
            <div class="section-title">🎯 GI ENGAGEMENT ANGLE</div>
            <div class="section-subtitle">Why this researcher matters for GI right now</div>
            <div class="section-body">{gi_angle_safe}</div>
        </div>
        """)

        # Suggested Opening Move (escape HTML in generated prose)
        opening_move_safe = html.escape(opening_move)
        render_html(f"""
        <div class="opening-move">
            <div class="section-title">💬 SUGGESTED OPENING MOVE</div>
            <div class="section-subtitle">Specific action to take this week</div>
            <div class="section-body">{opening_move_safe}</div>
        </div>
        """)

        render_html("<br>")

# ============ GI IN THE WILD TAB ============
with tab_wild:

    def render_card(item, is_featured=False):
        """Render a community mention card using components.html for guaranteed HTML rendering."""
        sentiment_colors_card = {"positive": "#00ff88", "neutral": "#888888", "skeptical": "#ff6b6b"}
        sentiment_icons_card = {"positive": "🟢", "neutral": "⚪", "skeptical": "🔴"}

        source = item["source"]
        source_color = "#ff6600" if source == "HN" else "#ff4500"
        source_label = "HN" if source == "HN" else f"r/{item.get('subreddit', 'Reddit')}"

        sentiment = item["sentiment"]
        sentiment_icon = sentiment_icons_card.get(sentiment, "⚪")
        sentiment_color_card = sentiment_colors_card.get(sentiment, "#888")

        upvotes = item.get("upvotes", 0)
        upvote_display = f"▲ {upvotes}" if upvotes > 0 else ""

        # Decode HTML entities first, then escape for display
        text_preview = html.escape(html.unescape(item["text"]))
        author = html.escape(html.unescape(item["author"]))
        date = item.get("date", "")
        url = html.escape(item.get("url", "#"))

        border_color = "#00ff88" if is_featured else "#2d3250"
        bg_color = "#1a2a1a" if is_featured else "#1e2130"
        featured_badge = '<span style="background: #00ff88; color: #000; padding: 2px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; margin-right: 8px;">⭐ FEATURED</span>' if is_featured else ""

        card_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: transparent; }}
            </style>
        </head>
        <body>
        <div style="background: {bg_color}; border-radius: 8px; padding: 16px; border: 2px solid {border_color};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div>
                    {featured_badge}
                    <span style="background: {source_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;">{source_label}</span>
                    <span style="color: {sentiment_color_card}; margin-left: 8px;">{sentiment_icon} {sentiment}</span>
                </div>
                <div style="color: #888; font-size: 12px;">
                    <span style="color: #00d4ff; font-weight: bold;">{upvote_display}</span>
                    <span style="margin-left: 10px;">{date}</span>
                </div>
            </div>
            <div style="color: #e0e0e0; font-size: 14px; line-height: 1.5; margin-bottom: 10px;">
                {text_preview}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #666; font-size: 12px;">by {author}</span>
                <a href="{url}" target="_blank" style="color: #00d4ff; font-size: 12px; text-decoration: none;">View conversation →</a>
            </div>
        </div>
        </body>
        </html>
        """
        card_height = 220 if is_featured else 180
        components.html(card_html, height=card_height, scrolling=False)

    render_html("<h3>GI in the Wild</h3>")
    render_html("<p style='color: #cccccc; font-style: italic;'>What is the AI research community saying about General Intuition when we're not tagged?</p>")

    # Refresh button
    col_refresh_wild, col_spacer_wild = st.columns([1, 4])
    with col_refresh_wild:
        if st.button("Refresh", key="refresh_wild"):
            if "community_mentions" in st.session_state:
                del st.session_state["community_mentions"]
            st.rerun()

    render_html("<hr style='border: none; border-top: 1px solid #333; margin: 20px 0;'>")

    # Load or retrieve cached mentions
    if "community_mentions" not in st.session_state:
        st.session_state["community_mentions"] = load_community_mentions()

    mentions_data = st.session_state["community_mentions"]
    featured = mentions_data["featured"]
    direct = mentions_data["direct"]
    adjacent = mentions_data["adjacent"]
    direct_count = mentions_data["direct_count"]
    adjacent_count = mentions_data["adjacent_count"]
    overall_sentiment = mentions_data["overall_sentiment"]
    limited_direct = mentions_data.get("limited_direct", False)
    limited_adjacent = mentions_data.get("limited_adjacent", False)

    # Sentiment colors
    sentiment_colors = {
        "positive": "#00ff88",
        "neutral": "#888888",
        "skeptical": "#ff6b6b",
    }

    # ======== SECTION A: DIRECT MENTIONS ========
    render_html("<h3>Direct Mentions</h3>")
    render_html("<p style='color: #cccccc; font-style: italic;'>Results that specifically mention GI, Pim de Witte, or Medal AI</p>")

    sentiment_color = sentiment_colors.get(overall_sentiment, "#888")

    if direct_count == 0:
        render_html(f"""
        <div style="background: #2a1a1a; border-left: 4px solid #ff6b6b; padding: 16px; border-radius: 4px; margin: 16px 0;">
            <strong style="color: #ff6b6b; font-size: 18px;">GI has 0 direct community mentions this month.</strong><br>
            <span style="color: #ccc; font-size: 14px;">This is the gap to close. The AI research community isn't talking about GI yet — every mention from here is growth.</span>
        </div>
        """)
    else:
        if limited_direct:
            render_html(f"""
            <div style="background: #2a2a1a; border-left: 4px solid #ffcc00; padding: 14px; border-radius: 4px; margin-bottom: 16px;">
                <strong style="color: #ffcc00;">Limited recent mentions found — this is the gap to close.</strong><br>
                <span style="color: #ccc;">{direct_count} mention{'s' if direct_count != 1 else ''} in the last 2 years. Sentiment: </span>
                <span style="color: {sentiment_color}; font-weight: bold;">{overall_sentiment}</span>
            </div>
            """)
        else:
            render_html(f"""
            <div style="background: #1e2130; border-left: 4px solid {sentiment_color}; padding: 14px; border-radius: 4px; margin-bottom: 16px;">
                <strong style="color: #00d4ff;">{direct_count} direct mention{'s' if direct_count != 1 else ''}</strong> found across HN and Reddit.<br>
                <span style="color: #ccc;">Overall sentiment: </span>
                <span style="color: {sentiment_color}; font-weight: bold;">{overall_sentiment}</span>
            </div>
            """)

        # Featured mentions first
        for item in featured:
            render_card(item, is_featured=True)

        # Other direct mentions
        for item in direct:
            render_card(item, is_featured=False)

    render_html("<hr style='border: none; border-top: 1px solid #333; margin: 20px 0;'>")

    # ======== SECTION B: CONVERSATIONS GI SHOULD BE IN ========
    render_html("<h3>Conversations GI Should Be In</h3>")
    render_html("<p style='color: #cccccc; font-style: italic;'>Adjacent discourse about gaming AI, world models, and spatial AI — where GI's voice is missing</p>")

    if adjacent_count == 0:
        st.info("No adjacent conversations found this month. Try refreshing or check back later.")
    else:
        if limited_adjacent:
            render_html(f"""
            <div style="background: #2a2a1a; border-left: 4px solid #ffcc00; padding: 14px; border-radius: 4px; margin-bottom: 16px;">
                <strong style="color: #ffcc00;">Limited recent conversations found.</strong><br>
                <span style="color: #ccc;">{adjacent_count} conversation{'s' if adjacent_count != 1 else ''} about gaming AI and world models in the last 2 years.</span>
            </div>
            """)
        else:
            render_html(f"""
            <div style="background: #1a1a2a; border-left: 4px solid #7c3aed; padding: 14px; border-radius: 4px; margin-bottom: 16px;">
                <strong style="color: #a78bfa;">{adjacent_count} conversation{'s' if adjacent_count != 1 else ''}</strong> about gaming AI, world models, and related topics.<br>
                <span style="color: #ccc;">These represent opportunities for GI to contribute technical expertise.</span>
            </div>
            """)

        for item in adjacent:
            render_card(item, is_featured=False)

    render_html("<hr style='border: none; border-top: 1px solid #333; margin: 20px 0;'>")
    render_html("""
    <div style="color: #666; font-size: 12px;">
        <strong>Direct search:</strong> "General Intuition" + AI, "Pim de Witte" + AI/Medal/startup, "Medal AI", gen_intuition<br>
        <strong>Adjacent search:</strong> "gaming AI" + "world model", "video game" + "training data", spatial AI gaming<br>
        <strong>Sources:</strong> HN (stories & comments), Reddit (r/MachineLearning, r/artificial, r/LocalLLaMA, r/singularity, r/ChatGPT, r/OpenAI, r/Futurology, r/technology, r/programming)<br>
        <strong>Relevance filter:</strong> Results must contain AI/ML/startup/model/gaming keywords
    </div>
    """)

# ============ WEEKLY BRIEF TAB ============
with tab_weekly:
    render_html("<h3>Weekly Intelligence Brief</h3>")
    render_html("<p style='color: #cccccc; font-style: italic;'>Your weekly research intelligence summary</p>")

    brief_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekly_brief.html")

    if os.path.exists(brief_path):
        # Show last modified timestamp
        mtime = os.path.getmtime(brief_path)
        last_modified = datetime.fromtimestamp(mtime).strftime("%B %d, %Y at %I:%M %p")
        render_html(f"<p style='color: #888; font-size: 13px;'>Last generated: {last_modified}</p>")

        # Action buttons side by side
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            with open(brief_path, "r") as f:
                brief_content = f.read()
            st.download_button(
                label="Download as PDF",
                data=brief_content,
                file_name="weekly_brief.html",
                mime="text/html",
                key="download_brief"
            )
        with col2:
            if st.button("Send to Slack Now", key="send_slack"):
                with st.spinner("Sending to Slack..."):
                    import subprocess
                    result = subprocess.run(
                        ["./venv/bin/python", "weekly_brief.py", "--send-slack"],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__))
                    )
                    if result.returncode == 0:
                        st.success("Sent to Slack!")
                    else:
                        st.error(f"Failed to send: {result.stderr}")

        render_html("<hr style='border: none; border-top: 1px solid #333; margin: 20px 0;'>")

        # Read and display the brief
        with open(brief_path, "r") as f:
            brief_html = f.read()
        components.html(brief_html, height=900, scrolling=True)
    else:
        render_html("<p style='color: #888;'>No weekly brief has been generated yet.</p>")

        if st.button("📄 Generate Brief Now", key="gen_brief"):
            with st.spinner("Generating weekly brief (this may take ~30 seconds)..."):
                import subprocess
                result = subprocess.run(
                    ["./venv/bin/python", "weekly_brief.py", "--now"],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(os.path.abspath(__file__))
                )
                if result.returncode == 0:
                    st.success("Weekly brief generated!")
                    st.rerun()
                else:
                    st.error(f"Generation failed: {result.stderr}")

render_html("---")
render_html("*Built by Giulia Piller — GI Social Analytics Case Study*")
