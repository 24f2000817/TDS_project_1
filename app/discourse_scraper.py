import requests
import json

def scrape_and_save():
    base_url = "https://discourse.onlinedegree.iitm.ac.in"
    topic_list = requests.get(base_url + "/latest.json").json()

    posts = []
    for topic in topic_list['topic_list']['topics']:
        tid = topic["id"]
        detail = requests.get(f"{base_url}/t/{tid}.json").json()
        posts.append({
            "id": tid,
            "title": topic["title"],
            "content": "\n".join([p["cooked"] for p in detail["post_stream"]["posts"]]),
            "url": f"{base_url}/t/{tid}"
        })

    with open("app/data/discourse_posts.jsonl", "w") as f:
        for post in posts:
            f.write(json.dumps(post) + "\n")
