# -*- coding: utf-8 -*-
"""
Page 4: Sentiment Map of Han Culture in London's Chinatown (Based on Flickr)
------------------------------------------------
1. Search for photos containing keywords related to "Han culture" within the London Chinatown bounding box, max 900 per keyword, target at least 3000 total.
2. Perform sentiment analysis (TextBlob) on the textual information (title, description, tags) of each photo to obtain a sentiment polarity score (-1 to 1).
3. Divide into two categories: positive (>0), negative (<=0).
4. Save results as CSV: latitude, longitude, sentiment score, title, author, image link.
5. Output sentiment statistics: highest score, lowest score, average score.
6. Generate interactive HTML map: positive (green), negative (red), with clickable pop‑ups.
"""

import requests
import time
import csv
import pandas as pd
import folium
from textblob import TextBlob
import os
import numpy as np

# ========== Configuration Parameters ==========
API_KEY = "Flickr_API_Key"   # Your Flickr API Key
CHINATOWN_BBOX = "-0.1350,51.5085,-0.1250,51.5140"   # London Chinatown bounding box

# "Han culture" related keywords (mixed Chinese and English)
KEYWORDS = [
    "chinese culture", "traditional chinese", "calligraphy",
    "chinese painting", "chinese new year", "mid-autumn festival",
    "dragon dance", "lion dance", "kung fu", "tai chi", "chinese tea",
    "porcelain", "silk", "chinese opera", "chinese music",
    "春节", "中秋节", "舞狮"
]

# Number of photos per page
PER_PAGE = 200
REQUEST_DELAY = 0.5      # Request interval (seconds)
MAX_PER_KEYWORD = 2000    # Maximum number of photos per keyword
TARGET_TOTAL = 3000      # Target total number of photos (at least)

# Output files
CSV_OUTPUT = "chinatown_han_culture.csv"
MAP_OUTPUT = "chinatown_han_culture_map.html"

# ========== 1. Search for all relevant photos (max 900 per keyword, at least 3000 total) ==========
def search_photos_by_keywords(keywords, bbox, target_total, max_per_keyword, per_page=100):
    """
    Search for each keyword, up to max_per_keyword photos per keyword, until target_total is reached or keywords are exhausted.
    Returns a deduplicated list of photos.
    """
    url = "https://api.flickr.com/services/rest/"
    all_photos = {}
    total_collected = 0

    for kw in keywords:
        if total_collected >= target_total:
            break

        print(f"Searching keyword: {kw}")
        kw_page = 1
        kw_collected = 0
        while kw_collected < max_per_keyword and total_collected < target_total:
            params = {
                "method": "flickr.photos.search",
                "api_key": API_KEY,
                "text": kw,
                "bbox": bbox,
                "format": "json",
                "nojsoncallback": 1,
                "extras": "geo,tags,description,url_m,owner_name",
                "per_page": per_page,
                "page": kw_page,
                "content_type": 1,
                "safe_search": 1,
                "has_geo": 1
            }

            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  Request error (keyword={kw}, page={kw_page}): {e}")
                break

            if data.get("stat") != "ok":
                print(f"  API return error: {data.get('message', 'unknown error')}")
                break

            photo_list = data.get("photos", {}).get("photo", [])
            if not photo_list:
                print("  No more photos, stopping current keyword.")
                break

            new_count = 0
            for p in photo_list:
                pid = p.get("id")
                if not pid:
                    continue
                # Ensure valid latitude/longitude
                try:
                    lat = float(p.get("latitude", 0))
                    lon = float(p.get("longitude", 0))
                except (ValueError, TypeError):
                    continue
                if lat == 0 or lon == 0:
                    continue

                if pid not in all_photos:
                    all_photos[pid] = {
                        "id": pid,
                        "title": p.get("title", ""),
                        "owner": p.get("ownername", ""),
                        "lat": lat,
                        "lon": lon,
                        "url_m": p.get("url_m", ""),
                        "tags": p.get("tags", ""),
                        "description": p.get("description", {}).get("_content", "")
                    }
                    new_count += 1
                    kw_collected += 1
                    total_collected += 1
                    if kw_collected >= max_per_keyword or total_collected >= target_total:
                        break

            print(f"  Keyword '{kw}' page {kw_page}: fetched {len(photo_list)} photos, added {new_count} new, cumulative for this keyword: {kw_collected}, total cumulative: {total_collected}")
            kw_page += 1
            time.sleep(REQUEST_DELAY)

            if new_count == 0 and kw_page > 5:   # No new photos for several pages, can stop this keyword early
                print("  No new photos expected for this keyword, stopping early.")
                break

        time.sleep(REQUEST_DELAY)  # Slight delay between keywords

    return list(all_photos.values())

# ========== 2. Sentiment Analysis ==========
def analyze_sentiment(photo):
    """Combine title, tags, description and compute sentiment polarity using TextBlob"""
    combined = f"{photo['title']} {photo['tags']} {photo['description']}"
    # Simple cleaning: remove extra spaces
    combined = " ".join(combined.split())
    if not combined.strip():
        return 0.0
    blob = TextBlob(combined)
    return blob.sentiment.polarity

# ========== 3. Save CSV ==========
def save_to_csv(photos, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "owner", "lat", "lon", "url_m", "tags", "description", "sentiment"])
        for p in photos:
            writer.writerow([
                p["id"],
                p["title"],
                p["owner"],
                p["lat"],
                p["lon"],
                p["url_m"],
                p["tags"],
                p["description"],
                p["sentiment"]
            ])
    print(f"Saved {len(photos)} records to {filename}")

# ========== 4. Generate interactive map ==========
def create_sentiment_map(photos):
    # Compute center point
    lats = [p["lat"] for p in photos]
    lons = [p["lon"] for p in photos]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)

    # Create map (using OpenStreetMap base map)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

    # Add points by sentiment
    for p in photos:
        sentiment = p["sentiment"]
        # positive (>0) green, negative (<=0) red
        if sentiment > 0:
            color = "green"
        else:
            color = "red"

        # Pop‑up information
        popup_text = f"""
        <b>Title:</b> {p['title']}<br>
        <b>Author:</b> {p['owner']}<br>
        <b>Sentiment Score:</b> {sentiment:.3f}<br>
        <a href="{p['url_m']}" target="_blank">View Image</a>
        """
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(m)

    return m

# ========== Main Program ==========
def main():
    print("=" * 60)
    print("Page 4: Sentiment Map of Han Culture in London's Chinatown")
    print("=" * 60)

    # Check if a CSV already exists; if yes, optionally load it to avoid re‑fetching
    if os.path.exists(CSV_OUTPUT):
        print(f"Existing data file {CSV_OUTPUT} found. Re‑fetch? (y/n)")
        choice = input().strip().lower()
        if choice == 'n':
            df = pd.read_csv(CSV_OUTPUT)
            photos = df.to_dict('records')
            print(f"Loaded {len(photos)} records")
        else:
            photos = search_photos_by_keywords(KEYWORDS, CHINATOWN_BBOX, TARGET_TOTAL, MAX_PER_KEYWORD, PER_PAGE)
            if not photos:
                print("No photos obtained, program terminated.")
                return
            print(f"Total collected {len(photos)} photos, performing sentiment analysis...")
            for p in photos:
                p["sentiment"] = analyze_sentiment(p)
            save_to_csv(photos, CSV_OUTPUT)
    else:
        photos = search_photos_by_keywords(KEYWORDS, CHINATOWN_BBOX, TARGET_TOTAL, MAX_PER_KEYWORD, PER_PAGE)
        if not photos:
            print("No photos obtained, program terminated.")
            return
        print(f"Total collected {len(photos)} photos, performing sentiment analysis...")
        for p in photos:
            p["sentiment"] = analyze_sentiment(p)
        save_to_csv(photos, CSV_OUTPUT)

    # Sentiment distribution statistics (two classes: positive >0, negative <=0)
    sentiments = [p["sentiment"] for p in photos]
    pos = sum(1 for s in sentiments if s > 0)
    neg = sum(1 for s in sentiments if s <= 0)   # 0 is considered negative
    print(f"\nSentiment distribution: positive {pos}, negative {neg}")

    # Sentiment score statistics
    max_sent = max(sentiments)
    min_sent = min(sentiments)
    avg_sent = np.mean(sentiments)
    print(f"Sentiment score statistics: highest {max_sent:.3f}, lowest {min_sent:.3f}, average {avg_sent:.3f}")

    # Generate map
    print("Generating sentiment map...")
    m = create_sentiment_map(photos)
    m.save(MAP_OUTPUT)
    print(f"Map saved to {MAP_OUTPUT}")

if __name__ == "__main__":
    main()
