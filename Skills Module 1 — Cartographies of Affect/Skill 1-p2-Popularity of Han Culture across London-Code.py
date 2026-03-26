# -*- coding: utf-8 -*-
"""
Flickr Han Culture Popularity Analysis (City of London)
----------------------------------
By searching for keywords related to "Han culture" on Flickr, collect all geotagged photos
and generate a heatmap showing the density of photos across London (i.e., the distribution
of "Han culture" popularity).
Output files:
- london_han_popularity.csv: contains latitude, longitude and metadata for each photo
- london_han_popularity_map.html: interactive heatmap
"""

import requests
import time
import csv
import folium
from folium.plugins import HeatMap

# ========== Configuration Parameters ==========
API_KEY = "7e3a26760a42c3a6bc9b66b327df0fc6"  # Replace with your valid Flickr API Key

# London bounding box (format: min_lon, min_lat, max_lon, max_lat)
LONDON_BBOX = "-0.510,51.280,0.334,51.686"  # Covers Greater London area

# Keywords representing "Han culture" (mix of Chinese and English, can be adjusted)
SEARCH_KEYWORDS = [
    "chinese culture", "han chinese", "chinese new year", "chinatown",
    "chinese festival", "chinese art", "chinese calligraphy", "chinese food",
    "chinese people", "chinese diaspora", "chinese tradition"
]

# Maximum number of photos to fetch per keyword (to avoid API overload)
MAX_PHOTOS_PER_KEYWORD = 5000
REQUEST_DELAY = 1  # Request interval (seconds) to avoid rate limiting

# Output files
CSV_OUTPUT = "london_han_popularity.csv"
HTML_OUTPUT = "london_han_popularity_map.html"

# ========== Flickr API Search Function ==========
def search_flickr(keyword):
    """
    Search for geotagged photos within the London bounding box based on a keyword
    Returns a list of photos (each as a dict containing id, title, lat, lon, url_m, etc.)
    """
    url = "https://api.flickr.com/services/rest/"
    photos = []
    page = 1

    while len(photos) < MAX_PHOTOS_PER_KEYWORD:
        params = {
            "method": "flickr.photos.search",
            "api_key": API_KEY,
            "bbox": LONDON_BBOX,
            "text": keyword,
            "format": "json",
            "nojsoncallback": 1,
            "extras": "geo,tags,description,url_m",
            "per_page": 100,          # Flickr allows up to 250 per page, 100 is safer
            "page": page,
            "content_type": 1,         # Photos only
            "safe_search": 1,           # Safe mode
            "has_geo": 1                # Must have geographic coordinates
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Request error (keyword={keyword}, page={page}): {e}")
            break

        # Check API return status
        if data.get("stat") != "ok":
            print(f"  API return error (keyword={keyword}): {data.get('message', 'unknown error')}")
            break

        photo_list = data.get("photos", {}).get("photo", [])
        if not photo_list:
            break  # No more photos

        for p in photo_list:
            # Ensure valid latitude/longitude
            try:
                lat = float(p.get("latitude", 0))
                lon = float(p.get("longitude", 0))
            except (ValueError, TypeError):
                continue
            if lat == 0 and lon == 0:
                continue

            photos.append(p)
            if len(photos) >= MAX_PHOTOS_PER_KEYWORD:
                break

        page += 1
        time.sleep(REQUEST_DELAY)   # Polite delay

    print(f"  Keyword '{keyword}' collected {len(photos)} photos")
    return photos

# ========== Main Program ==========
def main():
    if not API_KEY or API_KEY == "YOUR_FLICKR_API_KEY":
        raise ValueError("Please fill in a valid Flickr API_KEY in the code")

    all_photos = {}          # Use dict to deduplicate by photo id

    print("Start fetching Flickr photos...")
    for kw in SEARCH_KEYWORDS:
        print(f"Processing keyword: {kw}")
        results = search_flickr(kw)
        for p in results:
            pid = p.get("id")
            if not pid:
                continue
            if pid not in all_photos:
                # Extract required fields
                all_photos[pid] = {
                    "id": pid,
                    "title": p.get("title", ""),
                    "owner": p.get("owner", ""),
                    "lat": float(p.get("latitude", 0)),
                    "lon": float(p.get("longitude", 0)),
                    "url": p.get("url_m", ""),
                    "tags": p.get("tags", ""),
                    "description": p.get("description", {}).get("_content", "")
                }

    print(f"\nTotal collected {len(all_photos)} unique photos (after deduplication)")

    if not all_photos:
        print("No photos collected, please check API Key or keywords.")
        return

    # ===== Save to CSV =====
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Latitude", "Longitude", "Title", "Owner", "URL", "Tags", "Description"])
        for p in all_photos.values():
            writer.writerow([
                p["lat"],
                p["lon"],
                p["title"],
                p["owner"],
                p["url"],
                p["tags"],
                p["description"]
            ])
    print(f"CSV saved to {CSV_OUTPUT}")

    # ===== Generate heatmap =====
    # Prepare heatmap data: format [[lat, lon, intensity], ...] here intensity is set to 1, meaning equal contribution from each point
    heat_data = [[p["lat"], p["lon"], 1] for p in all_photos.values()]

    # Create Folium map with central London as initial view
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, 
                   tiles="CartoDB dark_matter")   # Dark basemap to highlight heat

    # Add heat layer
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

    # Save map
    m.save(HTML_OUTPUT)
    print(f"Heatmap saved to {HTML_OUTPUT}")

if __name__ == "__main__":
    main()