# -*- coding: utf-8 -*-
"""
Page 2: Pedestrian Flow Analysis in Chinatown (Based on Flickr Photo Density)
------------------------------------------------
Search for all geotagged photos within the Chinatown bounding box (no keyword filtering),
generate a heatmap (density) and a dot distribution map (all photo locations),
to visually display pedestrian activity intensity.
Output files:
- chinatown_activity.csv: latitude, longitude and metadata of all photos
- chinatown_activity_heatmap.html: interactive heatmap
- chinatown_activity_dots.html: interactive dot distribution map
"""

import requests
import time
import csv
import folium
from folium.plugins import HeatMap

# ========== Configuration Parameters ==========
API_KEY = "7e3a26760a42c3a6bc9b66b327df0fc6"  # Your Flickr API Key

# London Chinatown bounding box (adjustable as needed)
CHINATOWN_BBOX = "-0.1350,51.5085,-0.1250,51.5140"

# Number of photos per page (set to 150 to balance speed and data volume)
PER_PAGE = 150
REQUEST_DELAY = 0.5  # Request interval (seconds)

# Maximum total number of photos (to prevent infinite fetching)
MAX_PHOTOS_TOTAL = 15000

# Output files
CSV_OUTPUT = "chinatown_activity.csv"
HEATMAP_OUTPUT = "chinatown_activity_heatmap.html"
DOTMAP_OUTPUT = "chinatown_activity_dots.html"

# ========== Flickr API Search Function (no keywords, with limit) ==========
def search_all_photos(max_photos):
    """
    Search for all public geotagged photos within the specified bounding box,
    return up to max_photos photos.
    Returns a list of photos.
    """
    url = "https://api.flickr.com/services/rest/"
    photos = []
    page = 1

    while len(photos) < max_photos:
        params = {
            "method": "flickr.photos.search",
            "api_key": API_KEY,
            "bbox": CHINATOWN_BBOX,
            "format": "json",
            "nojsoncallback": 1,
            "extras": "geo,tags,description,url_m",
            "per_page": min(PER_PAGE, max_photos - len(photos)),  # Last page may be less than PER_PAGE
            "page": page,
            "content_type": 1,
            "safe_search": 1,
            "has_geo": 1
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Request error (page={page}): {e}")
            break

        if data.get("stat") != "ok":
            print(f"API return error: {data.get('message', 'unknown error')}")
            break

        photo_list = data.get("photos", {}).get("photo", [])
        if not photo_list:
            print("No more photos, stopping pagination.")
            break

        for p in photo_list:
            try:
                lat = float(p.get("latitude", 0))
                lon = float(p.get("longitude", 0))
            except (ValueError, TypeError):
                continue
            if lat == 0 and lon == 0:
                continue

            photos.append(p)
            if len(photos) >= max_photos:
                break

        print(f"Page {page}, fetched {len(photo_list)} photos, total {len(photos)} photos")
        page += 1
        time.sleep(REQUEST_DELAY)

    return photos

# ========== Main Program ==========
def main():
    if not API_KEY:
        raise ValueError("Please fill in a valid Flickr API_KEY")

    print(f"Start fetching geotagged photos in London Chinatown area (limit {MAX_PHOTOS_TOTAL} photos)...")
    results = search_all_photos(MAX_PHOTOS_TOTAL)

    if not results:
        print("No photos collected. Please check the bounding box or API Key.")
        return

    print(f"\nTotal collected {len(results)} photos (before deduplication)")

    # Deduplicate by photo ID
    unique_photos = {}
    for p in results:
        pid = p.get("id")
        if pid and pid not in unique_photos:
            unique_photos[pid] = {
                "id": pid,
                "title": p.get("title", ""),
                "owner": p.get("owner", ""),
                "lat": float(p.get("latitude", 0)),
                "lon": float(p.get("longitude", 0)),
                "url": p.get("url_m", ""),
                "tags": p.get("tags", ""),
                "description": p.get("description", {}).get("_content", "")
            }

    print(f"{len(unique_photos)} photos remaining after deduplication")

    # ===== Save CSV =====
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Latitude", "Longitude", "Title", "Owner", "URL", "Tags", "Description"])
        for p in unique_photos.values():
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
    heat_data = [[p["lat"], p["lon"], 1] for p in unique_photos.values()]
    m_heat = folium.Map(location=[51.5117, -0.1304], zoom_start=16,
                        tiles="CartoDB dark_matter")
    HeatMap(heat_data, radius=12, blur=8, max_zoom=18).add_to(m_heat)
    m_heat.save(HEATMAP_OUTPUT)
    print(f"Heatmap saved to {HEATMAP_OUTPUT}")

    # ===== Generate dot distribution map =====
    m_dots = folium.Map(location=[51.5117, -0.1304], zoom_start=16,
                        tiles="CartoDB dark_matter")
    for p in unique_photos.values():
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=2,                # Small dots to avoid over‑occlusion
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"Title: {p['title']}<br>Owner: {p['owner']}"
        ).add_to(m_dots)
    m_dots.save(DOTMAP_OUTPUT)
    print(f"Dot distribution map saved to {DOTMAP_OUTPUT}")


if __name__ == "__main__":
    main()