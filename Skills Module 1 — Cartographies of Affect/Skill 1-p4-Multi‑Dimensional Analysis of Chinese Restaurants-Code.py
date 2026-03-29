# -*- coding: utf-8 -*-
"""
Page 3: Analysis of Chinese Restaurants in London's Chinatown (Based on All Flickr Photos)
------------------------------------------------
1. Fetch all geotagged photos from London's Chinatown (no keyword restrictions)
2. Filter food-related photos via text keyword matching
3. Further differentiate into Chinese restaurants / other restaurants, generate layered point map (transparent background, independent layers)
4. Sub‑categorise Chinese restaurant photos and generate distribution map (transparent background, independent layers for each subcategory)
5. Generate heatmap + contour lines layer for Chinese restaurants (transparent background, independent layers)
"""

import requests
import time
import csv
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import colors
import json
from collections import Counter
from scipy.spatial import cKDTree

# ========== Configuration Parameters ==========
API_KEY = "Flickr_API_Key"   # Your Flickr API Key
CHINATOWN_BBOX = "-0.1350,51.5085,-0.1250,51.5140"   # London Chinatown bounding box

# Food‑related keywords (for local text classification)
FOOD_KEYWORDS = [
    "restaurant", "food", "dining", "cafe", "eat", "lunch", "dinner", "menu",
    "dish", "cuisine", "meal", "snack", "takeaway", "bistro", "bar", "pub",
    "coffee", "tea", "cake", "pastry", "pizza", "burger", "sushi", "noodle",
    "rice", "soup", "steak", "seafood", "grill", "bbq", "sandwich", "salad",
    "dessert", "ice cream", "chocolate", "bakery", "breakfast", "brunch"
]

# Chinese restaurant keywords
CHINESE_KEYWORDS = [
    "chinese", "china", "chinatown", "dumpling", "dim sum", "wonton", "spring roll",
    "sichuan", "cantonese", "hot pot", "火锅", "麻辣烫", "川菜", "粤菜", "点心",
    "炒饭", "炒面", "烧卖", "虾饺", "肠粉", "烤鸭", "叉烧", "烧鹅", "蒸鱼",
    "麻婆豆腐", "宫保鸡丁", "水煮鱼", "酸菜鱼", "担担面", "拉面", "小笼包",
    "生煎包", "煎饼", "煎饺", "蒸饺", "云吞", "炒河", "煲仔饭", "港式", "粤式",
    "川味", "湘菜", "东北菜", "台湾", "奶茶", "珍珠奶茶", "bubble tea", "milk tea",
    "boba", "chinese tea"
]

# Subcategory keywords (for finer classification of Chinese restaurant photos)
SUBCATEGORY_KEYWORDS = {
    "Drinks": ["奶茶", "bubble tea", "milk tea", "boba", "茶", "tea", "咖啡", "coffee"],
    "Desserts": ["dessert", "cake", "pastry", "sweet", "冰", "冰淇淋", "ice cream", "布丁", "pudding"],
    "Hot Pot": ["hot pot", "火锅", "shabu", "steamboat"],
    "Noodle Houses": ["noodle", "面", "拉面", "乌冬", "ramen"],
    "Sichuan Cuisine": ["sichuan", "川", "麻辣", "spicy", "麻婆", "宫保", "水煮"],
    "Cantonese Cuisine": ["cantonese", "粤", "dim sum", "点心", "烧卖", "虾饺", "肠粉", "烤鸭"],
    "Malatang": ["麻辣烫", "malatang"],
    "Others": []   # default
}

# Subcategory colors (avoid using grey)
CATEGORY_COLORS = {
    "Drinks": "lightblue",
    "Desserts": "pink",
    "Hot Pot": "red",
    "Noodle Houses": "orange",
    "Sichuan Cuisine": "darkred",
    "Cantonese Cuisine": "green",
    "Malatang": "purple",
    "Others": "lightgray"  # light grey, distinguished from the dark grey used in the point map
}

# Number of photos per page
PER_PAGE = 150
REQUEST_DELAY = 0.5
MAX_PHOTOS_TOTAL = 15000   # Keep consistent with Page 2

# Output files
ALL_PHOTOS_CSV = "chinatown_all_photos.csv"
FOOD_PHOTOS_CSV = "chinatown_food_photos.csv"
DOTS_MAP = "chinatown_food_dots.html"
HEATMAP_MAP = "chinatown_chinese_heatmap.html"
CATEGORY_MAP = "chinatown_chinese_category.html"

# ========== 1. Fetch all photos from Flickr (no keywords) ==========
def search_all_photos(max_photos):
    """Search for all public geotagged photos within the specified bounding box, return up to max_photos photos"""
    url = "https://api.flickr.com/services/rest/"
    photos = []
    page = 1
    photo_ids = set()
    no_new_count = 0   # number of consecutive pages with no new photos

    while len(photos) < max_photos:
        params = {
            "method": "flickr.photos.search",
            "api_key": API_KEY,
            "bbox": CHINATOWN_BBOX,
            "format": "json",
            "nojsoncallback": 1,
            "extras": "geo,tags,description,url_m",
            "per_page": min(PER_PAGE, max_photos - len(photos)),
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

        new_photos_count = 0
        for p in photo_list:
            pid = p.get("id")
            if not pid or pid in photo_ids:
                continue
            try:
                lat = float(p.get("latitude", 0))
                lon = float(p.get("longitude", 0))
            except (ValueError, TypeError):
                continue
            if lat == 0 or lon == 0:
                continue

            photo_ids.add(pid)
            photos.append(p)
            new_photos_count += 1
            if len(photos) >= max_photos:
                break

        if new_photos_count == 0:
            no_new_count += 1
            if no_new_count >= 3:
                print(f"No new photos for {no_new_count} consecutive pages, stopping pagination.")
                break
        else:
            no_new_count = 0  # reset counter

        print(f"Page {page}: fetched {len(photo_list)} photos, added {new_photos_count} new photos, total {len(photos)} photos")
        page += 1
        time.sleep(REQUEST_DELAY)

    return photos

def save_photos_to_csv(photos, filename):
    """Save photo metadata to CSV"""
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "title", "owner", "lat", "lon", "url", "tags", "description"])
        for p in photos:
            tags = p.get("tags", "")
            desc = p.get("description", {}).get("_content", "")
            writer.writerow([
                p["id"],
                p.get("title", ""),
                p.get("owner", ""),
                float(p["latitude"]),
                float(p["longitude"]),
                p.get("url_m", ""),
                tags,
                desc
            ])
    print(f"Saved {len(photos)} photos to {filename}")

# ========== 2. Read from CSV and classify ==========
def load_and_classify(csv_file):
    """Read CSV, add is_food, is_chinese, subcategory columns"""
    df = pd.read_csv(csv_file)

    # Combine text
    df["combined_text"] = df["title"].fillna('') + " " + df["tags"].fillna('') + " " + df["description"].fillna('')
    df["combined_text"] = df["combined_text"].str.lower()

    # Check if food‑related
    def is_food(text):
        for kw in FOOD_KEYWORDS:
            if kw.lower() in text:
                return True
        return False

    # Check if Chinese restaurant related
    def is_chinese(text):
        for kw in CHINESE_KEYWORDS:
            if kw.lower() in text:
                return True
        return False

    # Classify subcategory for Chinese restaurant photos
    def classify_subcategory(text):
        for cat, keywords in SUBCATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text:
                    return cat
        return "Others"

    df["is_food"] = df["combined_text"].apply(is_food)
    df["is_chinese"] = df["combined_text"].apply(is_chinese)

    # Keep only food‑related photos
    df_food = df[df["is_food"]].copy()
    print(f"\nTotal original photos: {len(df)}")
    print(f"Food‑related photos: {len(df_food)}")

    # Further sub‑categorise Chinese restaurant photos among food photos
    df_food["subcategory"] = "Others"
    df_food.loc[df_food["is_chinese"], "subcategory"] = df_food.loc[df_food["is_chinese"], "combined_text"].apply(classify_subcategory)

    print("\nChinese restaurant photo statistics:")
    chinese_count = df_food["is_chinese"].sum()
    print(f"Chinese‑related photos: {chinese_count}")
    print(f"Other restaurant photos: {len(df_food) - chinese_count}")

    if chinese_count > 0:
        print("\nDistribution of Chinese restaurant subcategories:")
        print(df_food[df_food["is_chinese"]]["subcategory"].value_counts())

    return df_food


# ========== Helper function: create map with transparent background ==========
def create_transparent_base_map(center_lat, center_lon, zoom_start=16):
    """Create a map with no base tile, transparent background, and add a base layer that is off by default"""
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=zoom_start,
                   tiles=None,
                   control_scale=True)

    # Add CSS to make map container transparent
    transparent_css = """
    <style>
        .leaflet-container {
            background-color: transparent !important;
        }
    </style>
    """
    m.get_root().html.add_child(folium.Element(transparent_css))

    # Add an optional base map layer (off by default)
    folium.TileLayer('CartoDB positron', name="Street Map", show=False).add_to(m)

    return m


# ========== 3. Generate point distribution map (layered, transparent background) ==========
def create_dots_map(df):
    """
    Generate layered point distribution map:
    - Other restaurants layer (grey)
    - Chinese restaurants layer (red)
    - Base map layer (off by default)
    All layers independently toggleable, background transparent.
    """
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()

    m = create_transparent_base_map(center_lat, center_lon)

    # Other restaurant points (grey)
    other_group = folium.FeatureGroup(name="Other Restaurants", show=True)
    other_df = df[~df["is_chinese"]]
    for _, row in other_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color="gray",
            fill=True,
            fill_color="gray",
            fill_opacity=0.6,
            popup=f"{row['title']}<br>Tags: {row['tags']}"
        ).add_to(other_group)

    # Chinese restaurant points (red)
    chinese_group = folium.FeatureGroup(name="Chinese Restaurants", show=True)
    chinese_df = df[df["is_chinese"]]
    for _, row in chinese_df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=f"{row['title']}<br>Tags: {row['tags']}"
        ).add_to(chinese_group)

    other_group.add_to(m)
    chinese_group.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ========== 4. Generate heatmap + contour lines layer ==========
def generate_contour_geojson(df, grid_size=150, num_levels=10, smooth_sigma=1.5):
    """Generate density grid from point coordinates, extract contour lines, return GeoJSON"""
    lats = df["lat"].values
    lons = df["lon"].values

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    lon_edges = np.linspace(lon_min - lon_pad, lon_max + lon_pad, grid_size + 1)
    lat_edges = np.linspace(lat_min - lat_pad, lat_max + lat_pad, grid_size + 1)

    H, lon_edges, lat_edges = np.histogram2d(lons, lats, bins=[lon_edges, lat_edges])
    H = H.T
    H_smooth = gaussian_filter(H, sigma=smooth_sigma)

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    X, Y = np.meshgrid(lon_centers, lat_centers)

    # Determine contour levels
    levels = np.percentile(H_smooth[H_smooth > 0], np.linspace(10, 90, num_levels))
    levels = np.unique(levels)
    if len(levels) < 2:
        levels = np.linspace(H_smooth.min(), H_smooth.max(), num_levels+2)[1:-1]

    fig = plt.figure()
    contour_set = plt.contour(X, Y, H_smooth, levels=levels, linewidths=1)
    plt.close(fig)

    cmap = plt.cm.YlOrRd
    norm = colors.Normalize(vmin=levels.min(), vmax=levels.max())
    features = []

    for level_idx, segs in enumerate(contour_set.allsegs):
        level_value = contour_set.levels[level_idx]
        color = cmap(norm(level_value))
        rgba = f"rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},{color[3]})"

        for segment in segs:
            if len(segment) < 2:
                continue
            line_coords = [[p[0], p[1]] for p in segment.tolist()]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": line_coords
                },
                "properties": {
                    "level": float(level_value),
                    "stroke": rgba,
                    "stroke-width": 1.5,
                    "stroke-opacity": 0.8
                }
            })

    return {"type": "FeatureCollection", "features": features}

def create_heatmap_contour_map(df_chinese):
    """Generate heatmap and contour lines based on Chinese restaurant photos (transparent background)"""
    center_lat = df_chinese["lat"].mean()
    center_lon = df_chinese["lon"].mean()

    m = create_transparent_base_map(center_lat, center_lon)

    # Heatmap layer
    heat_group = folium.FeatureGroup(name="Heatmap", show=True)
    heat_data = [[row["lat"], row["lon"], 1] for _, row in df_chinese.iterrows()]
    HeatMap(heat_data, radius=12, blur=8, max_zoom=18).add_to(heat_group)

    # Contour lines layer
    contour_geojson = generate_contour_geojson(df_chinese)
    contour_group = folium.FeatureGroup(name="Contour Lines", show=True)

    def style_function(feature):
        props = feature.get("properties", {})
        return {
            "color": props.get("stroke", "#ff0000"),
            "weight": props.get("stroke-width", 1.5),
            "opacity": props.get("stroke-opacity", 0.8)
        }

    folium.GeoJson(contour_geojson, name="Contour Lines", style_function=style_function).add_to(contour_group)

    heat_group.add_to(m)
    contour_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ========== 5. Generate subcategory distribution map for Chinese restaurants (transparent background, avoid grey) ==========
def create_category_map(df_chinese):
    """Display Chinese restaurant photo points coloured by subcategory, each subcategory as an independent layer, background transparent"""
    center_lat = df_chinese["lat"].mean()
    center_lon = df_chinese["lon"].mean()

    m = create_transparent_base_map(center_lat, center_lon)

    for cat, color in CATEGORY_COLORS.items():
        group = folium.FeatureGroup(name=cat, show=True)
        cat_df = df_chinese[df_chinese["subcategory"] == cat]
        for _, row in cat_df.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{row['title']}<br>Category: {cat}"
            ).add_to(group)
        group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ========== Main Program ==========
def main():
    print("=" * 60)
    print("Page 3: Analysis of Chinese Restaurants in London's Chinatown (Based on All Flickr Photos)")
    print("=" * 60)

    # 1. Fetch all Chinatown photos (skip if already exists)
    try:
        df_all = pd.read_csv(ALL_PHOTOS_CSV)
        print(f"Loaded {len(df_all)} photos from {ALL_PHOTOS_CSV}")
    except FileNotFoundError:
        print("Start fetching all geotagged photos from Flickr (no keywords)...")
        photos = search_all_photos(MAX_PHOTOS_TOTAL)
        if not photos:
            print("No photos fetched, please check API Key or network.")
            return
        save_photos_to_csv(photos, ALL_PHOTOS_CSV)
        df_all = pd.read_csv(ALL_PHOTOS_CSV)

    # 2. Classify, filter food‑related photos
    df_food = load_and_classify(ALL_PHOTOS_CSV)

    if len(df_food) == 0:
        print("No food‑related photos found, analysis terminated.")
        return

    # Save food photos CSV
    df_food.to_csv(FOOD_PHOTOS_CSV, index=False, encoding='utf-8')
    print(f"Food photos saved to {FOOD_PHOTOS_CSV}")

    # 3. Generate point distribution map
    print("\nGenerating point distribution map (Other Restaurants / Chinese Restaurants)...")
    dots_map = create_dots_map(df_food)
    dots_map.save(DOTS_MAP)
    print(f"Saved: {DOTS_MAP}")

    chinese_df = df_food[df_food["is_chinese"]]
    if len(chinese_df) > 0:
        # 4. Generate heatmap + contour lines
        print("\nGenerating heatmap + contour lines for Chinese restaurants...")
        heat_map = create_heatmap_contour_map(chinese_df)
        heat_map.save(HEATMAP_MAP)
        print(f"Saved: {HEATMAP_MAP}")

        # 5. Generate subcategory distribution map
        print("\nGenerating subcategory distribution map for Chinese restaurants...")
        cat_map = create_category_map(chinese_df)
        cat_map.save(CATEGORY_MAP)
        print(f"Saved: {CATEGORY_MAP}")
    else:
        print("No Chinese restaurant photos found, skipping subsequent analyses.")

    print("\nAll maps generated!")


if __name__ == "__main__":
    main()
