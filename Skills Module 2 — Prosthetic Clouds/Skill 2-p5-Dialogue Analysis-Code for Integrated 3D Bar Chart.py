import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from collections import Counter

# ==================== 1. Read data ====================
csv_file = 'sentiment_analysis_results (1).csv'
df = pd.read_csv(csv_file, encoding='utf-8')

# ==================== 2. Helper function: convert time to seconds ====================
def time_to_seconds(t):
    parts = list(map(int, str(t).split(':')))
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    else:
        return 0

df['start_sec'] = df['start_time'].apply(time_to_seconds)
df['end_sec'] = df['end_time'].apply(time_to_seconds)
df['duration'] = df['end_sec'] - df['start_sec']
df['duration'] = df['duration'].clip(lower=0)  # prevent negative values

# ==================== 3. Word count ====================
df['word_count'] = df['text'].fillna('').apply(lambda x: len(str(x).split()))

# ==================== 4. Keyword extraction (Top 15) ====================
stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
    'a', 'an', 'and', 'but', 'or', 'for', 'nor', 'on', 'in', 'at', 'to', 'of', 'with',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may',
    'might', 'must', 'this', 'that', 'these', 'those', 'the', 'and', 'so', 'just', 'like',
    'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how'
}
all_words = []
for text in df['text'].dropna():
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    all_words.extend([w for w in words if w not in stop_words and len(w) > 1])
word_freq = Counter(all_words)
top15 = word_freq.most_common(15)
keywords = [w for w, _ in top15]
keyword_counts = [c for _, c in top15]

# ==================== 5. Define intervals ====================
# Time intervals (every 5 minutes, i.e., 300 seconds)
max_time = df['start_sec'].max()
time_bins = range(0, int(max_time) + 300, 300)
time_labels = [f"{int(b//60)}-{int((b+300)//60)}" for b in time_bins[:-1]]
time_counts = []
for i in range(len(time_bins)-1):
    cnt = ((df['start_sec'] >= time_bins[i]) & (df['start_sec'] < time_bins[i+1])).sum()
    time_counts.append(cnt)

# Length intervals (custom)
word_bins = [0, 1, 3, 5, 7, 10, 15, 20, 30, 50]
word_labels = [f"{word_bins[i]}-{word_bins[i+1]}" for i in range(len(word_bins)-1)]
word_counts = []
for i in range(len(word_bins)-1):
    cnt = ((df['word_count'] >= word_bins[i]) & (df['word_count'] < word_bins[i+1])).sum()
    word_counts.append(cnt)

# Duration intervals (1‑second steps, but merge >10 seconds into "10+")
max_dur = df['duration'].max()
dur_bins = list(range(0, int(min(max_dur, 10)) + 1)) + [100]  # 0-1,1-2,...,10+, use 100 as upper bound
dur_labels = [f"{dur_bins[i]}-{dur_bins[i+1]}s" for i in range(len(dur_bins)-1)]
dur_labels[-1] = "10+s"  # modify the last label
dur_counts = []
for i in range(len(dur_bins)-1):
    if i == len(dur_bins)-2:
        cnt = (df['duration'] >= dur_bins[i]).sum()
    else:
        cnt = ((df['duration'] >= dur_bins[i]) & (df['duration'] < dur_bins[i+1])).sum()
    dur_counts.append(cnt)

# ==================== 6. Consolidate data ====================
# Assign an X coordinate for each dimension
dim_names = ['Time Distribution\n(5-min bins)', 'Length Distribution\n(word count)', 'Top 15 Keywords', 'Duration Distribution\n(seconds)']
dim_x = [0, 1, 2, 3]
# Y‑axis labels and Z values for each dimension
y_labels = [time_labels, word_labels, keywords, dur_labels]
z_values = [time_counts, word_counts, keyword_counts, dur_counts]

# To facilitate plotting, we determine (x, y, z) coordinates for each bar
# Since the Y‑axis labels have different lengths, we use indices as Y coordinates
xs, ys, zs, colors = [], [], [], []
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # different colors for four dimensions
for dim_idx, (x, yl, zv) in enumerate(zip(dim_x, y_labels, z_values)):
    for y_idx, (y_label, z) in enumerate(zip(yl, zv)):
        xs.append(x)
        ys.append(y_idx)          # Y coordinate uses index
        zs.append(z)
        colors.append(color_palette[dim_idx])

# ==================== 7. Draw 3D bar chart ====================
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Set width and depth of each bar (size along X and Y)
dx = 0.6
dy = 0.6
ax.bar3d(xs, ys, np.zeros_like(zs), dx, dy, zs, shade=True, color=colors, alpha=0.8)

# Add custom Y‑axis tick labels (each dimension region is different; here we annotate with text)
# Because the Y axis is a uniform index, we cannot directly assign different labels per region, so we use text annotations
# Place text at corresponding positions
for dim_idx, (x, yl, zv) in enumerate(zip(dim_x, y_labels, z_values)):
    for y_idx, (y_label, z) in enumerate(zip(yl, zv)):
        if z > 0:  # add label only above bars that have data
            ax.text(x + 0.3, y_idx + 0.2, z + 0.5, y_label, fontsize=6, ha='center', va='bottom', rotation=45)

# Set axis labels and ranges
ax.set_xlabel('Analysis Dimension', fontsize=12)
ax.set_ylabel('Category Index', fontsize=12)
ax.set_zlabel('Frequency', fontsize=12)
ax.set_xticks(dim_x)
ax.set_xticklabels(dim_names, fontsize=9, rotation=15, ha='right')
ax.set_yticks([])  # hide Y axis ticks because we use text annotations
ax.set_title('Combined 3D Bar Chart of Four Dialogue Analyses', fontsize=14)

# Add legend (manual creation)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_palette[i], alpha=0.8, label=dim_names[i]) for i in range(4)]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Adjust viewing angle
ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.savefig('combined_3d_bar.png', dpi=200, bbox_inches='tight')
plt.close()
print("3D bar chart saved as combined_3d_bar.png")