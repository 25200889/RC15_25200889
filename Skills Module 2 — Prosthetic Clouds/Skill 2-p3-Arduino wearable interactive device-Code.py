import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def save_dashboard_sharp_wide_glow(
    csv_file_path,
    out_png_path="combined_dashboard_sharp_wide_glow.png",
    width_in=16,
    height_in=9,
    dpi=120
):
    # --- 1. Read data ---
    try:
        df = pd.read_csv(csv_file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_file_path, encoding='gbk')

    # --- 2. Data cleaning ---
    cols = {col.lower(): col for col in df.columns}

    flex_col = cols.get('flex')
    gsr_col = cols.get('gsr')
    bpm_col = cols.get('bpm')
    if not bpm_col:
        bpm_col = next((col for col in df.columns if 'heart' in col.lower()), None)

    for col in [flex_col, gsr_col, bpm_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    subset_cols = [c for c in [flex_col, gsr_col, bpm_col] if c]
    if subset_cols:
        df = df.dropna(subset=subset_cols)

    if 'time' in df.columns:
        df['time_obj'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce')
        df = df.dropna(subset=['time_obj'])
        time_labels = df['time'].astype(str).values
    else:
        time_labels = np.array([str(i) for i in range(len(df))])

    x_raw = np.arange(len(df))

    # --- 3. Set up canvas ---
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(width_in, height_in), dpi=dpi, sharex=True)
    fig.patch.set_facecolor('black')

    # --- Color map (modified: use only lowest and highest colors for gradient) ---
    def get_gradient_cmap(color_theme):
        if color_theme == 'flex':
            colors = ["#d13b5c", "#ffc89c"]          # lowest -> highest: brighter red-orange gradient
        elif color_theme == 'gsr':
            colors = ["#5a2d8c", "#f0a6d0"]          # lowest -> highest: brighter purple-pink gradient
        elif color_theme == 'bpm':
            colors = ["#0a3b7a", "#7acfb0"]          # lowest -> highest: more saturated blue-green gradient
        else:
            colors = ["black", "white"]
        return LinearSegmentedColormap.from_list(f"{color_theme}_line", colors, N=512)

    # --- Core plotting function (logic unchanged) ---
    def plot_sharp_wide_glow(ax, y_data, color_theme, ylabel):
        ax.set_facecolor('black')

        # 1. Interpolation
        x_new = np.linspace(x_raw.min(), x_raw.max(), num=len(x_raw)*10)
        y_new = np.interp(x_new, x_raw, y_data)

        # 2. Prepare gradient segment data
        points = np.array([x_new, y_new]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        cmap = get_gradient_cmap(color_theme)
        norm = plt.Normalize(y_new.min(), y_new.max())

        # 3. Draw ultra-wide diffuse glow
        glow_color = cmap(0.7)
        y_range = y_new.max() - y_new.min()
        if y_range == 0: y_range = 1.0

        base_spread = y_range * 0.12
        layers = 20
        max_spread_factor = 6.0

        for i in range(layers):
            factor = max_spread_factor * (1 - i/layers)
            current_spread = base_spread * factor
            alpha = 0.01 + (0.03 * (i/layers))

            ax.fill_between(
                x_new,
                y_new - current_spread,
                y_new + current_spread,
                color=glow_color,
                alpha=alpha,
                edgecolor='none',
                lw=0
            )

        # 4. Draw core gradient lines
        lc_main = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.8, alpha=1.0)
        lc_main.set_array(y_new)
        ax.add_collection(lc_main)

        # 5. Axes ranges
        ax.set_xlim(x_raw.min(), x_raw.max())
        margin = y_range * 0.6
        ax.set_ylim(y_new.min() - margin, y_new.max() + margin)

        # 6. Grid and decorations
        ax.minorticks_on()
        ax.grid(True, which='major', color='white', linestyle='--', linewidth=0.5, alpha=0.15)
        ax.grid(True, which='minor', color='white', linestyle=':', linewidth=0.3, alpha=0.05)

        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_alpha(0.4)
            spine.set_linestyle('--')
            spine.set_linewidth(0.5)

        ax.set_ylabel(ylabel, color='white', fontsize=10, fontname='monospace')
        ax.tick_params(axis='y', colors='white', labelsize=8)

    # --- 4. Plot each trace ---
    if flex_col:
        plot_sharp_wide_glow(ax1, df[flex_col].values, 'flex', 'Flex')

    if gsr_col:
        plot_sharp_wide_glow(ax2, df[gsr_col].values, 'gsr', 'uS')

    if bpm_col:
        plot_sharp_wide_glow(ax3, df[bpm_col].values, 'bpm', 'BPM')

    # --- 5. Bottom X-axis ---
    num_grids = 12
    tick_indices = np.linspace(0, len(df)-1, num_grids, dtype=int)
    ax3.set_xticks(tick_indices)

    target_labels = 6
    show_every = max(num_grids // target_labels, 1)

    labels = []
    for i, idx in enumerate(tick_indices):
        labels.append(time_labels[idx] if (i % show_every == 0) else "")

    ax3.set_xticklabels(labels, rotation=0, ha='center', fontsize=9, color='white')
    ax3.set_xlabel('Time', color='white', fontsize=12, fontname='monospace')
    ax3.tick_params(axis='x', colors='white')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.suptitle('Bio-Feedback Sharp Wide Glow', color='white', fontsize=16, fontname='monospace', y=0.97)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.10, hspace=0.15)
    plt.savefig(out_png_path, dpi=dpi, facecolor='black')
    plt.show()

# --- Run ---
file_path = 'data2 Final Version.csv'
save_dashboard_sharp_wide_glow(file_path)