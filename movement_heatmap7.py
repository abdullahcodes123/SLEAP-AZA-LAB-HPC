import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


########################################
# detection helpers
########################################

def find_most_reliable_bodypart(df, threshold=0.0):
    """
    Pick the body part with the highest reliable detection rate
    across both tracks.
    """
    bodyparts = sorted({
        col.split(".")[0]
        for col in df.columns
        if col.endswith(".score") and col != "instance.score"
    })

    results = {}
    for bp in bodyparts:
        score_col = f"{bp}.score"
        tmp = df[["track", score_col]].copy()

        pct_by_track = tmp.groupby("track")[score_col].apply(
            lambda s: (s > threshold).mean() * 100
        )
        avg_pct = pct_by_track.mean()

        results[bp] = {"avg_pct": avg_pct}

    highest_finded = max(results, key=lambda bp: results[bp]["avg_pct"])
    return highest_finded


def get_xy_for_bodypart(df, bodypart, threshold=0.0):
    """
    Get x,y for track_0 and track_1 for that body part.
    Keep only rows with score > threshold.
    """
    x_col = f"{bodypart}.x"
    y_col = f"{bodypart}.y"
    score_col = f"{bodypart}.score"

    t0 = df[df["track"] == "track_0"][["frame_idx", x_col, y_col, score_col]].copy()
    t0 = t0[t0[score_col] > threshold]
    t0 = t0.rename(columns={x_col: "x0", y_col: "y0"})

    t1 = df[df["track"] == "track_1"][["frame_idx", x_col, y_col, score_col]].copy()
    t1 = t1[t1[score_col] > threshold]
    t1 = t1.rename(columns={x_col: "x1", y_col: "y1"})

    return t0, t1


########################################
# pixel -> cm calibration
########################################

def compute_pixel_to_cm_scale(t0, t1, arena_cm_x=36.0, arena_cm_y=22.0):
    """
    Estimate cm per pixel in x and y for THIS recording.

    arena is physically 36 cm (x) by 22 cm (y).

    We approximate pixel span by taking min/max of the chosen bodypart
    across both animals.
    """
    all_x = np.concatenate([t0["x0"].to_numpy(), t1["x1"].to_numpy()])
    all_y = np.concatenate([t0["y0"].to_numpy(), t1["y1"].to_numpy()])

    px_span_x = all_x.max() - all_x.min()
    px_span_y = all_y.max() - all_y.min()

    sx = arena_cm_x / px_span_x if px_span_x != 0 else 0.0
    sy = arena_cm_y / px_span_y if px_span_y != 0 else 0.0

    return sx, sy


def compute_distance_series_cm(t0, t1, sx, sy):
    """
    Merge by frame_idx and compute per-frame distance between animals in cm.
    Use anisotropic scaling:
      dx_cm = dx_px * sx
      dy_cm = dy_px * sy
      dist_cm = sqrt(dx_cm^2 + dy_cm^2)
    """
    merged = pd.merge(
        t0[["frame_idx", "x0", "y0"]],
        t1[["frame_idx", "x1", "y1"]],
        on="frame_idx",
        how="inner"
    )
    merged = merged.sort_values("frame_idx").reset_index(drop=True)

    dx_px = merged["x0"] - merged["x1"]
    dy_px = merged["y0"] - merged["y1"]

    dx_cm = dx_px * sx
    dy_cm = dy_px * sy

    dist_cm = np.sqrt(dx_cm**2 + dy_cm**2).to_numpy()
    return dist_cm


########################################
# batch loading per condition
########################################

def load_all_csv_paths(folder_path):
    """
    Return absolute paths for all .csv files in given folder.
    """
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]
    csv_files.sort()
    return [os.path.join(folder_path, f) for f in csv_files]


def distances_cm_for_single_csv(csv_path):
    """
    For one CSV:
    1. read file
    2. pick stable body part
    3. get coords for both animals
    4. compute cm/px scale using arena 36x22 cm
    5. compute distance_cm time series
    """
    df = pd.read_csv(csv_path)

    bodypart = find_most_reliable_bodypart(df)
    t0, t1 = get_xy_for_bodypart(df, bodypart)

    sx, sy = compute_pixel_to_cm_scale(t0, t1, arena_cm_x=36.0, arena_cm_y=22.0)

    dist_cm = compute_distance_series_cm(t0, t1, sx, sy)
    return dist_cm


def collect_distances_for_condition_cm(folder_path):
    """
    For a condition (mozart or whitenoise):
    compute distance_cm array for every CSV in that folder.
    Return list of arrays.
    """
    csv_paths = load_all_csv_paths(folder_path)
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    all_dists_cm = []
    for p in csv_paths:
        dist_cm = distances_cm_for_single_csv(p)
        all_dists_cm.append(dist_cm)

    return all_dists_cm


########################################
# histogram logic with shared edges up to ~42 cm
########################################

def build_common_edges_cm(arena_x_cm=36.0,
                          arena_y_cm=22.0,
                          bin_width_cm=2.0):
    """
    Create global bin edges [0,2,4,...,~42] cm.

    max_possible = diagonal = sqrt(36^2 + 22^2) ~ 42.0+
    Round that up to next multiple of 2 cm
    so both conditions have exactly same x-axis.
    """
    max_possible = math.sqrt(arena_x_cm**2 + arena_y_cm**2)
    max_rounded = math.ceil(max_possible / bin_width_cm) * bin_width_cm

    num_bins = int(max_rounded / bin_width_cm)
    edges_cm = np.linspace(0,
                           max_rounded,
                           num_bins + 1)
    return edges_cm


def average_histogram_cm(all_dist_arrays_cm, edges_cm):
    """
    all_dist_arrays_cm: list of distance(cm) arrays, one per CSV in that condition
    edges_cm: shared bin edges in cm (0,2,4,...)

    For each CSV:
      hist = counts per bin
    Then take mean across CSVs (simple average).
    """
    per_file_counts = []
    for dist_cm in all_dist_arrays_cm:
        counts, _ = np.histogram(dist_cm, bins=edges_cm)
        per_file_counts.append(counts.astype(float))

    per_file_counts = np.stack(per_file_counts, axis=0)  # shape (num_files, num_bins)
    mean_counts = per_file_counts.mean(axis=0)

    centers_cm = 0.5 * (edges_cm[:-1] + edges_cm[1:])
    labels_cm = [
        f"{edges_cm[i]:.0f}-{edges_cm[i+1]:.0f} cm"
        for i in range(len(edges_cm) - 1)
    ]

    return centers_cm, labels_cm, mean_counts


def plot_condition_histogram_cm(centers_cm, labels_cm, mean_counts, condition_name):
    """
    Bar plot of average frame count per distance bin.
    X axis bins are already aligned across conditions.
    """
    if len(centers_cm) > 1:
        bar_width = (centers_cm[1] - centers_cm[0]) * 0.9
    else:
        bar_width = 1.5

    plt.figure(figsize=(10,5))
    plt.bar(
        centers_cm,
        mean_counts,
        width=bar_width,
        edgecolor="black",
        alpha=0.7,
    )

    plt.title(f"Average proximity distribution ({condition_name})")
    plt.xlabel("Inter-mouse distance bin (cm)")
    plt.ylabel("Average frame count per bin across sessions")

    plt.xticks(centers_cm, labels_cm, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


########################################
# main
########################################

def main():
    base_root = os.path.expanduser("~/Desktop/SLEAP_ANALYSIS/MOVEMENT_HEAT_MAP")
    mozart_folder = os.path.join(base_root, "mozart")
    whitenoise_folder = os.path.join(base_root, "whitenoise")

    # 1. collect per-session distances in cm for both conditions
    mozart_dists_cm = collect_distances_for_condition_cm(mozart_folder)
    whitenoise_dists_cm = collect_distances_for_condition_cm(whitenoise_folder)

    # 2. build shared bin edges up to diagonal (~42 cm), step = 2 cm
    edges_cm = build_common_edges_cm(
        arena_x_cm=36.0,
        arena_y_cm=22.0,
        bin_width_cm=2.0
    )

    # 3. average histogram for each condition using SAME edges
    moz_centers, moz_labels, moz_mean_counts = average_histogram_cm(
        mozart_dists_cm,
        edges_cm=edges_cm
    )
    wn_centers, wn_labels, wn_mean_counts = average_histogram_cm(
        whitenoise_dists_cm,
        edges_cm=edges_cm
    )

    # 4. plot separately
    plot_condition_histogram_cm(moz_centers, moz_labels, moz_mean_counts, "mozart")
    plot_condition_histogram_cm(wn_centers, wn_labels, wn_mean_counts, "whitenoise")


if __name__ == "__main__":
    main()
