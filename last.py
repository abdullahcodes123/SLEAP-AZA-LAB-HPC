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
    choose the body part with the highest reliable detection rate
    averaged across both tracks.
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
    pull x,y for track_0 and track_1 for a given bodypart,
    keep only rows where score > threshold
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
    estimate cm-per-pixel scale in x and y for THIS session
    using observed pixel span vs known arena size (36x22 cm)
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
    merge by frame_idx and compute per-frame inter-mouse distance in cm
    with anisotropic scaling (sx for dx_px, sy for dy_px)
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
    return absolute paths for all .csv files in given folder
    """
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]
    csv_files.sort()
    return [os.path.join(folder_path, f) for f in csv_files]


def distances_cm_for_single_csv(csv_path):
    """
    for ONE csv file:
      1. read
      2. pick stable bodypart
      3. get coords for both animals
      4. compute cm/px scale
      5. compute distance series in cm between animals
    """
    df = pd.read_csv(csv_path)

    bodypart = find_most_reliable_bodypart(df)
    t0, t1 = get_xy_for_bodypart(df, bodypart)

    sx, sy = compute_pixel_to_cm_scale(t0, t1, arena_cm_x=36.0, arena_cm_y=22.0)

    dist_cm = compute_distance_series_cm(t0, t1, sx, sy)
    return dist_cm


def collect_distances_for_condition_cm(folder_path):
    """
    for a CONDITION folder (e.g. mozart or whitenoise):
    compute distance_cm array for every CSV.
    return list of arrays (one array per session)
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
    create global bin edges [0,2,4,...,~42] cm
    max_possible ~= diagonal length of arena
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
    edges_cm: shared bin edges [0,2,4,...]

    for each CSV:
        hist = counts per bin
    then average across CSVs (simple mean)
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


########################################
# plotting both conditions on ONE figure
########################################

def plot_both_conditions_histogram_cm(
    centers_cm,
    labels_cm,
    moz_mean_counts,
    wn_mean_counts,
    cond_labels=("mozart", "whitenoise")
):
    """
    single axes, grouped bars:
    - bar for mozart (left offset)
    - bar for whitenoise (right offset)
    legend on the right side so it's obvious which is which
    """
    # we define bar width as 90% of bin spacing, then split that in half
    if len(centers_cm) > 1:
        full_bar_width = (centers_cm[1] - centers_cm[0]) * 0.9
    else:
        full_bar_width = 1.5

    half_width = full_bar_width / 2.0

    # left bars: centers_cm - half_width/2
    moz_positions = centers_cm - (half_width / 2.0)
    # right bars: centers_cm + half_width/2
    wn_positions = centers_cm + (half_width / 2.0)

    plt.figure(figsize=(12, 5))

    # mozart bars
    plt.bar(
        moz_positions,
        moz_mean_counts,
        width=half_width,
        edgecolor="black",
        alpha=0.7,
        label=cond_labels[0],
        color="tab:blue",        # renk 1
    )

    # whitenoise bars
    plt.bar(
        wn_positions,
        wn_mean_counts,
        width=half_width,
        edgecolor="black",
        alpha=0.7,
        label=cond_labels[1],
        color="tab:orange",      # renk 2
    )

    plt.title("Average proximity distribution (mozart vs whitenoise)")
    plt.xlabel("Inter-mouse distance bin (cm)")
    plt.ylabel("Average frame count per bin across sessions")

    # x tickleri ortak merkezlere göre yazıyoruz ama label olarak 0-2 cm vs gösteriyoruz
    plt.xticks(centers_cm, labels_cm, rotation=45, ha="right")

    # legend'i sağ tarafa al
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

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

    # sanity: centers and labels MUST match or plotting breaks
    assert np.allclose(moz_centers, wn_centers), "bin centers mismatch mozart vs whitenoise"
    assert moz_labels == wn_labels, "bin labels mismatch mozart vs whitenoise"

    # 4. plot both on SAME axes with two colors + legend on the right
    plot_both_conditions_histogram_cm(
        centers_cm=moz_centers,
        labels_cm=moz_labels,
        moz_mean_counts=moz_mean_counts,
        wn_mean_counts=wn_mean_counts,
        cond_labels=("mozart", "whitenoise"),
    )


if __name__ == "__main__":
    main()
