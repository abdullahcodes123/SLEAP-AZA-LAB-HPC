#!/usr/bin/env bash
set -euo pipefail

# GPU belleği gerektikçe artsın; CPU threadlerini kıs
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-1}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# 0) sleap-track yolu
SLEAP_EXE="${SLEAP_EXE:-/home/at2438/.conda/envs/sleap/bin/sleap-track}"
if [[ ! -x "$SLEAP_EXE" ]]; then
  SLEAP_EXE="$(command -v sleap-track || true)"
fi
[[ -z "${SLEAP_EXE:-}" || ! -x "$SLEAP_EXE" ]] && { echo "HATA: sleap-track bulunamadı."; exit 1; }
echo "sleap-track: $SLEAP_EXE"

# 1) Model kökü ve en güncel run’ları otomatik seç
MODEL_ROOT="${MODEL_ROOT:-/home/at2438/SLEAP_FILES/PROGRESS/training_job_files/models}"
CENTROID_DIR="${CENTROID_DIR:-$(ls -d "$MODEL_ROOT"/*centroid* 2>/dev/null | tail -1)}"
CENTERED_DIR="${CENTERED_DIR:-$(ls -d "$MODEL_ROOT"/*centered_instance* 2>/dev/null | tail -1)}"
[[ -d "${CENTROID_DIR:-}" ]] || { echo "HATA: centroid run bulunamadı."; exit 1; }
[[ -d "${CENTERED_DIR:-}" ]] || { echo "HATA: centered_instance run bulunamadı."; exit 1; }

# 1a) .h5 kontrolü (GUI’deki 'untrained' hatasını önler)
CENT_H5="$(ls "$CENTROID_DIR"/best_model.h5 2>/dev/null || ls "$CENTROID_DIR"/latest_model.h5 2>/dev/null || true)"
INST_H5="$(ls "$CENTERED_DIR"/best_model.h5 2>/dev/null || ls "$CENTERED_DIR"/latest_model.h5 2>/dev/null || true)"
[[ -f "${CENT_H5:-}" ]] || { echo "HATA: $CENTROID_DIR içinde .h5 yok (eğitilmemiş)."; exit 1; }
[[ -f "${INST_H5:-}" ]] || { echo "HATA: $CENTERED_DIR içinde .h5 yok (eğitilmemiş)."; exit 1; }

echo "Centroid: $CENTROID_DIR"
echo "Centered: $CENTERED_DIR"

# 2) Ekran görüntüsü ile uyumlu ayarlar
MAX_INSTANCES="${MAX_INSTANCES:-2}"   # GUI: Max Instances = 2
BATCH_SIZE="${BATCH_SIZE:-4}"         # GUI: Batch Size = 4
TRACKER="${TRACKER:-flow}"            # GUI: Tracker method = flow
SIMILARITY="${SIMILARITY:-centroid}"
MATCH="${MATCH:-hungarian}"
MAX_TRACKS="${MAX_TRACKS:-2}"
TARGET_COUNT="${TARGET_COUNT:-2}"
POST_CONNECT="${POST_CONNECT:-1}"

# Bellek-dostu tracking (gerekirse değiştir)
TRACK_WINDOW="${TRACK_WINDOW:-2}"
OF_WINDOW_SIZE="${OF_WINDOW_SIZE:-15}"
OF_MAX_LEVELS="${OF_MAX_LEVELS:-2}"
IMG_SCALE="${IMG_SCALE:-0.5}"

# 3) Video
cd ~/SLEAP_FILES
[[ -d VIDEOS ]] || { echo "HATA: ~/SLEAP_FILES/VIDEOS yok."; exit 1; }
VID="${VID:-$(ls -1 VIDEOS/* 2>/dev/null | head -n1)}"
[[ -f "${VID:-}" ]] || { echo "HATA: VIDEO bulunamadı."; exit 1; }
echo "Video: $VID"

# 4) Çıktı
OUT_DIR="${OUT_DIR:-/home/at2438/SLEAP_FILES/PROGRESS/training_job_files}"
mkdir -p "$OUT_DIR"
stem="$(basename "$VID")"; stem="${stem%.*}"
OUT="$OUT_DIR/predictions.${stem}.full.slp"
[[ -e "$OUT" ]] && OUT="$OUT_DIR/predictions.${stem}.full.$(date +%y%m%d_%H%M%S).slp"
echo "Çıktı: $OUT"

# 5) Tüm video + tracking
echo "Tahmin + tracking başlıyor..."
"$SLEAP_EXE" \
  -m "$CENTROID_DIR" \
  -m "$CENTERED_DIR" \
  --max_instances "$MAX_INSTANCES" \
  --batch_size "$BATCH_SIZE" \
  --tracking.tracker "$TRACKER" \
  --tracking.similarity "$SIMILARITY" \
  --tracking.match "$MATCH" \
  --tracking.max_tracks "$MAX_TRACKS" \
  --tracking.target_instance_count "$TARGET_COUNT" \
  --tracking.post_connect_single_breaks "$POST_CONNECT" \
  --tracking.track_window "$TRACK_WINDOW" \
  --tracking.of_window_size "$OF_WINDOW_SIZE" \
  --tracking.of_max_levels "$OF_MAX_LEVELS" \
  --tracking.img_scale "$IMG_SCALE" \
  -o "$OUT" \
  "$VID"

echo "Bitti. Çıktı: $OUT"
