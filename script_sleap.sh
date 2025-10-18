#!/usr/bin/env bash
set -euo pipefail

# =================== KULLANICI AYARLARI ===================
VID="/home/at2438/SLEAP_FILES/VIDEOS/cage_b.mp4"
TRAIN_DIR="/home/at2438/SLEAP_FILES/PROGRESS/training_job_files"
CLEAN_COUNT=2           # aynı anda görünen hayvan sayısı; bilmiyorsan 0 bırak
N_FRAMES=20             # rastgele örneklenecek kare sayısı
TF_VER="2.7.1"
H5_VER="3.1.0"
# ==========================================================

# conda fonksiyonu
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$HOME/.bashrc" || true
fi

# modülleri ve python yollarını temizle
module --force purge || true
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# sadece CUDA+cuDNN yükle (TensorFlow modülü YÜKLEME)
module load CUDA/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1 || true

# ortam
conda activate sleap

# ffprobe yoksa kur
if ! command -v ffprobe >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg
fi

# TF/h5py uyumluluğu; versiyon tutmuyorsa düzelt
fix_tf_env() {
  cur_tf="$(python - <<'PY'
try:
    import tensorflow as tf; print(tf.__version__)
except Exception:
    print("")
PY
)"
  cur_h5="$(python - <<'PY'
try:
    import h5py; print(h5py.__version__)
except Exception:
    print("")
PY
)"
  if [ "${cur_tf:-}" != "$TF_VER" ] || [ "${cur_h5:-}" != "$H5_VER" ]; then
    pip install -U pip setuptools wheel
    pip install "tensorflow==$TF_VER" "h5py==$H5_VER"
  fi
}
fix_tf_env

# GPU testi
python - <<'PY'
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))
PY

# dosya kontrolleri
cd "$TRAIN_DIR"
[ -f centroid.json ] || { echo "centroid.json yok: $TRAIN_DIR"; exit 1; }
[ -f centered_instance.json ] || { echo "centered_instance.json yok: $TRAIN_DIR"; exit 1; }
[ -f labels.v001.pkg.slp ] || { echo "labels.v001.pkg.slp yok: $TRAIN_DIR"; exit 1; }

# centered_instance için NOVIZ profilini üret (viz callback hatasını önler)
cp -f centered_instance.json centered_instance.noviz.json
sed -E -i 's/"save_visualizations":[[:space:]]*true/"save_visualizations": false/' centered_instance.noviz.json
sed -E -i 's/"visualizations":[[:space:]]*true/"visualizations": false/' centered_instance.noviz.json

# eğitim
sleap-train --gpu auto centroid.json labels.v001.pkg.slp
sleap-train --gpu auto centered_instance.noviz.json labels.v001.pkg.slp

# en yeni model klasörlerini bul (zaman damgasına göre)
CENT_DIR="$(ls -dt "$TRAIN_DIR"/models/*centroid* 2>/dev/null | head -1 || true)"
INST_DIR="$(ls -dt "$TRAIN_DIR"/models/*centered_instance* 2>/dev/null | head -1 || true)"
if [ -z "${CENT_DIR:-}" ] || [ -z "${INST_DIR:-}" ]; then
  echo "Model klasörleri bulunamadı: $TRAIN_DIR/models altında *centroid* ve *centered_instance* bekleniyor."
  exit 1
fi

# video ve kare sayısı
[ -f "$VID" ] || { echo "Video yok: $VID"; exit 1; }
COUNT="$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames \
        -of default=nokey=1:noprint_wrappers=1 "$VID" 2>/dev/null || true)"
if [ -z "$COUNT" ] || [ "$COUNT" = "N/A" ]; then
  COUNT="$(ffprobe -v error -select_streams v:0 \
          -show_entries stream=nb_frames \
          -of default:nokey=1:noprint_wrappers=1 "$VID" 2>/dev/null || true)"
fi
if [ -z "$COUNT" ] || [ "$COUNT" = "N/A" ]; then
  echo "Frame sayısı alınamadı."; exit 1
fi

# rastgele kareler
N="$N_FRAMES"; if [ "$COUNT" -lt "$N" ]; then N="$COUNT"; fi
FRAMES="$(shuf -i 0:$((COUNT-1)) -n "$N" | sort -n | paste -sd, -)"
echo "Seçilen kareler: $FRAMES"

# çıktı ve tracking
OUT="${VID%.*}.rand${N}.predictions.slp"
sleap-track \
  -m "$CENT_DIR" \
  -m "$INST_DIR" \
  --gpu auto \
  --frames "$FRAMES" \
  --batch_size 32 \
  --tracking.tracker simple \
  --tracking.clean_instance_count "$CLEAN_COUNT" \
  --tracking.clean_iou_threshold 0.3 \
  -o "$OUT" \
  "$VID"

echo "Bitti: $OUT"
