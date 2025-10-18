#!/usr/bin/env bash
set -euo pipefail

# =================== KULLANICI AYARLARI ===================
TRAIN_DIR="/home/at2438/SLEAP_FILES/PROGRESS/training_job_files"
COUNT="${COUNT:-20}"                     # rastgele kare sayısı (env ile ezilebilir)
TF_VER="2.7.1"
H5_VER="3.1.0"
# ==========================================================

MODEL_ROOT="$TRAIN_DIR/models"
OUT_DIR="$TRAIN_DIR"

# 0) conda fonksiyonu
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  source "$HOME/.bashrc" || true
fi

# 1) modülleri ve python yollarını temizle
module --force purge || true
unset PYTHONPATH
export PYTHONNOUSERSITE=1

# 2) sadece CUDA + cuDNN (TF modülü yükleme)
module load CUDA/11.3.1 cuDNN/8.2.1.32-CUDA-11.3.1 || true

# 3) ortam
conda activate sleap

# 4) ffprobe yoksa kur
if ! command -v ffprobe >/dev/null 2>&1; then
  conda install -y -c conda-forge ffmpeg
fi

# 5) TF/h5py uyumluluğunu gerekirse düzelt
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

# 6) GPU testi (bilgi amaçlı)
python - <<'PY'
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))
PY

# 7) Eğitim dosyalarını sadece eğitim yapılacaksa kontrol et
if [[ "${SKIP_TRAIN:-0}" -ne 1 ]]; then
  cd "$TRAIN_DIR"
  [ -f centroid.json ] || { echo "centroid.json yok: $TRAIN_DIR"; exit 1; }
  [ -f centered_instance.json ] || { echo "centered_instance.json yok: $TRAIN_DIR"; exit 1; }
  [ -f labels.v001.pkg.slp ] || { echo "labels.v001.pkg.slp yok: $TRAIN_DIR"; exit 1; }

  # viz callback hatasını önlemek için NOVIZ profilini üret
  cp -f centered_instance.json centered_instance.noviz.json
  sed -E -i 's/"save_visualizations":[[:space:]]*true/"save_visualizations": false/' centered_instance.noviz.json
  sed -E -i 's/"visualizations":[[:space:]]*true/"visualizations": false/' centered_instance.noviz.json

  # Eğitim (centroid -> centered_instance)
  sleap-train --gpu auto centroid.json labels.v001.pkg.slp
  sleap-train --gpu auto centered_instance.noviz.json labels.v001.pkg.slp
fi

# 8) En yeni run klasörlerini bul (iki örüntüyü de dene)
latest_run() { ls -dt "$MODEL_ROOT"/$1 2>/dev/null | head -n1 || true; }

CENTROID_DIR="${CENTROID_DIR:-$(latest_run "*centroid*")}"
[[ -z "${CENTROID_DIR:-}" ]] && CENTROID_DIR="$(latest_run "*.centroid")"

CENTERED_DIR="${CENTERED_DIR:-$(latest_run "*centered_instance*")}"
[[ -z "${CENTERED_DIR:-}" ]] && CENTERED_DIR="$(latest_run "*.centered_instance")"

[[ -z "${CENTROID_DIR:-}"  || ! -d "$CENTROID_DIR"  ]] && { echo "HATA: En yeni centroid run bulunamadı ($MODEL_ROOT/*centroid*)."; exit 1; }
[[ -z "${CENTERED_DIR:-}" || ! -d "$CENTERED_DIR" ]] && { echo "HATA: En yeni centered_instance run bulunamadı ($MODEL_ROOT/*centered_instance*)."; exit 1; }

echo "Seçilen centroid run:       $CENTROID_DIR"
echo "Seçilen centered_instance:  $CENTERED_DIR"

# 9) Model girdisini seç: varsa .h5, yoksa klasör
pick_model_input() {
  local d="$1"
  for f in best_model.h5 latest_model.h5; do
    [[ -f "$d/$f" ]] && { echo "$d/$f"; return; }
  done
  echo "$d"
}
CENTROID_IN="$(pick_model_input "$CENTROID_DIR")"
CENTERED_IN="$(pick_model_input "$CENTERED_DIR")"

# 10) sleap-track yolu (önce conda’daki)
SLEAP_EXE="${SLEAP_EXE:-/home/at2438/.conda/envs/sleap/bin/sleap-track}"
if [[ ! -x "$SLEAP_EXE" ]]; then
  SLEAP_EXE="$(command -v sleap-track || true)"
fi
[[ -z "${SLEAP_EXE:-}" || ! -x "$SLEAP_EXE" ]] && { echo "HATA: sleap-track bulunamadı. (conda ortamını aktifleştir)"; exit 1; }
echo "sleap-track: $SLEAP_EXE"

# 11) VID ve OUT: env ile ezilebilir; VID boşsa otomatik seç
VID="${VID:-}"
if [[ -z "$VID" ]]; then
  if ls "$HOME/SLEAP_FILES/VIDEOS/"* >/dev/null 2>&1; then
    VID="$(ls -1 "$HOME/SLEAP_FILES/VIDEOS/"* | head -n1)"
  else
    echo "HATA: Video bulunamadı. VID=... verin veya ~/SLEAP_FILES/VIDEOS içine dosya koyun."
    exit 1
  fi
fi
OUT="${OUT:-$OUT_DIR/predictions.sample${COUNT}.slp}"

echo "Video: $VID"
echo "Çıktı: $OUT"

# 12) Toplam kare sayısını bul (ffprobe varsa)
NFRAMES="${NFRAMES:-}"
if command -v ffprobe >/dev/null 2>&1; then
  frames="$(ffprobe -v error -count_frames -select_streams v:0 \
            -show_entries stream=nb_read_frames -of csv=p=0 "$VID" 2>/dev/null || true)"
  if [[ -z "${frames:-}" || "$frames" == "N/A" ]]; then
    frames="$(ffprobe -v error -select_streams v:0 \
              -show_entries stream=nb_frames -of csv=p=0 "$VID" 2>/dev/null || true)"
  fi
  [[ -n "${frames:-}" && "$frames" != "N/A" ]] && NFRAMES="$frames"
fi
NFRAMES="${NFRAMES:-108000}"

# 13) Rastgele kare listesi
frames_file="$OUT_DIR/frames${COUNT}.txt"
echo "Rastgele $COUNT kare seçiliyor (0..$((NFRAMES-1)))..."
shuf -i 0-$((NFRAMES-1)) -n "$COUNT" | sort -n > "$frames_file"
echo "Seçilen kareler -> $frames_file"
FRAMES_CSV="$(paste -sd, "$frames_file")"

# 14) '--frames' desteği var mı?
if ! "$SLEAP_EXE" --help 2>&1 | grep -q -- '--frames'; then
  echo "Uyarı: bu SLEAP '--frames' desteklemiyor; tüm videoyu işleyecek."
  FRAMES_ARG=()
else
  FRAMES_ARG=(--frames "$FRAMES_CSV")
fi

# 15) Tahmin
echo "Tahmin başlıyor..."
"$SLEAP_EXE" \
  -m "$CENTROID_IN" \
  -m "$CENTERED_IN" \
  "${FRAMES_ARG[@]}" \
  -o "$OUT" \
  "$VID"

echo "Bitti. Çıktı: $OUT"
