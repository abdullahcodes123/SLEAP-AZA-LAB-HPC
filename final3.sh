#!/usr/bin/env bash
set -euo pipefail

# =================== KULLANICI AYARLARI ===================
TRAIN_DIR="${TRAIN_DIR:-/home/at2438/SLEAP_FILES/PROGRESS/training_job_files}"
COUNT="${COUNT:-20}"                 # rastgele kare sayısı
TF_VER="${TF_VER:-2.7.1}"
H5_VER="${H5_VER:-3.1.0}"

# Eski→yeni kök dizin dönüşümleri (virgül ile ayır, 'ESKI::YENI' biçimi)
# Örnek (Mac→HPC ve tersine): PATH_MAP="/Users/abdullah/SLEAP_FILES::/home/at2438/SLEAP_FILES,/Volumes/data::/home/at2438/data"
PATH_MAP="${PATH_MAP:-}"

# Video arama kökleri (boşsa makul tahminler)
if [[ -n "${VIDEO_HINTS:-}" ]]; then
  read -r -a VIDEO_HINTS_ARR <<<"$VIDEO_HINTS"
else
  VIDEO_HINTS_ARR=(
    "$HOME/SLEAP_FILES/VIDEOS"
    "/home/at2438/SLEAP_FILES/VIDEOS"
    "/home/at2438/SLEAP_FILES"
    "/Users/$USER/SLEAP_FILES/VIDEOS"
    "/Users/$USER/SLEAP_FILES"
  )
fi
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

# 7) Eğitim (opsiyonel)
if [[ "${SKIP_TRAIN:-0}" -ne 1 ]]; then
  cd "$TRAIN_DIR"
  [ -f centroid.json ] || { echo "centroid.json yok: $TRAIN_DIR"; exit 1; }
  [ -f centered_instance.json ] || { echo "centered_instance.json yok: $TRAIN_DIR"; exit 1; }
  [ -f labels.v001.pkg.slp ] || { echo "labels.v001.pkg.slp yok: $TRAIN_DIR"; exit 1; }

  # NOVIZ profilini üret
  cp -f centered_instance.json centered_instance.noviz.json
  sed -E -i 's/"save_visualizations":[[:space:]]*true/"save_visualizations": false/' centered_instance.noviz.json
  sed -E -i 's/"visualizations":[[:space:]]*true/"visualizations": false/' centered_instance.noviz.json

  # --video-paths argümanlarını hazırla (eğitimde .slp içindeki farklı path’leri çözmek için)
  VP=(); for p in "${VIDEO_HINTS_ARR[@]}"; do [[ -d "$p" ]] && VP+=(--video-paths "$p"); done

  sleap-train --gpu auto centroid.json            labels.v001.pkg.slp "${VP[@]}"
  sleap-train --gpu auto centered_instance.noviz.json labels.v001.pkg.slp "${VP[@]}"
  # Eğitimde video yolu araması: resmi destek.  [oai_citation:3‡SLEAP Documentation](https://docs.sleap.ai/dev/reference/command-line-interfaces/?utm_source=chatgpt.com)
fi

# 8) En yeni run klasörlerini bul
latest_run() { ls -dt "$MODEL_ROOT"/$1 2>/dev/null | head -n1 || true; }

CENTROID_DIR="${CENTROID_DIR:-$(latest_run "*centroid*")}"
[[ -z "${CENTROID_DIR:-}" ]] && CENTROID_DIR="$(latest_run "*.centroid")"

CENTERED_DIR="${CENTERED_DIR:-$(latest_run "*centered_instance*")}"
[[ -z "${CENTERED_DIR:-}" ]] && CENTERED_DIR="$(latest_run "*.centered_instance")"

[[ -z "${CENTROID_DIR:-}"  || ! -d "$CENTROID_DIR"  ]] && { echo "HATA: En yeni centroid run yok ($MODEL_ROOT/*centroid*)."; exit 1; }
[[ -z "${CENTERED_DIR:-}" || ! -d "$CENTERED_DIR" ]] && { echo "HATA: En yeni centered_instance run yok ($MODEL_ROOT/*centered_instance*)."; exit 1; }

echo "Seçilen centroid run:       $CENTROID_DIR"
echo "Seçilen centered_instance:  $CENTERED_DIR"

# 9) Model girdisini seç
pick_model_input() {
  local d="$1"
  for f in best_model.h5 latest_model.h5; do
    [[ -f "$d/$f" ]] && { echo "$d/$f"; return; }
  done
  echo "$d"
}
CENTROID_IN="$(pick_model_input "$CENTROID_DIR")"
CENTERED_IN="$(pick_model_input "$CENTERED_DIR")"

# 10) sleap-track yolu
SLEAP_EXE="${SLEAP_EXE:-$(command -v sleap-track || true)}"
[[ -z "${SLEAP_EXE:-}" || ! -x "$SLEAP_EXE" ]] && { echo "HATA: sleap-track bulunamadı. (conda ortamını aktifleştir)"; exit 1; }
echo "sleap-track: $SLEAP_EXE"

# 11) VID ve OUT
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

echo "Girdi: $VID"
echo "Çıktı: $OUT"

# 12) Eğer girdi .slp ise, path’leri otomatik düzelt (.pathfixed.slp üret)
fix_slp_if_needed() {
  local in_slp="$1"
  [[ "${in_slp##*.}" != "slp" ]] && { echo "$in_slp"; return; }

  local out_slp="${in_slp%.slp}.pathfixed.slp"
  SLP_IN="$in_slp" SLP_OUT="$out_slp" PY_PATH_MAP="$PATH_MAP" PY_HINTS="${VIDEO_HINTS_ARR[*]}" python - <<'PY' || true
import os, json, sys
in_path = os.environ["SLP_IN"]
out_path = os.environ["SLP_OUT"]
pm = os.environ.get("PY_PATH_MAP","").strip()
hints = [p for p in os.environ.get("PY_HINTS","").split(" ") if p]

def parse_map(s):
    m = {}
    if not s: return m
    for pair in s.split(","):
        if "::" in pair:
            old, new = pair.split("::",1)
            if old and new: m[old]=new
    return m

prefix_map = parse_map(pm)

# Önce sleap-io varsa doğrudan isimleri değiştir
try:
    import sleap_io as sio
    labels = sio.load_file(in_path, open_videos=False)
    if prefix_map:
        labels.replace_filenames(prefix_map=prefix_map)
    # Hints varsa ve video hala bulunamıyorsa, deneyerek yükle
    if hints:
        try:
            labels2 = sio.load_file(in_path, search_paths=hints)
            labels = labels2
        except Exception:
            pass
    labels.save(out_path)
    sys.exit(0)
except Exception:
    pass

# sleap-io yoksa legacy sleap ile search_paths kullan
try:
    import sleap
    from sleap.io.dataset import Labels
    labels = Labels.load_file(in_path, detect_videos=True, search_paths=hints if hints else None)
    labels.save(out_path)   # Kaydettiğinde yeni video yolları yazılır.  #  [oai_citation:4‡SLEAP](https://legacy.sleap.ai/api/sleap.io.dataset.html)
    sys.exit(0)
except Exception as e:
    sys.exit(1)
PY

  if [[ -f "$out_slp" ]]; then
    echo "SLP path düzeltildi -> $out_slp"
    echo "$out_slp"
  else
    echo "Uyarı: .slp path fix başarısız, orijinal kullanılacak."
    echo "$in_slp"
  fi
}

if [[ "${VID##*.}" == "slp" ]]; then
  VID="$(fix_slp_if_needed "$VID")"
fi

# 13) Toplam kare sayısını bul (video ise)
NFRAMES="${NFRAMES:-}"
if [[ "${VID##*.}" != "slp" ]] && command -v ffprobe >/dev/null 2>&1; then
  frames="$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$VID" 2>/dev/null || true)"
  if [[ -z "${frames:-}" || "$frames" == "N/A" ]]; then
    frames="$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$VID" 2>/dev/null || true)"
  fi
  [[ -n "${frames:-}" && "$frames" != "N/A" ]] && NFRAMES="$frames"
fi
NFRAMES="${NFRAMES:-108000}"

# 14) Rastgele kare listesi (yalnızca video girdisinde anlamlı)
frames_file="$OUT_DIR/frames${COUNT}.txt"
if [[ "${VID##*.}" != "slp" ]]; then
  echo "Rastgele $COUNT kare seçiliyor (0..$((NFRAMES-1)))..."
  shuf -i 0-$((NFRAMES-1)) -n "$COUNT" | sort -n > "$frames_file"
  FRAMES_CSV="$(paste -sd, "$frames_file")"
else
  FRAMES_CSV=""
fi

# 15) '--frames' desteği var mı?
if [[ -n "$FRAMES_CSV" ]] && ! "$SLEAP_EXE" --help 2>&1 | grep -q -- '--frames'; then
  echo "Uyarı: bu SLEAP '--frames' desteklemiyor; tüm videoyu işleyecek."
  FRAMES_ARG=()
else
  FRAMES_ARG=()
  [[ -n "$FRAMES_CSV" ]] && FRAMES_ARG=(--frames "$FRAMES_CSV")
fi

# 16) Tahmin
echo "Tahmin başlıyor..."
"$SLEAP_EXE" \
  -m "$CENTROID_IN" \
  -m "$CENTERED_IN" \
  "${FRAMES_ARG[@]}" \
  -o "$OUT" \
  "$VID"

echo "Bitti. Çıktı: $OUT"
