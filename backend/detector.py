import csv
import io
import logging
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.ops import batched_nms
from torchvision.transforms import v2 as T

# Source RT-DETRv2 PyTorch native (mendaftarkan semua modul, termasuk S2-CCFF).
DIR_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR_BACKEND / "rtdetrv2_pytorch"))
from src.core import YAMLConfig  # noqa: E402

logger = logging.getLogger(__name__)

AMBANG_CONFIDENCE = 0.25
AMBANG_IOU = 0.50
UKURAN_INPUT = (640, 640)  # sama dengan Resize pada val_dataloader saat training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Urutan kelas sama dengan label training (index 0-14, urut alfabet).
NAMA_KELAS = [
    "chitato_aburi",
    "chitato_bumbu_sapi",
    "chitato_lite_ayam_bawang",
    "chitato_lite_rumput_laut",
    "chitato_lite_salmon",
    "chitato_lite_saus_bawang",
    "chitato_sapi_panggang",
    "japota_madu",
    "japota_rumput_laut",
    "potabee_ayam_bakar",
    "potabee_rumput_laut",
    "potabee_sapi_pangang",
    "potabee_spicy",
    "qtela_balado",
    "qtela_ori",
]

# Pilihan model dari frontend beserta konfigurasi YAML dan checkpoint-nya.
# Model "p2" memakai S2CCFFHybridEncoder (modul S2-CCFF) sesuai YAML-nya.
FILE_MODEL = {
    "baseline": {
        "config": DIR_BACKEND / "models/RT-DETR/rtdetrv2_hgnetv2_l_baseline.yml",
        "checkpoint": DIR_BACKEND / "models/RT-DETR/RT-DETR.pth",
    },
    "p2": {
        "config": DIR_BACKEND / "models/RT-DETR-P2/rtdetrv2_hgnetv2_l_s2ccff.yml",
        "checkpoint": DIR_BACKEND / "models/RT-DETR-P2/RT-DETR-P2.pth",
    },
}
model_tersimpan = {}

# Preprocessing sama dengan val_dataloader: Resize 640x640, lalu skala 0-1 (tanpa normalisasi).
transform_gambar = T.Compose([
    T.Resize(UKURAN_INPUT),
    T.PILToTensor(),
    T.ToDtype(torch.float32, scale=True),
])


# Membungkus model dan postprocessor agar dapat dipakai bersama.
class ModelDeteksi:
    def __init__(self, model, postprocessor):
        self.model = model
        self.postprocessor = postprocessor

# Membaca nilai kalori per 100 gram dari file CSV.
def muat_data_kalori():
    data_kalori = {}
    with open("Kalori_snack.csv", newline="", encoding="utf-8-sig") as file:
        baris_csv = csv.reader(file)
        next(baris_csv)
        for baris in baris_csv:
            if len(baris) > 3 and baris[0].strip():
                nilai = baris[3].replace("kkal", "").strip().replace(",", ".")
                data_kalori[baris[0].strip()] = float(nilai)
    return data_kalori


DATA_KALORI = muat_data_kalori()


# Memuat model saat pertama dipakai, lalu memakai ulang model dari cache.
def muat_model(nama_model: str):
    if nama_model not in FILE_MODEL:
        raise ValueError(f"Model '{nama_model}' tidak dikenal. Pilihan: {list(FILE_MODEL)}")
    if nama_model not in model_tersimpan:
        berkas = FILE_MODEL[nama_model]
        try:
            # pretrained=False: bobot lengkap berasal dari checkpoint, bukan unduhan backbone.
            cfg = YAMLConfig(str(berkas["config"]), HGNetv2={"pretrained": False})
            file_bobot = berkas["checkpoint"].with_suffix(".slim.pth")  # versi ringan (slim_checkpoint.py)
            if not file_bobot.exists():
                file_bobot = berkas["checkpoint"]
            checkpoint = torch.load(file_bobot, map_location="cpu", weights_only=False)
            # Evaluasi saat training memakai bobot EMA; 'model' dipakai bila EMA tidak ada.
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
            model = cfg.model
            model.load_state_dict(state, strict=True)
            model = model.deploy().to(DEVICE)
            postprocessor = cfg.postprocessor.to(DEVICE).eval()
            model_tersimpan[nama_model] = ModelDeteksi(model, postprocessor)
        except Exception as error:
            raise RuntimeError(
                f"Model '{nama_model}' gagal dimuat dari {berkas['checkpoint']}: {error}"
            )
        logger.info(
            "Model '%s' dimuat (encoder %s, device %s). Kelas: %s",
            nama_model,
            type(model.encoder).__name__,
            DEVICE,
            NAMA_KELAS,
        )
    return model_tersimpan[nama_model]


# Mengembalikan semua snack yang memiliki kalori paling rendah.
def cari_kalori_terendah(daftar_deteksi):
    if not daftar_deteksi:
        return []
    kalori_minimum = min(item["calories"] for item in daftar_deteksi)
    nama_snack = sorted({
        item["name"]
        for item in daftar_deteksi
        if item["calories"] == kalori_minimum
    })
    return [
        {"name": nama, "calories": kalori_minimum}
        for nama in nama_snack
    ]


# Menjalankan deteksi gambar dan menyusun respons sesuai format API.
def detect(image_bytes: bytes, model_name: str = "baseline"):
    model = muat_model(model_name)
    gambar = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    lebar, tinggi = gambar.size
    masukan = transform_gambar(gambar).unsqueeze(0).to(DEVICE)
    ukuran_asli = torch.tensor([[lebar, tinggi]], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        keluaran = model.model(masukan)
        hasil_model = model.postprocessor(keluaran, ukuran_asli)[0]

    mask_confidence = hasil_model["scores"] >= AMBANG_CONFIDENCE
    labels = hasil_model["labels"][mask_confidence]
    boxes = hasil_model["boxes"][mask_confidence]
    scores = hasil_model["scores"][mask_confidence]
    indeks_terpilih = batched_nms(boxes, scores, labels, AMBANG_IOU)

    daftar_deteksi = []
    for label, kotak, skor in zip(
        labels[indeks_terpilih].tolist(),
        boxes[indeks_terpilih].tolist(),
        scores[indeks_terpilih].tolist(),
    ):
        nama_snack = NAMA_KELAS[int(label)]
        if nama_snack not in DATA_KALORI:
            logger.warning("Nama tidak ada di CSV, dilewati: %s", nama_snack)
            continue
        # Koordinat piksel gambar asli -> normalisasi 0-1, dibatasi pada tepi gambar.
        x1, y1, x2, y2 = kotak
        x1, x2 = (min(max(x1 / lebar, 0.0), 1.0), min(max(x2 / lebar, 0.0), 1.0))
        y1, y2 = (min(max(y1 / tinggi, 0.0), 1.0), min(max(y2 / tinggi, 0.0), 1.0))
        daftar_deteksi.append({
            "name": nama_snack,
            "score": round(float(skor), 2),
            "box": [y1, x1, y2, x2],
            "calories": DATA_KALORI[nama_snack],
        })
    return {
        "detections": daftar_deteksi,
        "lowest_calories": cari_kalori_terendah(daftar_deteksi),
    }
