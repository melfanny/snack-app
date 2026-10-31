import __main__
import csv
import io
import logging

from PIL import Image
from ultralytics import RTDETR

from modules.custom_modules import CSPOK, OmniKernelBlock, SPDConv

# Checkpoint dapat menyimpan lokasi class custom di namespace __main__.
__main__.SPDConv = SPDConv
__main__.CSPOK = CSPOK
__main__.OmniKernelBlock = OmniKernelBlock

logger = logging.getLogger(__name__)

AMBANG_CONFIDENCE = 0.50

# Pilihan model dari frontend beserta lokasi file bobotnya.
FILE_MODEL = {
    "baseline": "models/RT-DETR.pt",
    "p2": "models/RT-DETR-P2.pt",
}
model_tersimpan = {}


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
        try:
            model_tersimpan[nama_model] = RTDETR(FILE_MODEL[nama_model])
        except Exception as error:
            raise RuntimeError(
                f"Model '{nama_model}' gagal dimuat dari {FILE_MODEL[nama_model]}: {error}"
            )
        logger.info(
            "Model '%s' dimuat. Kelas: %s",
            nama_model,
            list(model_tersimpan[nama_model].names.values()),
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
    gambar = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    hasil_model = model.predict(
        gambar,
        conf=AMBANG_CONFIDENCE,
        verbose=False,
    )[0]

    daftar_deteksi = []
    for kotak in hasil_model.boxes:
        nama_snack = model.names[int(kotak.cls)]
        if nama_snack not in DATA_KALORI:
            logger.warning("Nama tidak ada di CSV, dilewati: %s", nama_snack)
            continue
        x1, y1, x2, y2 = kotak.xyxyn[0].tolist()
        daftar_deteksi.append({
            "name": nama_snack,
            "score": round(float(kotak.conf), 2),
            "box": [y1, x1, y2, x2],
            "calories": DATA_KALORI[nama_snack],
        })
    return {
        "detections": daftar_deteksi,
        "lowest_calories": cari_kalori_terendah(daftar_deteksi),
    }
