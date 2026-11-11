"""Membuat checkpoint ringan (hanya bobot EMA) dari checkpoint training.

File asli tidak diubah. Hasil disimpan di sebelahnya dengan akhiran .slim.pth,
dan otomatis dipakai oleh detector.py bila ada.

Pemakaian: python slim_checkpoint.py
"""
from pathlib import Path

import torch

DIR_MODEL = Path(__file__).resolve().parent / "models"
DAFTAR = [DIR_MODEL / "RT-DETR" / "RT-DETR.pth", DIR_MODEL / "RT-DETR-P2" / "RT-DETR-P2.pth"]

for asli in DAFTAR:
    checkpoint = torch.load(asli, map_location="cpu", weights_only=False)
    bobot = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    bobot = {k: v.detach().clone() for k, v in bobot.items()}
    tujuan = asli.with_suffix(".slim.pth")
    torch.save({"ema": {"module": bobot}}, tujuan)

    # Verifikasi: bobot hasil simpan harus identik dengan bobot asli.
    cek = torch.load(tujuan, map_location="cpu", weights_only=False)["ema"]["module"]
    assert cek.keys() == bobot.keys()
    assert all(torch.equal(cek[k], bobot[k]) for k in bobot)
    print(f"{asli.name}: {asli.stat().st_size / 1e6:.0f} MB -> {tujuan.name}: {tujuan.stat().st_size / 1e6:.0f} MB (identik)")
