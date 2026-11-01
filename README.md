### Menjalankan dengan Docker di server

Pastikan Docker Engine dan Docker Compose sudah terpasang, lalu jalankan dari folder utama project:

```sh
docker compose up -d --build
```

Buka `http://<IP_SERVER>/` dari browser. Nginx menyajikan frontend dan meneruskan request `/api` ke backend. Backend menggunakan PyTorch CPU, sehingga server tidak memerlukan GPU atau CUDA. Port 80 server harus tersedia dan dibuka pada firewall. Untuk menghentikan aplikasi, jalankan `docker compose down`.

Untuk melihat log:

```sh
docker compose logs -f
```

### Menjalankan secara lokal untuk pengembangan

Backend: `cd backend && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8001`
Frontend: `cd frontend && npm install && npm run dev -- --host` (buka `http://<IP_LAPTOP>:5173` dari HP; request `/api` otomatis diteruskan Vite ke backend port 8001)
