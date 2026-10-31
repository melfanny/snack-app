Backend:  `cd backend && pip install -r requirements.txt && uvicorn main:app --host 0.0.0.0 --port 8001`
Frontend: `cd frontend && npm install && npm run dev -- --host` (buka `http://<IP_LAPTOP>:5173` dari HP; request `/api` otomatis diteruskan Vite ke backend port 8001)
