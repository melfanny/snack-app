from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detector import detect

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Menerima foto dan pilihan model, lalu mengembalikan hasil deteksi.
@app.post("/api/detect")
async def api_detect(file: UploadFile = File(...), model: str = Form("baseline")):
    try:
        return detect(await file.read(), model)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
