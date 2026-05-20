from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.integration.smart_input_service import analyze_smart_input


# Membuat aplikasi FastAPI untuk AI Smart Input Talang.in
app = FastAPI(
    title="Talang.in AI Smart Input API",
    description="API untuk memproses AI Smart Transaction Input Talang.in",
    version="1.0.0",
)


# CORS digunakan agar frontend Talang.in bisa memanggil API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Schema entity jika ingin testing dengan entities manual
class Entity(BaseModel):
    text: str
    label: str
    start: Optional[int] = 0
    end: Optional[int] = 0


# Schema request dari frontend
class SmartInputRequest(BaseModel):
    text: str

    # entities opsional.
    # Kalau kosong, sistem akan memakai model NER hasil training.
    entities: List[Entity] = Field(default_factory=list)

    # daftar anggota grup dari frontend Talang.in
    group_members: List[str] = Field(default_factory=list)


@app.get("/")
def root():
    """
    Endpoint untuk mengecek apakah API berjalan.
    """
    return {
        "status": "success",
        "message": "Talang.in AI Smart Input API is running",
    }


@app.post("/parse-transaction")
def parse_transaction(request: SmartInputRequest):
    """
    Endpoint utama AI Smart Transaction Input.

    Alur:
    1. Menerima teks transaksi dari user.
    2. Jika entities kosong, model NER akan memprediksi PERSON, ITEM, PRICE.
    3. Hasil NER diproses menjadi format transaksi Talang.in.
    """

    # Ubah Pydantic object menjadi dictionary biasa
    entities = [entity.model_dump() for entity in request.entities]

    # Jalankan service utama AI Smart Input
    result = analyze_smart_input(
        text=request.text,
        entities=entities,
        group_members=request.group_members,
    )

    return result

# Endpoint ini digunakan untuk mengecek apakah backend AI hidup.
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Talang.in AI backend is running"
    }

# CORS digunakan agar frontend Talang.in bisa memanggil backend AI.
# Localhost untuk development, domain Vercel untuk production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://talang-in.vercel.app",
        "https://domain-vercel-tim.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)