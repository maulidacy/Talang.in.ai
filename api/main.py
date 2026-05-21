from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# analyze_smart_input dipakai endpoint utama untuk memproses AI Smart Input.
# get_predictor dipakai untuk load model lebih awal agar request pertama tidak terlalu lama.
from src.integration.smart_input_service import analyze_smart_input, get_predictor


# Membuat aplikasi FastAPI untuk backend AI Smart Input Talang.in
app = FastAPI(
    title="Talang.in AI Smart Input API",
    description="API untuk memproses AI Smart Transaction Input Talang.in",
    version="1.0.0",
)


# CORS digunakan agar frontend Talang.in bisa memanggil backend AI.
# Localhost dipakai untuk development.
# Domain Vercel dipakai untuk production.
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


# Schema entity jika ingin testing dengan entities manual.
class Entity(BaseModel):
    text: str
    label: str
    start: Optional[int] = 0
    end: Optional[int] = 0


# Schema request dari frontend.
class SmartInputRequest(BaseModel):
    text: str

    # entities opsional.
    # Kalau kosong, sistem akan memakai model NER hasil training.
    entities: List[Entity] = Field(default_factory=list)

    # daftar anggota grup dari frontend Talang.in.
    # Ini dipakai agar hasil PERSON bisa dicocokkan dengan member grup.
    group_members: List[str] = Field(default_factory=list)


def entity_to_dict(entity: Entity):
    """
    Mengubah object Pydantic menjadi dictionary.

    Dibuat kompatibel untuk Pydantic v1 dan v2:
    - Pydantic v2 memakai model_dump()
    - Pydantic v1 memakai dict()
    """
    if hasattr(entity, "model_dump"):
        return entity.model_dump()

    return entity.dict()


# Warm-up otomatis saat server start.
# Tujuannya agar model TensorFlow diload ke memory sejak awal,
# sehingga request pertama dari user tidak terlalu lama.
@app.on_event("startup")
def startup_warmup_model():
    try:
        predictor = get_predictor()

        # Prediksi dummy untuk memastikan model, weights, dan TensorFlow graph siap.
        predictor.predict_entities("Ayu bayar kopi 10k untuk Budi")

        print("AI model warm-up saat startup berhasil.")
    except Exception as error:
        # Kalau warm-up gagal, server tetap hidup.
        # Sistem masih bisa fallback ke rule-based jika model gagal.
        print(f"AI model warm-up saat startup gagal: {error}")


@app.get("/")
def root():
    """
    Endpoint root untuk mengecek apakah API berjalan.
    """
    return {
        "status": "success",
        "message": "Talang.in AI Smart Input API is running",
    }


@app.get("/health")
def health_check():
    """
    Endpoint health check untuk development dan production.
    """
    return {
        "status": "ok",
        "message": "Talang.in AI backend is running",
    }


@app.get("/warmup")
def warmup_endpoint():
    """
    Endpoint warm-up manual.

    Bisa dibuka sebelum demo:
    http://127.0.0.1:8000/warmup

    Tujuannya agar model sudah siap sebelum user mencoba AI Smart Input.
    """
    try:
        predictor = get_predictor()

        # Prediksi dummy agar model siap digunakan.
        predictor.predict_entities("Ayu bayar kopi 10k untuk Budi")

        return {
            "status": "success",
            "message": "AI model sudah berhasil di-warm-up",
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }


@app.post("/parse-transaction")
def parse_transaction(request: SmartInputRequest):
    """
    Endpoint utama AI Smart Transaction Input.

    Alur:
    1. Menerima teks transaksi dari user.
    2. Jika entities kosong, model NER akan memprediksi PERSON, ITEM, PRICE.
    3. Hasil NER diproses menjadi format transaksi Talang.in.
    4. Response dikirim ke frontend untuk mengisi form transaksi.
    """

    # Ubah Pydantic object menjadi dictionary biasa.
    entities = [entity_to_dict(entity) for entity in request.entities]

    # Jalankan service utama AI Smart Input.
    result = analyze_smart_input(
        text=request.text,
        entities=entities,
        group_members=request.group_members,
    )

    return result