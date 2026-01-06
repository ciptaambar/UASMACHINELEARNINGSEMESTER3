from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot_engine import ChatbotEngine

app = FastAPI(
    title="Beauty Paw Chatbot API",
    description="API untuk chatbot layanan pelanggan toko skincare Beauty Paw",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    chatbot = ChatbotEngine(model_dir="models", dataset_path="datasets.json")
    print("Chatbot berhasil diinisialisasi")
except Exception as e:
    print(f"Gagal menginisialisasi chatbot: {e}")
    chatbot = None

class PermintaanChat(BaseModel):
    message: str
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Halo, apa itu serum?"
            }
        }

class ResponChat(BaseModel):
    pesan_pengguna: str
    respon_bot: str
    intent: str
    kepercayaan: float
    class Config:
        json_schema_extra = {
            "example": {
                "pesan_pengguna": "Halo, apa itu serum?",
                "respon_bot": "Serum adalah produk konsentrat...",
                "intent": "serum_info",
                "kepercayaan": 0.95
            }
        }

@app.get("/")
async def beranda():
    return {
        "status": "aktif",
        "layanan": "Beauty Paw Chatbot API",
        "endpoint": {
            "chat": "/chat (POST)",
            "dokumentasi": "/docs",
            "redoc": "/redoc"
        }
    }

@app.post("/chat", response_model=ResponChat)
async def chat(request: PermintaanChat):
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Layanan chatbot tidak tersedia. Pastikan model sudah dilatih."
        )

    if not request.message or request.message.strip() == "":
        raise HTTPException(
            status_code=400,
            detail="Pesan tidak boleh kosong"
        )

    try:
        result = chatbot.get_response(request.message)

        return ResponChat(
            pesan_pengguna=request.message,
            respon_bot=result["response"],
            intent=result["intent"],
            kepercayaan=result["confidence"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses pesan: {str(e)}"
        )

@app.get("/intent")
async def daftar_intent():
    if chatbot is None:
        raise HTTPException(
            status_code=503,
            detail="Layanan chatbot tidak tersedia"
        )
    return {
        "total_intent": len(chatbot.intent_responses),
        "daftar_intent": list(chatbot.intent_responses.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
