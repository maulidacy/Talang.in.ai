from src.inference.predict import NERPredictor


# Load model hasil training
predictor = NERPredictor()

# Contoh input dari user
text = "Ayu bayar pizza 90k dan es teh 20k untuk Raka dan Nina"

# Jalankan prediksi entity
result = predictor.predict_entities(text)

print(result)