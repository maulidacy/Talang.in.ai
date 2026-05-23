from src.inference.predict import NERPredictor


# Load model hasil training
predictor = NERPredictor()

# Contoh input dari user
text = "Ayu bayar pizza 90k dan es teh 20k untuk Raka dan Nina"

# inference dari weights
predictor_weights = NERPredictor(
    model_path="models/best_ner_model.weights.h5", 
    format_type="weights"
)
print("Result Weights:", predictor_weights.predict_entities(text))
print("-" * 50)

# inference dari .keras
predictor_keras = NERPredictor(
    model_path="models/bill_ner_model.keras", 
    format_type="keras"
)
print("Result Keras Bundle:", predictor_keras.predict_entities(text))
print("-" * 50)