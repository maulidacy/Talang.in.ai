from src.integration.smart_input_service import analyze_smart_input


# Input ini hanya berisi teks mentah, tanpa entities manual.
# Entities akan diprediksi langsung oleh model NER.
text = "Ayu bayar pizza 90k dan es teh 20k untuk Raka dan Nina"

group_members = ["Ayu", "Raka", "Nina", "Budi"]

result = analyze_smart_input(
    text=text,
    group_members=group_members
)

print(result)