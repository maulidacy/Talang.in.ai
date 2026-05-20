import re


# Kamus harga informal bahasa Indonesia.
# Digunakan untuk mengubah kata seperti "goceng" atau "cepek" menjadi angka rupiah.
PRICE_WORDS = {
    "goceng": 5000,
    "seceng": 1000,
    "cenggo": 1500,
    "noceng": 2000,
    "ceban": 10000,
    "seceban": 10000,
    "noban": 20000,
    "cepek": 100000,
    "satu cepek": 100000,
    "setengah juta": 500000,
}


def normalize_price(value: str) -> int:
    """
    Mengubah format harga informal menjadi integer rupiah.

    Contoh:
    - "49 K" -> 49000
    - "Rp133k" -> 133000
    - "154 ribu" -> 154000
    - "2.8 juta" -> 2800000
    - "goceng" -> 5000
    """

    if not value:
        return 0

    # Normalisasi teks harga
    text = value.lower().strip()

    # Hapus simbol/kata yang tidak dibutuhkan
    text = text.replace("idr", "")
    text = text.replace("rp.", "")
    text = text.replace("rp", "")
    text = text.replace("±", "")
    text = text.replace("~", "")
    text = text.replace("?", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace(",-", "")
    text = text.replace(",", ".")
    text = re.sub(r"\s+", " ", text).strip()

    # Jika harga berupa kata informal langsung, kembalikan nilainya
    if text in PRICE_WORDS:
        return PRICE_WORDS[text]

    # Jika kata informal ada di dalam kalimat harga
    for word, amount in PRICE_WORDS.items():
        if word in text:
            return amount

    # Hapus kata tambahan yang sering muncul pada harga
    text = text.replace("per orang", "")
    text = text.replace("/orang", "")
    text = text.replace("per porsi", "")
    text = text.replace("orang", "")
    text = text.replace("porsi", "")
    text = text.replace("-an", "")
    text = text.strip()

    multiplier = 1

    # Format juta: 2.8 juta, 5jt, 5J
    if "juta" in text or text.endswith("jt") or text.endswith("j"):
        multiplier = 1_000_000
        text = text.replace("juta", "").replace("jt", "").replace("j", "").strip()

    # Format ribuan: 90k, 90rb, 90 ribu, 90rebu
    elif "ribu" in text or "rebu" in text or text.endswith("rb") or text.endswith("k"):
        multiplier = 1_000
        text = (
            text.replace("ribu", "")
            .replace("rebu", "")
            .replace("rb", "")
            .replace("k", "")
            .strip()
        )

    # Format ribuan dengan titik: 40.000, 254.000
    if re.fullmatch(r"\d{1,3}(\.\d{3})+", text):
        return int(text.replace(".", ""))

    # Ambil angka dari teks
    number_match = re.search(r"\d+(\.\d+)?", text)

    if not number_match:
        return 0

    number = float(number_match.group())

    return int(number * multiplier)