import re


# Pattern untuk mendeteksi harga informal.
# Ini dipakai sebagai fallback jika model NER belum menangkap entity tertentu.
PRICE_PATTERN = re.compile(
    r"(rp\s*)?\d+([.,]\d+)?\s*(k|rb|ribu|rebu|jt|juta|j)?(-an)?|"
    r"goceng|seceng|cenggo|noceng|ceban|seceban|noban|cepek|satu cepek",
    re.IGNORECASE,
)


# Pattern untuk mendeteksi jumlah orang/porsi.
# Contoh: x4, 4 orang, 6 kepala, 2 porsi.
MULTIPLIER_PATTERN = re.compile(
    r"\bx\d+\b|\b\d+\s*(orang|kepala|porsi)\b",
    re.IGNORECASE,
)


def predict_entities_rule_based(text: str, group_members=None):
    """
    Fallback sederhana untuk mengambil entity dari teks.

    Fungsi ini dipakai jika:
    1. model NER gagal diload
    2. model NER belum menangkap beberapa entity
    3. ada nama anggota grup yang terlewat oleh model

    Output dibuat sama seperti hasil model:
    [
      {"text": "...", "label": "PERSON", "start": 0, "end": 3}
    ]
    """

    entities = []
    group_members = group_members or []

    # Deteksi PERSON berdasarkan nama anggota grup.
    # Ini membantu jika model melewatkan nama seperti Raka/Nina.
    for member in group_members:
        pattern = re.compile(rf"\b{re.escape(member)}\b", re.IGNORECASE)

        for match in pattern.finditer(text):
            entities.append({
                "text": match.group(),
                "label": "PERSON",
                "start": match.start(),
                "end": match.end(),
            })

    # Deteksi PRICE berdasarkan regex harga informal.
    for match in PRICE_PATTERN.finditer(text):
        raw_price = match.group().strip()

        if not raw_price:
            continue

        entities.append({
            "text": raw_price,
            "label": "PRICE",
            "start": match.start(),
            "end": match.end(),
        })

    # Deteksi MULTIPLIER seperti x4, 5 orang, 6 kepala.
    for match in MULTIPLIER_PATTERN.finditer(text):
        entities.append({
            "text": match.group(),
            "label": "MULTIPLIER",
            "start": match.start(),
            "end": match.end(),
        })

    # Deteksi ITEM sederhana dari pola kalimat.
    # Ini hanya fallback, nanti tetap lebih utama dari model NER asli.
    item_entities = extract_item_candidates(text)
    entities.extend(item_entities)

    # Urutkan entity berdasarkan posisi kemunculan di teks.
    entities = sorted(entities, key=lambda item: item["start"])

    return entities


def extract_item_candidates(text: str):
    """
    Mengambil kandidat ITEM secara rule-based.

    Contoh:
    "Ayu bayar pizza 90k dan es teh 20k untuk Raka"
    akan mencoba mengambil:
    - pizza
    - es teh

    Catatan:
    Ini hanya fallback sementara, bukan pengganti model NER.
    """

    items = []

    # Pattern untuk menangkap item setelah kata tertentu.
    patterns = [
        r"bayar\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
        r"beli\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
        r"tagihan\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
        r"nota\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
        r"rekap\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
        r"billing\s+(.+?)(?=\s+\d|\s+rp|\s+untuk|\s+buat|$)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).strip(" .,-:")

            if not candidate:
                continue

            # Pecah jika ada beberapa item yang dipisahkan "dan" atau koma.
            parts = re.split(r"\s+dan\s+|,", candidate)

            cursor = match.start(1)

            for part in parts:
                item_name = part.strip(" .,-:")

                if not item_name:
                    continue

                # Cari posisi item di teks asli.
                start = text.lower().find(item_name.lower(), cursor)

                if start == -1:
                    start = cursor

                end = start + len(item_name)

                items.append({
                    "text": item_name,
                    "label": "ITEM",
                    "start": start,
                    "end": end,
                })

                cursor = end

    return items