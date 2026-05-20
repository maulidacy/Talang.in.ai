# Service utama untuk AI Smart Input Talang.in
# File ini menggabungkan:
# 1. hasil model NER asli
# 2. fallback/rule-based entity
# 3. post-processing transaksi

from src.inference.predict import NERPredictor
from .entity_parser import parse_entities_to_transaction
from .rule_based_ner import predict_entities_rule_based


# Predictor dibuat global agar model tidak reload berulang-ulang setiap request API
_predictor = None


def get_predictor():
    """
    Load model NER sekali saja.
    Jika sudah pernah diload, pakai ulang predictor yang sama.
    """
    global _predictor

    if _predictor is None:
        _predictor = NERPredictor()

    return _predictor


def merge_missing_group_members(text, entities, group_members):
    """
    Menambahkan PERSON dari group_members jika namanya ada di text,
    tetapi tidak berhasil ditangkap oleh model NER.

    Contoh:
    Model menangkap Ayu dan Nina, tapi melewatkan Raka.
    Kalau Raka ada di group_members dan muncul di text, maka Raka ditambahkan.
    """
    if not group_members:
        return entities

    existing_persons = {
        entity["text"].lower()
        for entity in entities
        if entity.get("label") == "PERSON"
    }

    lower_text = text.lower()
    merged_entities = list(entities)

    for member in group_members:
        member_lower = member.lower()

        # Jika nama member sudah ada di hasil model, tidak perlu ditambah lagi
        if member_lower in existing_persons:
            continue

        # Cari posisi nama member di text
        start = lower_text.find(member_lower)

        if start != -1:
            merged_entities.append({
                "text": text[start:start + len(member)],
                "label": "PERSON",
                "start": start,
                "end": start + len(member),
            })

    return sorted(merged_entities, key=lambda item: item.get("start", 0))


def analyze_smart_input(text, entities=None, group_members=None):
    """
    Service utama AI Smart Transaction Input.

    Alur:
    1. Jika frontend mengirim entities manual, pakai entities tersebut.
    2. Jika entities kosong, jalankan model NER asli.
    3. Jika model gagal, pakai fallback rule-based.
    4. Lengkapi PERSON dari group_members jika ada yang terlewat.
    5. Ubah entities menjadi struktur transaksi Talang.in.
    """
    group_members = group_members or []
    entities = entities or []

    # Jika entities tidak dikirim, pakai model NER asli
    if not entities:
        try:
            predictor = get_predictor()
            prediction = predictor.predict_entities(text)
            entities = prediction.get("entities", [])
        except Exception as error:
            # Fallback jika model gagal diload atau inference error
            print(f"Model inference failed, using rule-based fallback: {error}")

            entities = predict_entities_rule_based(
                text=text,
                group_members=group_members,
            )

    # Tambahkan anggota grup yang muncul di text tetapi terlewat oleh model
    entities = merge_missing_group_members(
        text=text,
        entities=entities,
        group_members=group_members,
    )

    # Ubah entity menjadi format transaksi Talang.in
    result = parse_entities_to_transaction(text, entities)

    # Cocokkan format nama peserta dengan nama asli dari group_members
    if group_members:
        result = match_with_group_members(result, group_members)

    result["status"] = "success"
    result["message"] = "AI Smart Input berhasil diproses"

    return result


def match_with_group_members(result, group_members):
    """
    Mencocokkan nama PERSON hasil AI dengan nama anggota grup asli.
    Contoh:
    model membaca 'ayu', group member asli 'Ayu'
    hasil akhir tetap 'Ayu'
    """
    normalized_members = {
        member.lower(): member
        for member in group_members
    }

    matched_participants = []

    for participant in result.get("participants", []):
        raw_name = participant["name"]
        normalized_name = raw_name.lower()

        if normalized_name in normalized_members:
            participant["name"] = normalized_members[normalized_name]
            matched_participants.append(participant)

    if matched_participants:
        result["participants"] = matched_participants

    paid_by = result.get("paidBy", "")

    if paid_by.lower() in normalized_members:
        result["paidBy"] = normalized_members[paid_by.lower()]

    return result