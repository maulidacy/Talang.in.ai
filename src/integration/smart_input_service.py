# Service utama untuk AI Smart Input Talang.in
# File ini menggabungkan:
# 1. hasil model NER asli
# 2. fallback/rule-based entity
# 3. post-processing transaksi

import re
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
    Menangkap SEMUA kemunculan nama anggota grup di dalam teks,
    termasuk jika nama tersebut muncul beberapa kali di baris berbeda.
    """
    if not group_members:
        return entities

    lower_text = text.lower()
    merged_entities = list(entities)

    for member in group_members:
        member_lower = member.lower()
        
        # Gunakan regex finditer untuk mencari semua indeks kemunculan nama
        for match in re.finditer(rf'\b{re.escape(member_lower)}\b', lower_text):
            start = match.start()
            end = match.end()

            # Pastikan posisi ini belum dicover oleh entitas yang sudah ada
            is_covered = any(e.get("start", 0) <= start < e.get("end", 0) for e in merged_entities)
            
            if not is_covered:
                merged_entities.append({
                    "text": text[start:end],
                    "label": "PERSON",
                    "start": start,
                    "end": end,
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
    result = parse_entities_to_transaction(text, entities, group_members=group_members)

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