import re
from .price_normalizer import normalize_price


def get_entities_by_label(entities, label):
    """
    Mengambil entity berdasarkan label tertentu.
    """
    return [
        entity
        for entity in entities
        if entity.get("label") == label
    ]


def unique_values(values):
    """
    Menghapus data duplikat tetapi tetap menjaga urutan kemunculan.
    """
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def classify_category(title):
    """
    Klasifikasi kategori sederhana berdasarkan keyword item.
    """
    text = title.lower()
    food_keywords = [
        "kopi", "ayam", "nasi", "mie", "pizza", "burger", "roti",
        "susu", "teh", "matcha", "coklat", "kue", "cake", "biscoff",
        "spaghetti", "cireng", "kambing", "ikan", "soto", "bakso",
        "es", "jus", "steak", "rice", "latte", "boba", "goreng", "kola", "sushi", "sate", "martabak"
    ]
    transport_keywords = ["gojek", "grab", "taxi", "bensin", "parkir"]
    utility_keywords = ["listrik", "air", "wifi", "internet", "pulsa"]

    if any(keyword in text for keyword in food_keywords):
        return "Makanan"
    if any(keyword in text for keyword in transport_keywords):
        return "Transportasi"
    if any(keyword in text for keyword in utility_keywords):
        return "Utilitas"
    return "Lainnya"


def split_amount_exact(amount, members):
    """
    Membagi nominal agar totalnya tetap pas tanpa selisih rupiah.
    """
    if not members:
        return []

    base_amount = amount // len(members)
    remainder = amount % len(members)
    result = []

    for index, name in enumerate(members):
        member_amount = base_amount + (1 if index < remainder else 0)
        result.append({
            "name": name,
            "amount": member_amount,
        })
    return result


def calculate_equal_split(amount, participants):
    """
    Menghitung pembagian rata.
    """
    return split_amount_exact(amount, participants)


def calculate_itemized_split(items, participants):
    """
    Menghitung pembagian akumulatif berdasarkan item-item yang dikonsumsi.
    """
    if not participants:
        return []

    balances = {name: 0 for name in participants}

    for item in items:
        item_members = item.get("members") or participants
        split_result = split_amount_exact(item["amount"], item_members)

        for split in split_result:
            name = split["name"]
            if name not in balances:
                balances[name] = 0
            balances[name] += split["amount"]

    return [
        {"name": name, "amount": amount}
        for name, amount in balances.items()
    ]


def parse_entities_to_transaction(text, entities, group_members=None):
    """
    Mengubah hasil entity NER menjadi struktur transaksi terfragmentasi lengkap.
    """
    group_members = group_members or []
    sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))

    person_entities = get_entities_by_label(sorted_entities, "PERSON")
    item_entities = get_entities_by_label(sorted_entities, "ITEM")
    price_entities = get_entities_by_label(sorted_entities, "PRICE")
    multiplier_entities = get_entities_by_label(sorted_entities, "MULTIPLIER")

    all_global_persons = unique_values([e["text"] for e in person_entities])
    global_paid_by = all_global_persons[0] if all_global_persons else "Unknown"

    valid_items = []
    global_modifiers = {"tax": 0, "discount": 0}
    assigned_price_starts = set()

    if item_entities:
        for i, current_item in enumerate(item_entities):
            item_start = current_item["start"]
            item_end_boundary = item_entities[i + 1]["start"] if i + 1 < len(item_entities) else len(text)
            item_segment_text = text[item_start:item_end_boundary]

            # Prefiks & Perluasan Nama Item
            left_context = text[max(0, item_start - 15):item_start]
            match_prefix = re.search(r'\b(nasi|mie|es|jus|roti|matcha|koka|pizza)\b\s*$', left_context, re.IGNORECASE)
            
            if match_prefix:
                extended_name = f"{match_prefix.group(1)} {current_item['text']}"
                actual_segment_start = text.find(match_prefix.group(1), max(0, item_start - 15))
            else:
                extended_name = current_item["text"]
                actual_segment_start = item_start

            # Kuantitas / Multiplier Nama Item
            local_multipliers = [
                m for m in multiplier_entities
                if max(0, actual_segment_start - 12) <= m["start"] <= item_end_boundary
            ]
            
            quantity = 1
            if local_multipliers:
                m_ent = local_multipliers[0]
                digits = re.findall(r'\d+', m_ent["text"])
                if digits:
                    quantity = int(digits[0])
                if m_ent["start"] < item_start and m_ent["text"] not in extended_name:
                    extended_name = f"{m_ent['text']} {extended_name}"

            segment_text_full = text[actual_segment_start:item_end_boundary]

            # Deteksi Pembayar Spesifik per Item Segment (Multi-Payer)
            item_paid_by = global_paid_by
            lookback_context = text[max(0, actual_segment_start - 35):current_item["end"]]
            for p in all_global_persons:
                if re.search(rf'\b{re.escape(p)}\b\s*(?:bayar|beli|talangin)', lookback_context, re.IGNORECASE):
                    item_paid_by = p

            # Asosiasi Harga Unit vs Harga Total
            local_prices = [
                p for p in price_entities 
                if actual_segment_start <= p["start"] <= item_end_boundary
            ]
            
            item_amount = 0
            if local_prices:
                price_ent = local_prices[0]
                assigned_price_starts.add(price_ent["start"])
                base_price = normalize_price(price_ent["text"])
                
                is_unit_price = "@" in text[max(actual_segment_start, price_ent["start"] - 4):price_ent["start"]]
                item_amount = base_price * quantity if is_unit_price else base_price

            # Filter Pengecualian Anggota
            exclude_match = re.search(r'\b(kecuali|tanpa)\b', segment_text_full, re.IGNORECASE)
            local_persons = [
                p for p in person_entities 
                if actual_segment_start <= p["start"] <= item_end_boundary
            ]
            has_assignment_keyword = bool(re.search(r'\b(untuk|buat|bagi|ke)\b', segment_text_full, re.IGNORECASE))

            # FIX BUG B: Gunakan pool data group asli dari database jika keyword pencabutan aktif
            if exclude_match:
                exclude_idx = actual_segment_start + exclude_match.start()
                excluded_names = {p["text"].lower() for p in local_persons if p["start"] > exclude_idx}
                
                baseline_pool = group_members if group_members else all_global_persons
                item_members = [m for m in baseline_pool if m.lower() not in excluded_names]
                if not item_members:
                    item_members = [item_paid_by]
            elif has_assignment_keyword and local_persons:
                item_members = unique_values([p["text"] for p in local_persons])
                if len(item_members) > 1 and item_paid_by in item_members:
                    payer_pos = text.find(item_paid_by, actual_segment_start)
                    if payer_pos < text.find(current_item["text"], actual_segment_start) and not re.search(rf'\b{item_paid_by}\b', item_segment_text, re.IGNORECASE):
                        item_members = [m for m in item_members if m != item_paid_by]
            else:
                item_members = all_global_persons if all_global_persons else [item_paid_by]

            if item_amount > 0:
                valid_items.append({
                    "name": extended_name,
                    "amount": item_amount,
                    "members": item_members,
                    "paidBy": item_paid_by
                })

    # Menggunakan lookup window terarah lambat untuk mencegah polusi teks sekunder
    for p_ent in price_entities:
        if p_ent["start"] not in assigned_price_starts:
            p_idx = p_ent["start"]
            surrounding_text = text[max(0, p_idx - 12):min(len(text), p_idx + 6)].lower()
            modifier_val = normalize_price(p_ent["text"])
            
            if any(k in surrounding_text for k in ["diskon", "promo", "potongan", "discount"]):
                global_modifiers["discount"] += modifier_val
            elif any(k in surrounding_text for k in ["tax", "pajak", "service", "charge"]):
                global_modifiers["tax"] += modifier_val

    # Distribusi Proporsional Pajak / Diskon ke Setiap Item
    if valid_items:
        split_method = "itemized" if len(valid_items) > 1 else "equal"
        base_total_amount = sum(i["amount"] for i in valid_items)
        net_modifier = global_modifiers["tax"] - global_modifiers["discount"]
        global_amount = max(0, base_total_amount + net_modifier)

        if base_total_amount > 0 and net_modifier != 0:
            allocated_modifier_total = 0
            for idx, item in enumerate(valid_items):
                if idx == len(valid_items) - 1:
                    item["amount"] += (net_modifier - allocated_modifier_total)
                else:
                    proportion = item["amount"] / base_total_amount
                    allocated = int(net_modifier * proportion)
                    item["amount"] += allocated
                    allocated_modifier_total += allocated

        title = ", ".join(unique_values([i["name"] for i in valid_items]))
        participant_amounts = calculate_itemized_split(valid_items, all_global_persons)
    else:
        split_method = "equal"
        title = "Transaksi AI"
        prices = [normalize_price(e["text"]) for e in price_entities]
        global_amount = sum(prices) if prices else 0
        participant_amounts = calculate_equal_split(global_amount, all_global_persons)

    category = classify_category(title)

    return {
        "title": title,
        "amount": global_amount,
        "paidBy": global_paid_by,
        "category": category,
        "splitMethod": split_method,
        "participants": participant_amounts,
        "items": valid_items,
        "rawEntities": sorted_entities,
    }