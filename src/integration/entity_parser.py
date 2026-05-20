from .price_normalizer import normalize_price


def get_entities_by_label(entities, label):
    """
    Mengambil entity berdasarkan label tertentu.
    Contoh label: PERSON, ITEM, PRICE, MULTIPLIER.
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
    Untuk tahap awal cukup rule-based agar stabil.
    """

    text = title.lower()

    food_keywords = [
        "kopi", "ayam", "nasi", "mie", "pizza", "burger", "roti",
        "susu", "teh", "matcha", "coklat", "kue", "cake", "biscoff",
        "spaghetti", "cireng", "kambing", "ikan", "soto", "bakso",
        "es", "jus", "steak", "rice", "latte", "boba", "goreng",
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
    Membagi nominal agar totalnya tetap pas.

    Contoh:
    110000 dibagi 3:
    - 36667
    - 36667
    - 36666

    Total tetap 110000, tidak lebih 1 rupiah.
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


def pair_items_with_prices(items, prices):
    """
    Memasangkan ITEM dengan PRICE berdasarkan urutan kemunculan.
    Contoh:
    ITEM pizza -> PRICE 90k
    ITEM es teh -> PRICE 20k
    """

    paired_items = []

    for index, item in enumerate(items):
        price_entity = prices[index] if index < len(prices) else None
        amount = normalize_price(price_entity["text"]) if price_entity else 0

        paired_items.append({
            "name": item["text"],
            "amount": amount,
            "members": [],
        })

    return paired_items


def calculate_equal_split(amount, participants):
    """
    Menghitung pembagian rata.
    """
    return split_amount_exact(amount, participants)


def calculate_itemized_split(items, participants):
    """
    Menghitung pembagian berdasarkan item.
    Untuk tahap awal, semua item dibagi ke semua participants.
    Nanti bisa dikembangkan agar setiap item punya member berbeda.
    """

    if not participants:
        return []

    balances = {name: 0 for name in participants}

    for item in items:
        # Jika item belum punya members, default-nya dibagi ke semua participants
        item_members = item.get("members") or participants

        if not item_members:
            continue

        split_result = split_amount_exact(item["amount"], item_members)

        for split in split_result:
            name = split["name"]

            if name not in balances:
                balances[name] = 0

            balances[name] += split["amount"]

        # Simpan members item agar response frontend lengkap
        item["members"] = item_members

    return [
        {
            "name": name,
            "amount": amount,
        }
        for name, amount in balances.items()
    ]


def parse_entities_to_transaction(text, entities):
    """
    Mengubah hasil entity NER menjadi struktur transaksi Talang.in.

    Input:
    - text: raw text dari user
    - entities: hasil model NER

    Output:
    - title
    - amount
    - paidBy
    - category
    - splitMethod
    - participants
    - items
    """

    # Urutkan entity berdasarkan posisi di kalimat
    sorted_entities = sorted(
        entities,
        key=lambda entity: entity.get("start", 0)
    )

    person_entities = get_entities_by_label(sorted_entities, "PERSON")
    item_entities = get_entities_by_label(sorted_entities, "ITEM")
    price_entities = get_entities_by_label(sorted_entities, "PRICE")

    # Ambil nama, item, dan harga
    persons = unique_values([entity["text"] for entity in person_entities])
    items_text = unique_values([entity["text"] for entity in item_entities])
    prices = [normalize_price(entity["text"]) for entity in price_entities]

    # Untuk tahap awal, PERSON pertama dianggap sebagai pembayar
    paid_by = persons[0] if persons else ""

    # Semua PERSON dianggap sebagai peserta split
    participants = persons if persons else []

    # Judul transaksi diambil dari semua ITEM yang terbaca
    title = ", ".join(items_text) if items_text else "Transaksi AI"

    # Kategori ditentukan dari title
    category = classify_category(title)

    # Jika ada keyword total, ambil harga terbesar sebagai total
    has_total_keyword = any(
        keyword in text.lower()
        for keyword in ["total", "grand total", "amount", "nominal"]
    )

    if not prices:
        amount = 0
    elif has_total_keyword:
        amount = max(prices)
    elif len(prices) > 1 and len(item_entities) > 1:
        amount = sum(prices)
    else:
        amount = prices[0]

    # Pasangkan item dan harga
    parsed_items = pair_items_with_prices(item_entities, price_entities)

    # Ambil item yang punya amount valid
    valid_items = [
        item
        for item in parsed_items
        if item["amount"] > 0
    ]

    # Jika ada lebih dari 1 item valid, pakai itemized
    if len(valid_items) > 1:
        split_method = "itemized"

        for item in valid_items:
            item["members"] = participants

        participant_amounts = calculate_itemized_split(valid_items, participants)

    # Jika hanya satu item atau tidak ada item, pakai equal split
    else:
        split_method = "equal"
        valid_items = []
        participant_amounts = calculate_equal_split(amount, participants)

    return {
        "title": title,
        "amount": amount,
        "paidBy": paid_by,
        "category": category,
        "splitMethod": split_method,
        "participants": participant_amounts,
        "items": valid_items,
        "rawEntities": sorted_entities,
    }