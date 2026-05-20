import json
import os

import numpy as np
import tensorflow as tf

from src.models.ner_model import BillNERModel
from src.preprocessing.tokenizer import BillTokenizer


class NERPredictor:
    """
    Class untuk menjalankan inference model NER.

    Input:
    - raw text dari user

    Output:
    - entities seperti PERSON, ITEM, PRICE, MULTIPLIER
    """

    def __init__(
        self,
        model_weights_path="models/best_ner_model.weights.h5",
        vocab_dir="outputs/vocabs",
        config_path="outputs/training_config.json",
    ):
        # Path hasil training
        self.model_weights_path = model_weights_path
        self.vocab_dir = vocab_dir
        self.config_path = config_path

        # Tokenizer dipakai untuk memecah text menjadi token dan span posisi
        self.tokenizer = BillTokenizer()

        # Load vocabulary hasil training
        self.word2idx = self._load_json(os.path.join(vocab_dir, "word2idx.json"))
        self.char2idx = self._load_json(os.path.join(vocab_dir, "char2idx.json"))
        self.tag2idx = self._load_json(os.path.join(vocab_dir, "tag2idx.json"))

        # Load konfigurasi model saat training
        self.config = self._load_json(config_path)

        # Membalik tag2idx menjadi idx2tag
        # Contoh: 3 -> B-PERSON
        self.idx2tag = {
            int(index): tag
            for tag, index in self.tag2idx.items()
        }

        # Build ulang arsitektur model
        self.model = self._build_model()

        # Load bobot model hasil training
        self.model.load_weights(model_weights_path)

    def _load_json(self, path):
        """
        Membaca file JSON.
        """
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _build_model(self):
        """
        Membuat ulang arsitektur model dengan ukuran yang sama seperti saat training.
        Model harus dibuild dulu sebelum load_weights.
        """

        model = BillNERModel(
            vocab_size=self.config["vocab_size"],
            char_vocab=self.config["char_vocab"],
            num_tags=self.config["num_tags"],
            word_embed_dim=self.config["word_embed_dim"],
            char_embed_dim=self.config["char_embed_dim"],
            char_cnn_dim=self.config["char_cnn_dim"],
            bilstm_units=self.config["bilstm_units"],
            dropout=self.config["dropout"],
        )

        # Dummy input untuk membangun layer model sebelum load weights
        dummy_inputs = {
            "word_inputs": tf.zeros((1, self.config["max_len"]), dtype=tf.int32),
            "char_inputs": tf.zeros(
                (1, self.config["max_len"], self.config["max_word_len"]),
                dtype=tf.int32,
            ),
        }

        # Panggil model sekali agar semua layer terbentuk
        model(dummy_inputs, training=False)

        return model

    def _vectorize_text(self, text):
        """
        Mengubah raw text menjadi input numerik untuk model:
        - word_inputs untuk word embedding
        - char_inputs untuk character CNN
        """

        # Ambil token dan posisi karakter dari tokenizer
        token_spans = self.tokenizer._tokenize_spans(text)
        tokens = [token for token, _, _ in token_spans]

        max_len = self.config["max_len"]
        max_word_len = self.config["max_word_len"]

        # Siapkan array kosong sesuai ukuran input model
        word_inputs = np.zeros((1, max_len), dtype=np.int32)
        char_inputs = np.zeros((1, max_len, max_word_len), dtype=np.int32)

        # Isi word index dan character index
        for token_index, token in enumerate(tokens[:max_len]):
            word_inputs[0, token_index] = self.word2idx.get(
                token,
                self.word2idx.get("[UNK]", 1),
            )

            for char_index, char in enumerate(token[:max_word_len]):
                char_inputs[0, token_index, char_index] = self.char2idx.get(
                    char,
                    self.char2idx.get("[UNK]", 1),
                )

        return {
            "word_inputs": word_inputs,
            "char_inputs": char_inputs,
        }, token_spans[:max_len]

    def _bio_to_entities(self, text, token_spans, tags):
        """
        Mengubah BIO tags menjadi entity span.
        Contoh:
        B-PERSON I-PERSON -> entity PERSON
        """

        entities = []

        current_label = None
        current_start = None
        current_end = None

        for index, tag in enumerate(tags):
            if index >= len(token_spans):
                break

            _, start, end = token_spans[index]

            # Tag O atau PAD berarti bukan entity
            if tag in ["O", "[PAD]"]:
                if current_label is not None:
                    entities.append({
                        "text": text[current_start:current_end],
                        "label": current_label,
                        "start": current_start,
                        "end": current_end,
                    })

                    current_label = None
                    current_start = None
                    current_end = None

                continue

            # B- berarti mulai entity baru
            if tag.startswith("B-"):
                if current_label is not None:
                    entities.append({
                        "text": text[current_start:current_end],
                        "label": current_label,
                        "start": current_start,
                        "end": current_end,
                    })

                current_label = tag[2:]
                current_start = start
                current_end = end

            # I- berarti melanjutkan entity sebelumnya
            elif tag.startswith("I-"):
                label = tag[2:]

                if current_label == label:
                    current_end = end
                else:
                    # Jika I- tidak sesuai, tetap dibuat entity baru agar tidak crash
                    if current_label is not None:
                        entities.append({
                            "text": text[current_start:current_end],
                            "label": current_label,
                            "start": current_start,
                            "end": current_end,
                        })

                    current_label = label
                    current_start = start
                    current_end = end

        # Tutup entity terakhir jika masih ada
        if current_label is not None:
            entities.append({
                "text": text[current_start:current_end],
                "label": current_label,
                "start": current_start,
                "end": current_end,
            })

        return entities

    def predict_entities(self, text):
        """
        Fungsi utama untuk prediksi entity dari raw text.
        """

        # Ubah text menjadi input numerik
        model_inputs, token_spans = self._vectorize_text(text)

        # Decode tag dari model CRF
        pred_ids = self.model.decode(model_inputs).numpy()[0]

        # Ubah tag id menjadi nama tag BIO
        tags = [
            self.idx2tag.get(int(tag_id), "[UNK]")
            for tag_id in pred_ids[:len(token_spans)]
        ]

        # Ubah BIO tag menjadi entity
        entities = self._bio_to_entities(text, token_spans, tags)

        return {
            "text": text,
            "tokens": [token for token, _, _ in token_spans],
            "tags": tags,
            "entities": entities,
        }