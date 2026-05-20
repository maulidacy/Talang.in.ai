import json
from pathlib import Path

import tensorflow as tf

# Import model utama
from src.models.ner_model import BillNERModel

# Import semua custom layer yang dipakai oleh model
from src.models.layers import (
    CharCNNEmbedding,
    WordEmbedding,
    BiLSTMEncoder,
    CRFLayer,
)


# BASE_DIR diarahkan ke folder utama ai-service
BASE_DIR = Path(__file__).resolve().parents[2]

# Lokasi file hasil training
VOCAB_DIR = BASE_DIR / "outputs" / "vocabs"
CONFIG_PATH = BASE_DIR / "outputs" / "training_config.json"
WEIGHTS_PATH = BASE_DIR / "models" / "best_ner_model.weights.h5"

# Lokasi output file .keras
KERAS_MODEL_PATH = BASE_DIR / "models" / "bill_ner_model.keras"


def load_json(path):
    """
    Fungsi untuk membaca file JSON.
    Dipakai untuk membaca vocab dan konfigurasi training.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def add_serialization_support():
    """
    Bagian ini menambahkan get_config secara sementara saat export.

    Kenapa perlu?
    Karena model kita memakai custom subclass dan custom layer.
    Agar bisa disimpan ke format .keras, Keras perlu tahu konfigurasi
    constructor dari setiap class custom.
    """

    def bill_ner_get_config(self):
        # Config ini dipakai Keras untuk membangun ulang BillNERModel
        config = tf.keras.Model.get_config(self)

        config.update({
            "vocab_size": self.vocab_size,
            "char_vocab": self.char_vocab,
            "num_tags": self.num_tags,
            "word_embed_dim": self.word_embed_dim,
            "char_embed_dim": self.char_embed_dim,
            "char_cnn_dim": self.char_cnn_dim,
            "bilstm_units": self.bilstm_units,
            "dropout": self.dropout_rate,
        })

        return config

    def word_embedding_get_config(self):
        # Config untuk membangun ulang WordEmbedding
        config = tf.keras.layers.Layer.get_config(self)

        config.update({
            "vocab_size": self.embedding.input_dim,
            "embed_dim": self.embedding.output_dim,
            "dropout": float(self.dropout.rate),
            "trainable_embed": self.embedding.trainable,
        })

        return config

    def char_cnn_get_config(self):
        # Ambil jumlah filter dan kernel size dari setiap conv block
        cnn_filters = []
        kernel_sizes = []

        for block in self.conv_blocks:
            conv_layer = block.layers[0]
            cnn_filters.append(conv_layer.filters)
            kernel_sizes.append(conv_layer.kernel_size[0])

        # Config untuk membangun ulang CharCNNEmbedding
        config = tf.keras.layers.Layer.get_config(self)

        config.update({
            "char_vocab": self.char_embedding.input_dim,
            "char_embed_dim": self.char_embedding.output_dim,
            "char_cnn_dim": self.proj.units,
            "cnn_filters": cnn_filters,
            "kernel_sizes": kernel_sizes,
            "dropout": float(self.dropout.rate),
        })

        return config

    def bilstm_get_config(self):
        # Config untuk membangun ulang BiLSTMEncoder
        config = tf.keras.layers.Layer.get_config(self)

        config.update({
            "units": self.bilstm1.forward_layer.units,
            "dropout": float(self.dropout.rate),
        })

        return config

    def crf_get_config(self):
        # Config untuk membangun ulang CRFLayer
        config = tf.keras.layers.Layer.get_config(self)

        config.update({
            "num_tags": self.num_tags,
        })

        return config

    # Pasang get_config ke setiap class custom
    BillNERModel.get_config = bill_ner_get_config
    WordEmbedding.get_config = word_embedding_get_config
    CharCNNEmbedding.get_config = char_cnn_get_config
    BiLSTMEncoder.get_config = bilstm_get_config
    CRFLayer.get_config = crf_get_config


def main():
    """
    Script ini digunakan untuk export model NER ke format .keras.

    Alur:
    1. Load vocab dan config training
    2. Build ulang arsitektur BillNERModel
    3. Load weights hasil training
    4. Simpan model ke format .keras
    5. Load ulang model .keras untuk verifikasi
    """

    # Tambahkan dukungan serialisasi untuk custom model dan custom layer
    add_serialization_support()

    # Load vocab hasil training
    word2idx = load_json(VOCAB_DIR / "word2idx.json")
    char2idx = load_json(VOCAB_DIR / "char2idx.json")
    tag2idx = load_json(VOCAB_DIR / "tag2idx.json")

    # Load config training
    config = load_json(CONFIG_PATH)

    # Ambil max_len dan max_word_len dari config.
    # Kalau tidak ada, gunakan default yang aman.
    max_len = config.get("max_len", 100)
    max_word_len = config.get("max_word_len", 15)

    # Parameter model harus sama dengan saat training
    vocab_size = len(word2idx)
    char_vocab = len(char2idx)
    num_tags = len(tag2idx)

    # Build ulang model dengan arsitektur yang sama
    model = BillNERModel(
        vocab_size=vocab_size,
        char_vocab=char_vocab,
        num_tags=num_tags,
    )

    # Simpan parameter ke object model agar bisa dibaca oleh get_config
    model.vocab_size = vocab_size
    model.char_vocab = char_vocab
    model.num_tags = num_tags
    model.word_embed_dim = 128
    model.char_embed_dim = 32
    model.char_cnn_dim = 128
    model.bilstm_units = 256
    model.dropout_rate = 0.3

    # Dummy input dipakai agar semua layer ter-build sebelum load weights
    dummy_inputs = {
        "word_inputs": tf.zeros((1, max_len), dtype=tf.int32),
        "char_inputs": tf.zeros((1, max_len, max_word_len), dtype=tf.int32),
    }

    # Panggil model sekali agar layer dan weight shape terbentuk
    _ = model(dummy_inputs, training=False)

    # Load weights terbaik hasil training
    model.load_weights(str(WEIGHTS_PATH))

    # Pastikan folder models ada
    KERAS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Simpan model ke format .keras
    model.save(str(KERAS_MODEL_PATH))

    print("Model berhasil diexport ke format .keras:")
    print(KERAS_MODEL_PATH)

    # Custom objects diperlukan saat load model karena model memakai class custom
    custom_objects = {
        "BillNERModel": BillNERModel,
        "WordEmbedding": WordEmbedding,
        "CharCNNEmbedding": CharCNNEmbedding,
        "BiLSTMEncoder": BiLSTMEncoder,
        "CRFLayer": CRFLayer,
    }

    # Register custom objects agar Keras mengenali class custom
    tf.keras.utils.get_custom_objects().update(custom_objects)

    # Load ulang model .keras untuk memastikan file benar-benar bisa dipakai
    loaded_model = tf.keras.models.load_model(
        str(KERAS_MODEL_PATH),
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,
    )

    # Tes forward pass setelah model diload ulang
    emissions, mask = loaded_model(dummy_inputs, training=False)

    print("Verifikasi load model .keras berhasil.")
    print("Emission shape:", emissions.shape)
    print("Mask shape:", mask.shape)


if __name__ == "__main__":
    main()