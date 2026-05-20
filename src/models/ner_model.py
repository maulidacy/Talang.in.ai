import tensorflow as tf
from tensorflow.keras import layers, Model
from .layers import CharCNNEmbedding, WordEmbedding, CRFLayer, BiLSTMEncoder

class BillNERModel(Model):
    def __init__(self, vocab_size, char_vocab, num_tags,
                 word_embed_dim=128, char_embed_dim=32, char_cnn_dim=128,
                 bilstm_units=256, dropout=0.3, **kwargs):
        super().__init__(**kwargs)

        self.word_embedding = WordEmbedding(vocab_size, word_embed_dim, dropout)

        self.char_cnn = CharCNNEmbedding(
            char_vocab=char_vocab,
            char_embed_dim=char_embed_dim,
            char_cnn_dim=char_cnn_dim,
            cnn_filters=[64, 128],
            kernel_sizes=[3, 5],
            dropout=dropout,
        )

        self.bilstm = BiLSTMEncoder(bilstm_units, dropout)

        # project bilstm output to tag space before CRF
        self.emission_proj = layers.Dense(num_tags, name='emission_proj')

        self.crf = CRFLayer(num_tags)

    def call(self, inputs, training=False):
        word_in = inputs['word_inputs']   # (batch, seq_len)
        char_in = inputs['char_inputs']   # (batch, seq_len, word_len)

        # build mask from word inputs: True where word_idx > 0 (non-padding)
        mask = tf.cast(word_in > 0, tf.bool)   # (batch, seq_len)

        # embed
        word_vec = self.word_embedding(word_in, training=training) # (b, s, word_embed)
        char_vec = self.char_cnn(char_in, training=training) # (b, s, char_cnn_dim)

        # concat and encode
        x = tf.concat([word_vec, char_vec], axis=-1) # (b, s, word+char)
        x = self.bilstm(x, mask=mask, training=training) # (b, s, bilstm*2)

        # project to tag emissions
        emissions = self.emission_proj(x) # (b, s, num_tags)
        return emissions, mask

    def decode(self, inputs):
        """inference: returns viterbi tag sequence"""
        emissions, mask = self(inputs, training=False)
        return self.crf.viterbi_decode(emissions, mask)