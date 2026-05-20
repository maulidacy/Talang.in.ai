import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Embedding, Conv1D, GlobalMaxPooling1D, Dropout, Bidirectional, LSTM, Dense, BatchNormalization, ReLU


class CharCNNEmbedding(Layer):
    """
    the char cnn is used to answer "what does this token look like at the character level?"
    this is crucial for prices like Rp133k, 335k-an, cenggo, and etc...
    those price patterns above are OOV at word level but has learnable character patterns

    design:
       input:  (batch, max_len, max_word_len)  — char indices per token
       output: (batch, max_len, CHAR_CNN_DIM) — one vector per token
       
    we reshape to (batch*max_len, max_word_len) to run the CNN over all tokens
    in parallel, then reshape back. this is the standard trick.

    two conv blocks with different kernel sizes to capture trigram and 5-gram patterns.
    global max pool collapses the char dimension → fixed-size vector regardless of word length.
    """
    
    def  __init__(self, char_vocab, char_embed_dim, char_cnn_dim, cnn_filters, kernel_sizes, dropout, **kwargs):
        super().__init__(**kwargs)
        # char_embed_dim: dimension of each character's embedding
        # cnn_filters: list of filter counts, one per conv block e.g. [64, 128]
        # kernel_sizes: list of kernel sizes matching cnn_filters e.g. [3, 5]
        
        self.char_embedding = Embedding(input_dim=char_vocab,
                                        output_dim=char_embed_dim,
                                        # mask_zero=True, # zero-index is PAD, masking propagates
                                        name='char_embed')
        
        # prallel conv blocks
        self.conv_blocks = []
        for filters, ksize in zip(cnn_filters, kernel_sizes):
            block = Sequential([Conv1D(filters, ksize, padding='same', use_bias=False),
                                BatchNormalization(),
                                ReLU()], name=f'conv_{ksize}')
            self.conv_blocks.append(block)
        
        # global max pool per conv block then concat -> CHAR_CNN_DIM total
        self.pool = GlobalMaxPooling1D()
        self.dropout = Dropout(dropout)
        
        # project concatenated conv outputs to CHAR_CNN_DIM
        total_filters = sum(cnn_filters)
        self.proj = Dense(char_cnn_dim, use_bias=False, name='char_proj')
    
    def call(self, char_inputs, training=False):
        # char inputs (batch, seq_len, word_len)
        batch = tf.shape(char_inputs)[0]
        seq = tf.shape(char_inputs)[1]
        
        word_len = tf.shape(char_inputs)[2]
        
        # flatten sequences so CNN sees (batch*seq_len, word_len)
        flat = tf.reshape(char_inputs, (-1, word_len)) # (batch * seq_len, word_len)
        x = self.char_embedding(flat) # (batch * seq_len, word_len, char_embed_dim)
        
        # conv and pool
        pooled = []
        for conv in self.conv_blocks:
            h = conv(x, training=training) # (batch * seq_len, word_len, filters)
            h = self.pool(h) # (batch * seq_len, filters)
            pooled.append(h)
        
        # concatenate all conv outputs along filter axis
        x = tf.concat(pooled, axis=-1) # (batch_size * seq_len, sum_filters)
        x = self.dropout(x, training=training)
        x = self.proj(x) # (batch_size * seq_len, char_cnn_dim)
        
        # restore sequence shape
        x = tf.reshape(x, (batch, seq, self.proj.units)) # (batch_size, seq_len, char_cnn_dim)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.proj.units)
    

class WordEmbedding(Layer):
    """
    word embeddings modes controlled by trainable:
       trainable=True  -> learned from scratch
       trainable=False -> frozen pre-trained weights

    we support loading pre-trained weights via load_pretrained() below.
    mask_zero=True propagates padding masks to the bilstm
    """
    def __init__(self, vocab_size, embed_dim, dropout, trainable_embed=True, **kwargs):
        super().__init__(**kwargs)
        self.embedding = Embedding(input_dim=vocab_size,
                                   output_dim=embed_dim,
                                   mask_zero=False,
                                   trainable=trainable_embed,
                                   name='word_embed')
        self.dropout = Dropout(dropout)
    
    def call(self, word_inputs, training=False):
        # word_inputs: (batch, seq_len)  → int indices
        x = self.embedding(word_inputs)      # (batch, seq_len, embed_dim)
        return self.dropout(x, training=training)

    def load_pretrained(self, embedding_matrix):
        # embedding_matrix = np array of shape (vocab_size, embed_dim)
        self.embedding.set_weights([embedding_matrix])
        print(f"loaded pretrained weights: {embedding_matrix.shape}")
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.embedding.output_dim)


class BiLSTMEncoder(Layer):
    """
    the bilstm sees the concatenated [word_embed; char_cnn_vec] at each timestep.
    return_sequences=True is mandatory — we need a tag at every position, not just the last.
    we stack two bilstm layers, the first returns sequences into the second.

    note: tf.keras Bidirectional wraps the LSTM and concatenates fwd+bwd outputs,
    so a LSTM(units=256) wrapped in Bidirectional -> output dim is 512.
    """
    
    def __init__(self, units, dropout, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        
        # first lstm layer returns sequences into the second
        self.bilstm1 = Bidirectional(LSTM(units, return_sequences=True, recurrent_dropout=dropout),
                                     name='bilstm_1')
        
        # second lstm layer
        self.bilstm2 = Bidirectional(LSTM(units, return_sequences=True, recurrent_dropout=dropout),
                                     name='bilstm_2')
        
        self.dropout = Dropout(dropout)
        
    def call(self, x, mask=None, training=False):
        # x: (batch, seq_len, word_embed_dim + char_cnn_dim)
        x = self.bilstm1(x, mask=mask, training=training)  # (b, s, units*2)
        x = self.dropout(x, training=training)
        x = self.bilstm2(x, mask=mask, training=training)  # (b, s, units*2)
        return x
    
    def compute_output_shape(self, input_shape):
        return self.bilstm2.compute_output_shape(input_shape)
    

class CRFLayer(Layer):
    """
    a linear-chain CRF adds a transition matrix on top of the bilstm emissions.
    the layer holds one learnable parameter: transition_params (num_tags, num_tags).
    transition_params[i, j] = score of transitioning from tag i to tag j.

    at training time we need:
       1. score of the true tag path      (sum of emissions + transitions along ground truth)
       2. log partition function Z        (sum over ALL possible paths, computed via forward alg)
    nll = log(Z) - score               (we minimize this)

    at inference time we run viterbi decoding to find the highest-scoring tag sequence.
    """
    
    def __init__(self, num_tags, **kwargs):
        super().__init__(**kwargs)
        self.num_tags = num_tags
        
        # transition_params[i, j]: score of going from tag i → tag j
        # initialized to zeros; the model learns which transitions are (im)possible
        self.transition_params = self.add_weight(name='transition_params',
                                                 shape=(self.num_tags, self.num_tags),
                                                 initializer='zeros',
                                                 trainable=True)
        self.built = True
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, emissions, mask=None):
        # at inference we run viterbi and return the best tag sequence
        # at training the loss function calls log_likelihood directly
        # emissions shape: (batch, seq_len, num_tags)
        return self.viterbi_decode(emissions, mask)
    
    def log_likelihood(self, emissions, tag_indices, mask):
        """
        compute the CRF log-likelihood for a batch.
        emissions:   (batch, seq_len, num_tags)
        tag_indices: (batch, seq_len)
        mask:        (batch, seq_len)

        returns scalar: mean negative log-likelihood over the batch.
        """
        batch_size = tf.shape(emissions)[0]
        seq_len = tf.shape(emissions)[1]
        
        # score of the true path
        true_score = self._score_sequence(emissions, tag_indices, mask)
        
        # log partition function
        log_Z = self._forward_algorithm(emissions, mask)
        
        # nll per sample
        nll = log_Z - true_score
        return tf.reduce_mean(nll) # mean over batch
    
    def _score_sequence(self, emissions, tag_indices, mask):
        """sum emission scores + transition scores along the true path."""
        batch_size = tf.shape(emissions)[0]
        seq_len    = tf.shape(emissions)[1]
        
        # emission scores: (batch, seq_len)
        batch_idx = tf.tile(tf.expand_dims(tf.range(batch_size), 1), [1, seq_len])
        seq_idx = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
        indices = tf.stack([batch_idx, seq_idx, tag_indices], axis=2)
        
        emit_scores = tf.gather_nd(emissions, indices) # (b, s)
        
        # transition scores: transition_params[y_{t-1}, y_t] for t in 1..T
        # slice off first and last to align consecutive pairs
        trans_scores = tf.gather_nd(self.transition_params,
                                    tf.stack([tag_indices[:, :-1], tag_indices[:, 1:]], axis=2)) # (b, s-1)
        
        # apply mask
        mask_f = tf.cast(mask, tf.float32)
        total = tf.reduce_sum(emit_scores * mask_f, axis=1)
        total += tf.reduce_sum(trans_scores * mask_f[:, 1:], axis=1)
        return total # (batch,)
    
    def _forward_algorithm(self, emissions, mask):
        """
        log-space forward algorithm to compute log Z (partition function).
        uses log-sum-exp for numerical stability.
        """
        seq_len = tf.shape(emissions)[1]
        mask_f  = tf.cast(mask, tf.float32)

        # initialize: alpha[0] = emissions at t=0
        alphas = emissions[:, 0, :] # (batch, num_tags)

        for t in tf.range(1, seq_len):
            # expand for broadcasting: (batch, num_tags, 1) + (num_tags, num_tags)
            # transition_scores[i, j] = alpha[i] + transition[i, j] + emission[j]
            emit_t  = emissions[:, t, :] # (batch, num_tags)
            trans_t = tf.expand_dims(alphas, 2) + self.transition_params  # (batch, num_tags, num_tags)
            # log-sum-exp over previous tags
            new_alphas = tf.reduce_logsumexp(trans_t, axis=1) + emit_t # (batch, num_tags)

            # only update alphas for non-padding positions
            mask_t = tf.expand_dims(mask_f[:, t], 1) # (batch, 1)
            alphas = new_alphas * mask_t + alphas * (1 - mask_t)

        # final log Z: log-sum-exp over last alphas
        return tf.reduce_logsumexp(alphas, axis=1) # (batch,)
    
    def viterbi_decode(self, emissions, mask):
        """
        viterbi decoding: find highest-scoring tag sequence.
        returns (tag_ids, viterbi_score) both shape (batch,) and (batch, seq_len).
        """
        seq_len = tf.shape(emissions)[1]
        mask_f  = tf.cast(mask, tf.float32)

        viterbi  = emissions[:, 0, :] # (batch, num_tags)
        backpointers = []

        for t in tf.range(1, seq_len):
            emit_t = emissions[:, t, :] # (batch, num_tags)
            # scores for all prev_tag → cur_tag transitions
            trans_scores = tf.expand_dims(viterbi, 2) + self.transition_params  # (b, T, T)
            best_prev    = tf.argmax(trans_scores, axis=1) # (b, num_tags)
            best_scores  = tf.reduce_max(trans_scores, axis=1) + emit_t

            mask_t = tf.expand_dims(mask_f[:, t], 1)
            viterbi = best_scores * mask_t + viterbi * (1 - mask_t)
            backpointers.append(best_prev)

        # backtrace
        best_last = tf.argmax(viterbi, axis=1) # (batch,)
        best_path = [best_last]

        for bp in reversed(backpointers):
            # gather best previous tag at each position
            batch_size = tf.shape(emissions)[0]
            idx = tf.stack([tf.range(tf.cast(batch_size, tf.int64), dtype=tf.int64), best_path[-1]], axis=1)
            best_last = tf.gather_nd(tf.cast(bp, tf.int64), idx)
            best_path.append(best_last)

        best_path = list(reversed(best_path))
        best_path = tf.stack(best_path, axis=1) # (batch, seq_len)
        return best_path
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])