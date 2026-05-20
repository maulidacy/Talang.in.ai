import pytest
import json
import os
import sys
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess import DatasetBuilder
from models.ner_model import BillNERModel

@pytest.fixture
def dummy_dataset_path(tmp_path):
    """creates a tiny temporary JSON dataset for testing."""
    data = {
        "data": [
            {
                "text": "1 Nasi Goreng Rp 15.000",
                "entities": [
                    {"start": 0, "end": 1, "label": "QTY"},
                    {"start": 2, "end": 13, "label": "ITEM"},
                    {"start": 14, "end": 23, "label": "PRICE"}
                ]
            },
            {
                "text": "Es Teh Manis 5k",
                "entities": [
                    {"start": 0, "end": 12, "label": "ITEM"},
                    {"start": 13, "end": 15, "label": "PRICE"}
                ]
            }
        ]
    }
    file_path = tmp_path / "dummy_dataset.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return str(file_path)

def test_model_forward_pass_and_gradients(dummy_dataset_path):
    """E2E Test: Data loading -> Batching -> Forward Pass -> Gradients"""
    
    # build dataset
    builder = DatasetBuilder(max_len=20, max_word_len=10, train_pct=1.0, val_pct=0.0, verbose=False)
    train_ds, _, _ = builder.build_from_file(dummy_dataset_path, batch_size=2)
    
    vocab_size = len(builder.word2idx)
    char_vocab = len(builder.char2idx)
    num_tags = len(builder.tag2idx)
    
    # init model
    model = BillNERModel(
        vocab_size=vocab_size,
        char_vocab=char_vocab,
        num_tags=num_tags,
        word_embed_dim=16,
        char_embed_dim=8,
        char_cnn_dim=16,
        bilstm_units=32,
        dropout=0.1
    )
    
    # take 1 batch
    batch_inputs, batch_tags = next(iter(train_ds))
    
    # assert shapes
    emissions, mask = model(batch_inputs, training=True)
    assert emissions.shape == (2, 20, num_tags), "Emissions shape is incorrect"
    assert mask.shape == (2, 20), "Mask shape is incorrect"
    
    # assert gradients
    with tf.GradientTape() as tape:
        e, m = model(batch_inputs, training=True)
        loss = model.crf.log_likelihood(e, tf.cast(batch_tags, tf.int32), m)
        
    grads = tape.gradient(loss, model.trainable_variables)
    
    # find any variables that didn't receive a gradient
    none_grads = [var.name for var, grad in zip(model.trainable_variables, grads) if grad is None]
    assert len(none_grads) == 0, f"Disconnected graph! Variables with None gradients: {none_grads}"
    
    # test viterbi shape inference
    preds = model.decode(batch_inputs)
    assert preds.shape == (2, 20), "Viterbi output shape is incorrect"