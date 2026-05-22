import os
import json
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf

from src.preprocessing.preprocess import DatasetBuilder
from src.models.ner_model import BillNERModel


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_idx2tag(tag2idx):
    return {idx: tag for tag, idx in tag2idx.items()}


def ids_to_tags(ids, idx2tag):
    return [idx2tag.get(int(idx), "[UNK]") for idx in ids]


def extract_entities_from_tags(tags):
    entities = []
    current_label = None
    start = None

    for index, tag in enumerate(tags):
        if tag in ["[PAD]", "O"]:
            if current_label is not None:
                entities.append((current_label, start, index - 1))
                current_label = None
                start = None
            continue

        if tag.startswith("B-"):
            if current_label is not None:
                entities.append((current_label, start, index - 1))

            current_label = tag[2:]
            start = index

        elif tag.startswith("I-"):
            label = tag[2:]

            if current_label is None:
                current_label = label
                start = index
            elif current_label != label:
                entities.append((current_label, start, index - 1))
                current_label = label
                start = index

    if current_label is not None:
        entities.append((current_label, start, len(tags) - 1))

    return set(entities)


def compute_token_accuracy(y_true, y_pred, mask):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    mask = mask.numpy().astype(bool)

    correct = (y_true == y_pred) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    return correct.sum() / total


def compute_entity_f1(y_true, y_pred, mask, idx2tag):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    mask = mask.numpy().astype(bool)

    total_true = 0
    total_pred = 0
    total_correct = 0

    for true_seq, pred_seq, mask_seq in zip(y_true, y_pred, mask):
        valid_len = int(mask_seq.sum())

        true_tags = ids_to_tags(true_seq[:valid_len], idx2tag)
        pred_tags = ids_to_tags(pred_seq[:valid_len], idx2tag)

        true_entities = extract_entities_from_tags(true_tags)
        pred_entities = extract_entities_from_tags(pred_tags)

        total_true += len(true_entities)
        total_pred += len(pred_entities)
        total_correct += len(true_entities & pred_entities)

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_true if total_true > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_model(model, dataset, idx2tag):
    losses = []
    token_accs = []

    all_precision = []
    all_recall = []
    all_f1 = []

    for batch_inputs, batch_tags in dataset:
        emissions, mask = model(batch_inputs, training=False)
        loss = model.crf.log_likelihood(
            emissions,
            tf.cast(batch_tags, tf.int32),
            mask,
        )

        preds = model.decode(batch_inputs)

        token_acc = compute_token_accuracy(batch_tags, preds, mask)
        precision, recall, f1 = compute_entity_f1(batch_tags, preds, mask, idx2tag)

        losses.append(float(loss.numpy()))
        token_accs.append(token_acc)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "token_accuracy": float(np.mean(token_accs)) if token_accs else 0.0,
        "precision": float(np.mean(all_precision)) if all_precision else 0.0,
        "recall": float(np.mean(all_recall)) if all_recall else 0.0,
        "entity_f1": float(np.mean(all_f1)) if all_f1 else 0.0,
    }


class TrainingLoggerCallback:
    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs):
        print(
            f"Epoch {epoch} | "
            f"train_loss={logs['train_loss']:.4f} | "
            f"val_loss={logs['val_loss']:.4f} | "
            f"token_acc={logs['val_token_accuracy']:.4f} | "
            f"precision={logs['val_precision']:.4f} | "
            f"recall={logs['val_recall']:.4f} | "
            f"f1={logs['val_entity_f1']:.4f}"
        )

        with self.writer.as_default():
            for key, value in logs.items():
                tf.summary.scalar(key, value, step=epoch)

        self.writer.flush()


def save_training_config(save_dir, builder, args):
    config = {
        "max_len": args.max_len,
        "max_word_len": args.max_word_len,
        "word_embed_dim": args.word_embed_dim,
        "char_embed_dim": args.char_embed_dim,
        "char_cnn_dim": args.char_cnn_dim,
        "bilstm_units": args.bilstm_units,
        "dropout": args.dropout,
        "vocab_size": len(builder.word2idx),
        "char_vocab": len(builder.char2idx),
        "num_tags": len(builder.tag2idx),
    }

    with open(os.path.join(save_dir, "training_config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)


def train(args):
    ensure_dir(args.output_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.log_dir)

    print("Building dataset...")

    builder = DatasetBuilder(
        max_len=args.max_len,
        max_word_len=args.max_word_len,
        min_word_freq=args.min_word_freq,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        seed=args.seed,
        verbose=True,
    )

    train_ds, val_ds, test_ds = builder.build_from_file(
        args.data_path,
        batch_size=args.batch_size,
    )

    vocab_dir = os.path.join(args.output_dir, "vocabs")
    builder.save_vocabs(vocab_dir)
    save_training_config(args.output_dir, builder, args)

    idx2tag = build_idx2tag(builder.tag2idx)

    print("Initializing model...")

    model = BillNERModel(
        vocab_size=len(builder.word2idx),
        char_vocab=len(builder.char2idx),
        num_tags=len(builder.tag2idx),
        word_embed_dim=args.word_embed_dim,
        char_embed_dim=args.char_embed_dim,
        char_cnn_dim=args.char_cnn_dim,
        bilstm_units=args.bilstm_units,
        dropout=args.dropout,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    sample_inputs, _ = next(iter(train_ds))
    _ = model(sample_inputs, training=False)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = os.path.join(args.log_dir, timestamp)
    logger = TrainingLoggerCallback(tensorboard_dir)

    best_f1 = -1.0
    best_weights_path = os.path.join(args.model_dir, "best_ner_model.weights.h5")

    print("Start training...")

    for epoch in range(1, args.epochs + 1):
        train_losses = []

        for batch_inputs, batch_tags in train_ds:
            with tf.GradientTape() as tape:
                emissions, mask = model(batch_inputs, training=True)
                loss = model.crf.log_likelihood(
                    emissions,
                    tf.cast(batch_tags, tf.int32),
                    mask,
                )

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_losses.append(float(loss.numpy()))

        val_metrics = evaluate_model(model, val_ds, idx2tag)

        logs = {
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_metrics["loss"],
            "val_token_accuracy": val_metrics["token_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_entity_f1": val_metrics["entity_f1"],
        }

        logger.on_epoch_end(epoch, logs)

        if val_metrics["entity_f1"] > best_f1:
            best_f1 = val_metrics["entity_f1"]
            model.save_weights(best_weights_path)
            print(f"Saved best model weights to {best_weights_path}")

    print("Evaluating best model on test set...")

    model.load_weights(best_weights_path)
    test_metrics = evaluate_model(model, test_ds, idx2tag)

    with open(os.path.join(args.output_dir, "evaluation_report.json"), "w", encoding="utf-8") as file:
        json.dump(test_metrics, file, ensure_ascii=False, indent=2)

    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))

    saved_model_dir = os.path.join(args.model_dir, "saved_model")

    try:
        tf.saved_model.save(model, saved_model_dir)
        print(f"SavedModel exported to {saved_model_dir}")
    except Exception as error:
        print(f"SavedModel export skipped: {error}")

    print("Training completed.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/dataset_fixed.json")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--log_dir", default="logs")

    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--max_word_len", type=int, default=15)
    parser.add_argument("--min_word_freq", type=int, default=2)

    parser.add_argument("--train_pct", type=float, default=0.8)
    parser.add_argument("--val_pct", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--word_embed_dim", type=int, default=128)
    parser.add_argument("--char_embed_dim", type=int, default=32)
    parser.add_argument("--char_cnn_dim", type=int, default=128)
    parser.add_argument("--bilstm_units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)