"""
Train a custom emotion recognition model from dataset/<label> images.
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_classes, input_shape):
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_labels(class_indices, output_path):
    labels = [None] * len(class_indices)
    for label, idx in class_indices.items():
        labels[idx] = label

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom emotion model")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset folder")
    parser.add_argument("--output", default="models/custom_emotion_model.keras", help="Model output path")
    parser.add_argument("--labels", default="models/custom_emotion_labels.json", help="Labels output path")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = args.dataset
    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Dataset folder not found: {dataset_dir}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    if train_gen.num_classes < 2:
        raise SystemExit("Need at least 2 emotion classes to train.")

    model = build_model(train_gen.num_classes, (args.img_size, args.img_size, 3))

    callbacks_list = [
        callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        callbacks.ModelCheckpoint(args.output, save_best_only=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks_list
    )

    save_labels(train_gen.class_indices, args.labels)
    print(f"Saved model to {args.output}")
    print(f"Saved labels to {args.labels}")


if __name__ == "__main__":
    main()
