"""Dicoding image classification submission using an open-source image dataset."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


SEED = 42
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_RATIO = 0.2
CLASS_NAMES = [
    "t-shirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]
DATASET_NAME = "Fashion-MNIST"
DATASET_SOURCE = "https://github.com/zalandoresearch/fashion-mnist"

ROOT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT_DIR
DATASET_DIR = ROOT_DIR / "dataset" / "fashion_mnist"
SPLIT_DIR = ROOT_DIR / "dataset_split" / "fashion_mnist"
SAVED_MODEL_DIR = SUBMISSION_DIR / "saved_model"
TFLITE_DIR = SUBMISSION_DIR / "tflite"
TFJS_DIR = SUBMISSION_DIR / "tfjs_model"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_dataset() -> None:
    """Download Fashion-MNIST and materialize it as class-based PNG folders."""

    expected_count = 70_000
    existing_count = sum(1 for _ in DATASET_DIR.glob("*/*.png")) if DATASET_DIR.exists() else 0
    if existing_count == expected_count:
        print(f"{DATASET_NAME} dataset already exists: {existing_count} images")
        return

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    for source_name, images, labels in [
        ("original_train", train_images, train_labels),
        ("original_test", test_images, test_labels),
    ]:
        for index, (image_array, label_index) in enumerate(zip(images, labels)):
            class_name = CLASS_NAMES[int(label_index)]
            class_dir = DATASET_DIR / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            image = Image.fromarray(image_array.astype(np.uint8), mode="L")
            image.save(class_dir / f"{source_name}_{index:05d}.png", optimize=True)

    print(f"Dataset source: {DATASET_NAME} ({DATASET_SOURCE})")
    print(f"Created dataset: {expected_count} images in {DATASET_DIR}")


def print_images_resolution(directory: Path) -> None:
    total_images = 0
    for class_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        unique_sizes = set()
        image_files = sorted(class_dir.glob("*.png"))
        total_images += len(image_files)
        for image_file in image_files:
            with Image.open(image_file) as image:
                unique_sizes.add(image.size)

        print(f"{class_dir.name}: {len(image_files)}")
        for size in sorted(unique_sizes)[:10]:
            print(f"- {size}")
        if len(unique_sizes) > 10:
            print(f"- ... {len(unique_sizes) - 10} ukuran lain")
        print("---------------")
    print(f"\nTotal: {total_images}")


def split_dataset() -> None:
    if SPLIT_DIR.exists():
        split_count = sum(1 for _ in SPLIT_DIR.glob("*/*/*.png"))
        if split_count == 70_000:
            print(f"Split dataset already exists: {split_count} images")
            return
        shutil.rmtree(SPLIT_DIR)

    rng = random.Random(SEED)

    for class_name in CLASS_NAMES:
        train_validation_files = sorted((DATASET_DIR / class_name).glob("original_train_*.png"))
        test_files = sorted((DATASET_DIR / class_name).glob("original_test_*.png"))
        rng.shuffle(train_validation_files)
        validation_size = int(len(train_validation_files) * VALIDATION_RATIO)
        split_map = {
            "train": train_validation_files[validation_size:],
            "validation": train_validation_files[:validation_size],
            "test": test_files,
        }

        for split_name, split_files in split_map.items():
            target_dir = SPLIT_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_file in split_files:
                shutil.copy2(source_file, target_dir / source_file.name)

    for split_name in ["train", "validation", "test"]:
        count = sum(1 for _ in (SPLIT_DIR / split_name).glob("*/*.png"))
        print(f"{split_name}: {count} images")


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    common_kwargs = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "label_mode": "categorical",
        "color_mode": "grayscale",
        "class_names": CLASS_NAMES,
        "shuffle": False,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        SPLIT_DIR / "train",
        shuffle=True,
        seed=SEED,
        **{key: value for key, value in common_kwargs.items() if key != "shuffle"},
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(SPLIT_DIR / "validation", **common_kwargs)
    test_ds = tf.keras.utils.image_dataset_from_directory(SPLIT_DIR / "test", **common_kwargs)

    autotune = tf.data.AUTOTUNE
    return (
        train_ds.cache().shuffle(1000, seed=SEED).prefetch(autotune),
        validation_ds.cache().prefetch(autotune),
        test_ds.cache().prefetch(autotune),
    )


def build_inference_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 1)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="fashion_mnist_inference_cnn",
    )
    return model


def build_training_model(num_classes: int) -> tf.keras.Model:
    classifier = build_inference_model(num_classes)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 1)),
            classifier,
        ],
        name="fashion_mnist_training_cnn",
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_model(num_classes: int) -> tf.keras.Model:
    return build_training_model(num_classes)


def plot_history(history: tf.keras.callbacks.History) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Training Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig(SUBMISSION_DIR / "training_plot.png", dpi=150)
    plt.show()


def get_inference_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if layer.name == "fashion_mnist_inference_cnn":
            return layer
    return model


def export_models(model: tf.keras.Model) -> None:
    inference_model = get_inference_model(model)

    for directory in [SAVED_MODEL_DIR, TFLITE_DIR, TFJS_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    inference_model.export(SAVED_MODEL_DIR)
    export_from_saved_model()


def export_from_saved_model() -> None:
    for directory in [TFLITE_DIR, TFJS_DIR]:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(SAVED_MODEL_DIR))
    tflite_model = converter.convert()
    (TFLITE_DIR / "model.tflite").write_bytes(tflite_model)
    (TFLITE_DIR / "label.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            str(SAVED_MODEL_DIR),
            str(TFJS_DIR),
        ],
        check=True,
    )

    metadata = {
        "dataset": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "train_images": 48_000,
        "validation_images": 12_000,
        "test_images": 10_000,
    }
    (SUBMISSION_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_tflite_inference() -> None:
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_DIR / "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for true_label in CLASS_NAMES:
        for sample_path in sorted((SPLIT_DIR / "test" / true_label).glob("*.png"))[:100]:
            image = Image.open(sample_path).convert("L").resize(IMAGE_SIZE)
            input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=(0, -1))

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]["index"])[0]
            predicted_index = int(np.argmax(predictions))
            predicted_label = CLASS_NAMES[predicted_index]

            if predicted_label == true_label:
                print(f"Sample image: {sample_path}")
                print(f"True label: {true_label}")
                print(f"Predicted label: {predicted_label}")
                print(f"Confidence: {predictions[predicted_index]:.4f}")
                print(f"All probabilities: {dict(zip(CLASS_NAMES, predictions.round(4).tolist()))}")
                return

    raise RuntimeError("No correctly classified TFLite sample found in the checked test images.")


def main() -> None:
    set_seed()
    generate_dataset()
    print_images_resolution(DATASET_DIR)
    split_dataset()
    train_ds, validation_ds, test_ds = load_datasets()

    model = build_model(len(CLASS_NAMES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(SUBMISSION_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(train_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)
    plot_history(history)

    train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy: {test_accuracy:.4f}")
    print(f"Training loss: {train_loss:.4f}")
    print(f"Testing loss: {test_loss:.4f}")

    export_models(model)
    run_tflite_inference()


if __name__ == "__main__":
    main()
