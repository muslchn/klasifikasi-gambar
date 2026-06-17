"""Dicoding image classification submission.

The project intentionally uses a generated geometric-shapes dataset so the
submission is reproducible without external downloads.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
SAMPLES_PER_CLASS = 4000
CLASS_NAMES = ["circle", "square", "triangle"]

ROOT_DIR = Path(__file__).resolve().parent
SUBMISSION_DIR = ROOT_DIR
DATASET_DIR = ROOT_DIR / "dataset" / "geometric_shapes"
SPLIT_DIR = ROOT_DIR / "dataset_split" / "geometric_shapes"
SAVED_MODEL_DIR = SUBMISSION_DIR / "saved_model"
TFLITE_DIR = SUBMISSION_DIR / "tflite"
TFJS_DIR = SUBMISSION_DIR / "tfjs_model"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def draw_shape(draw: ImageDraw.ImageDraw, label: str, width: int, height: int, rng: random.Random) -> None:
    margin = rng.randint(10, max(12, min(width, height) // 4))
    x1 = rng.randint(margin, max(margin, width // 3))
    y1 = rng.randint(margin, max(margin, height // 3))
    x2 = rng.randint(max(x1 + 24, (2 * width) // 3), width - margin)
    y2 = rng.randint(max(y1 + 24, (2 * height) // 3), height - margin)
    fill = tuple(rng.randint(40, 235) for _ in range(3))
    outline = tuple(max(0, channel - 35) for channel in fill)

    if label == "circle":
        draw.ellipse((x1, y1, x2, y2), fill=fill, outline=outline, width=3)
    elif label == "square":
        side = min(x2 - x1, y2 - y1)
        draw.rectangle((x1, y1, x1 + side, y1 + side), fill=fill, outline=outline, width=3)
    else:
        points = [(width // 2, y1), (x1, y2), (x2, y2)]
        jittered = [(x + rng.randint(-5, 5), y + rng.randint(-5, 5)) for x, y in points]
        draw.polygon(jittered, fill=fill, outline=outline)


def create_single_image(label: str, index: int) -> Image.Image:
    rng = random.Random(f"{SEED}-{label}-{index}")
    width = rng.randint(96, 192)
    height = rng.randint(96, 192)
    background = tuple(rng.randint(215, 255) for _ in range(3))
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    for _ in range(rng.randint(2, 7)):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        radius = rng.randint(1, 3)
        color = tuple(rng.randint(170, 245) for _ in range(3))
        draw.ellipse((x, y, x + radius, y + radius), fill=color)

    draw_shape(draw, label, width, height, rng)
    return image


def generate_dataset() -> None:
    expected_count = len(CLASS_NAMES) * SAMPLES_PER_CLASS
    existing_count = sum(1 for _ in DATASET_DIR.glob("*/*.png")) if DATASET_DIR.exists() else 0
    if existing_count >= expected_count:
        print(f"Dataset already exists: {existing_count} images")
        return

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for class_name in CLASS_NAMES:
        class_dir = DATASET_DIR / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(SAMPLES_PER_CLASS):
            image = create_single_image(class_name, index)
            image.save(class_dir / f"{class_name}_{index:04d}.png", optimize=True)

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
        if split_count == len(CLASS_NAMES) * SAMPLES_PER_CLASS:
            print(f"Split dataset already exists: {split_count} images")
            return
        shutil.rmtree(SPLIT_DIR)

    split_ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
    rng = random.Random(SEED)

    for class_name in CLASS_NAMES:
        files = sorted((DATASET_DIR / class_name).glob("*.png"))
        rng.shuffle(files)
        train_end = int(len(files) * split_ratios["train"])
        validation_end = train_end + int(len(files) * split_ratios["validation"])
        split_map = {
            "train": files[:train_end],
            "validation": files[train_end:validation_end],
            "test": files[validation_end:],
        }

        for split_name, split_files in split_map.items():
            target_dir = SPLIT_DIR / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_file in split_files:
                shutil.copy2(source_file, target_dir / source_file.name)

    for split_name in split_ratios:
        count = sum(1 for _ in (SPLIT_DIR / split_name).glob("*/*.png"))
        print(f"{split_name}: {count} images")


def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    common_kwargs = {
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "label_mode": "categorical",
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
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
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
        name="geometric_shapes_inference_cnn",
    )
    return model


def build_training_model(num_classes: int) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
        ],
        name="data_augmentation",
    )
    classifier = build_inference_model(num_classes)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),
            augmentation,
            classifier,
        ],
        name="geometric_shapes_training_cnn",
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
        if layer.name == "geometric_shapes_inference_cnn":
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
        "classes": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "samples_per_class": SAMPLES_PER_CLASS,
        "seed": SEED,
    }
    (SUBMISSION_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_tflite_inference() -> None:
    interpreter = tf.lite.Interpreter(model_path=str(TFLITE_DIR / "model.tflite"))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample_path = next((SPLIT_DIR / "test" / CLASS_NAMES[0]).glob("*.png"))
    image = Image.open(sample_path).convert("RGB").resize(IMAGE_SIZE)
    input_data = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_index = int(np.argmax(predictions))

    print(f"Sample image: {sample_path}")
    print(f"Predicted label: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {predictions[predicted_index]:.4f}")
    print(f"All probabilities: {dict(zip(CLASS_NAMES, predictions.round(4).tolist()))}")


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

    history = model.fit(train_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=callbacks)
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
