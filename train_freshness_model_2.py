import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # "2" = filter out INFO and WARNING

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = "FreshnessDataset copy"
OUTPUT_ROOT = "multi_dataset"

FRUITS = ["Apple", "Banana", "Carrot", "Orange", "Tomato"]
TARGET_PER_CLASS = 2400   # Number to sample per fruit per freshness group

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------
# CLEAN OUTPUT DIRECTORY
# -----------------------------
if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

os.makedirs(f"{OUTPUT_ROOT}/train", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/valid", exist_ok=True)
os.makedirs(f"{OUTPUT_ROOT}/test",  exist_ok=True)

# -----------------------------
# STEP 1 — COLLECT FILES
# -----------------------------
def collect_image_paths(base_dir):
    """Returns a list of absolute file paths for images in a directory."""
    p = Path(base_dir)
    return [str(x) for x in p.glob("*") if x.suffix.lower() in [".jpg", ".png", ".jpeg"]]

fresh_dirs = {fruit: f"{DATASET_ROOT}/Fresh/Fresh{fruit}" for fruit in FRUITS}
rotten_dirs = {fruit: f"{DATASET_ROOT}/Rotten/Rotten{fruit}" for fruit in FRUITS}

# -----------------------------
# STEP 2 — SAMPLE EXACTLY 2400 FOR TRAIN/VAL, REST → TEST
# -----------------------------
all_data = []  # Will store dictionaries with paths and labels

for fruit in FRUITS:
    for freshness, dir_path in [("Fresh", fresh_dirs[fruit]), ("Rotten", rotten_dirs[fruit])]:

        images = collect_image_paths(dir_path)
        random.shuffle(images)

        # ✨ SELECT EXACTLY 2400 FOR TRAIN/VAL
        selected = images[:TARGET_PER_CLASS]

        # ✨ REMAINDER → test set
        remainder = images[TARGET_PER_CLASS:]

        # Store each entry with a multi-label encoding
        for img in selected:
            all_data.append({
                "path": img,
                "fruit": fruit,
                "freshness": 1 if freshness == "Fresh" else 0
            })

        # Add the remainder directly to the testing folder
        test_out_dir = f"{OUTPUT_ROOT}/test/{freshness}/{fruit}"
        os.makedirs(test_out_dir, exist_ok=True)

        for img in remainder:
            shutil.copy(img, test_out_dir)

# -----------------------------
# STEP 3 — TRAIN/VAL SPLIT FROM SELECTED 2400-PER-CLASS
# -----------------------------
train_data, valid_data = train_test_split(
    all_data, test_size=0.2, shuffle=True, stratify=[x["fruit"] for x in all_data]
)

# -----------------------------
# STEP 4 — COPY TO STRUCTURED FOLDERS
# -----------------------------
def copy_dataset(data_list, target_split):
    for entry in data_list:
        fruit = entry["fruit"]
        freshness = "Fresh" if entry["freshness"] == 1 else "Rotten"

        out_dir = f"{OUTPUT_ROOT}/{target_split}/{freshness}/{fruit}"
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy(entry["path"], out_dir)

copy_dataset(train_data, "train")
copy_dataset(valid_data, "valid")

print("DATASET READY ✔")

# -----------------------------
# CNN MULTI-TASK MODEL
# -----------------------------
def build_model():
    inputs = Input(shape=(*IMAGE_SIZE, 3))

    x = layers.Rescaling(1./255)(inputs)

    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    # ------------------
    # Multi-task outputs
    # ------------------
    fruit_output = layers.Dense(len(FRUITS), activation='softmax', name="fruit_type")(x)
    fresh_output = layers.Dense(1, activation='sigmoid', name="freshness")(x)

    model = models.Model(inputs, [fruit_output, fresh_output])

    model.compile(
        optimizer='adam',
        loss={
            "fruit_type": "categorical_crossentropy",
            "freshness": "binary_crossentropy"
        },
        metrics={
            "fruit_type": "accuracy",
            "freshness": "accuracy"
        }
    )

    return model


model = build_model()
model.summary()

# -----------------------------
# DATA GENERATORS
# -----------------------------

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1
)

valid_gen = ImageDataGenerator(rescale=1./255)

def multi_label_generator(generator, directory):
    gen = generator.flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=True
    )

    while True:
        batch = next(gen)
        labels_fruit = []
        labels_fresh = []

        for fname in gen.filenames[gen.index_array]:
            # filename format: Fresh/Apple/img123.jpg
            freshness = 1 if "Fresh" in fname else 0
            fruit = next(f for f in FRUITS if f in fname)

            labels_fruit.append(FRUITS.index(fruit))
            labels_fresh.append(freshness)

        labels_fruit = tf.keras.utils.to_categorical(labels_fruit, num_classes=len(FRUITS))
        labels_fresh = np.array(labels_fresh)

        yield batch, {"fruit_type": labels_fruit, "freshness": labels_fresh}

train_gen_multi = multi_label_generator(train_gen, f"{OUTPUT_ROOT}/train")
val_gen_multi   = multi_label_generator(valid_gen, f"{OUTPUT_ROOT}/valid")

# -----------------------------
# TRAIN MODEL
# -----------------------------
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------------------------
# EARLY STOPPING + BEST MODEL CHECKPOINT
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,              # stop after 5 epochs without improvement
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "fruit_freshness_multitask_best.keras",
    monitor="val_loss",
    save_best_only=True
)

model.fit(
    train_gen_multi,
    steps_per_epoch=len(train_data) // BATCH_SIZE,
    validation_data=val_gen_multi,
    validation_steps=len(valid_data) // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

model.save("fruit_freshness_multitask.keras")

print("TRAINING COMPLETE ✔")
