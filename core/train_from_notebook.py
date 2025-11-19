# Extracted from Plant_Disease_Detector_using_CNN.ipynb

!pip install tensorflow

import os

import shutil

import random

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path



import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# -------------------------------------------------------

# CONFIG

# -------------------------------------------------------

DATA_DIR = Path("./PlantVillage")

ALL_DATA = DATA_DIR / "all_data"



TRAIN_DIR = DATA_DIR / "train"

VALID_DIR = DATA_DIR / "valid"

TEST_DIR  = DATA_DIR / "test"



BATCH_SIZE = 32

IMG_HEIGHT = 224

IMG_WIDTH = 224

EPOCHS_WARMUP = 5

EPOCHS_FINETUNE = 15



# -------------------------------------------------------

# STEP 1 — CLEAN + AUTO-SPLIT

# -------------------------------------------------------

def auto_split():

    if not ALL_DATA.exists():

        print("ERROR: PlantVillage/all_data missing.")

        return False



    # Clean old splits

    for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:

        if folder.exists():

            shutil.rmtree(folder)

        folder.mkdir(parents=True, exist_ok=True)



    classes = [d for d in ALL_DATA.iterdir() if d.is_dir()]

    print("Found classes:", [c.name for c in classes])



    for class_dir in classes:

        cname = class_dir.name

        imgs = list(class_dir.glob("*"))

        random.shuffle(imgs)

        total = len(imgs)



        train_n = int(0.8 * total)

        valid_n = int(0.9 * total)



        (TRAIN_DIR / cname).mkdir()

        (VALID_DIR / cname).mkdir()

        (TEST_DIR / cname).mkdir()



        for img in imgs[:train_n]:

            shutil.copy(img, TRAIN_DIR / cname)

        for img in imgs[train_n:valid_n]:

            shutil.copy(img, VALID_DIR / cname)

        for img in imgs[valid_n:]:

            shutil.copy(img, TEST_DIR / cname)



    print("Auto-split complete!")

    return True



# -------------------------------------------------------

# STEP 2 — LOAD PROCESSED DATA

# -------------------------------------------------------

def load_data():

    train_ds = tf.keras.utils.image_dataset_from_directory(

        TRAIN_DIR, image_size=(IMG_HEIGHT, IMG_WIDTH),

        batch_size=BATCH_SIZE, seed=123)



    val_ds = tf.keras.utils.image_dataset_from_directory(

        VALID_DIR, image_size=(IMG_HEIGHT, IMG_WIDTH),

        batch_size=BATCH_SIZE, seed=123)



    test_ds = tf.keras.utils.image_dataset_from_directory(

        TEST_DIR, image_size=(IMG_HEIGHT, IMG_WIDTH),

        batch_size=BATCH_SIZE, seed=123)



    class_names = train_ds.class_names

    print("Classes:", class_names)



    # Augmentation

    aug = tf.keras.Sequential([

        tf.keras.layers.RandomFlip("horizontal"),

        tf.keras.layers.RandomRotation(0.15),

        tf.keras.layers.RandomZoom(0.2),

    ])



    # Preprocessing for EfficientNet (IMPORTANT!)

    train_ds = train_ds.map(lambda x, y: (preprocess_input(aug(x)), y))

    val_ds   = val_ds.map(lambda x, y: (preprocess_input(x), y))

    test_ds  = test_ds.map(lambda x, y: (preprocess_input(x), y))



    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)

    val_ds = val_ds.prefetch(AUTOTUNE)

    test_ds = test_ds.prefetch(AUTOTUNE)



    return train_ds, val_ds, test_ds, class_names



# -------------------------------------------------------

# STEP 3 — CREATE MODEL (EfficientNetB0)

# -------------------------------------------------------

def build_model(num_classes):

    base = EfficientNetB0(include_top=False, weights="imagenet",

                          input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    base.trainable = False  # warmup only



    model = Sequential([

        base,

        GlobalAveragePooling2D(),

        Dropout(0.4),

        Dense(num_classes, activation="softmax")

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-4),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    model.summary()

    return model, base



# -------------------------------------------------------

# STEP 4 — TRAIN (Warmup + Fine Tune)

# -------------------------------------------------------

def train_model(model, base, train_ds, val_ds):

    print("\n--- WARMUP TRAINING ---")

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP)



    print("\n--- FINE-TUNING ---")

    base.trainable = True

    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-5),

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )



    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

    return history



# -------------------------------------------------------

# MAIN

# -------------------------------------------------------

if auto_split():

    train_ds, val_ds, test_ds, class_names = load_data()

    model, base = build_model(len(class_names))

    history = train_model(model, base, train_ds, val_ds)



    print("\nEvaluating test set...")

    loss, acc = model.evaluate(test_ds)

    print(f"Test Accuracy: {acc:.4f}")



    # Confusion matrix

    print("\nConfusion matrix:")

    y_true = np.concatenate([y for x, y in test_ds])

    y_pred = np.argmax(model.predict(test_ds), axis=1)



    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(cmap="Blues", xticks_rotation=45)

    plt.show()


   import json

   import os

   

   # Ensure directory exists

   os.makedirs('models', exist_ok=True)

   

   # Save class indices

   class_indices = {class_name: idx for class_name, idx in enumerate(class_names)}

   with open('models/class_indices.json', 'w') as f:

       json.dump(class_indices, f)

import json

import os



os.makedirs('models', exist_ok=True)



# CORRECT format: {index: name}

class_indices = {str(i): name for i, name in enumerate(class_names)}



with open('models/class_indices.json', 'w') as f:

    json.dump(class_indices, f, indent=2)

import json

import os



os.makedirs('models', exist_ok=True)

model.save('models/plant_disease_model.keras')



# Get class names from folder (no train_dataset needed!)

class_names = sorted(os.listdir('PlantVillage/train'))



class_indices = {str(i): name for i, name in enumerate(class_names)}

with open('models/class_indices.json', 'w') as f:

    json.dump(class_indices, f, indent=2)



print("✓ Done!")

print(f"Classes: {class_names}")

print(f"Model: {os.path.abspath('models/plant_disease_model.keras')}")

print(f"Classes JSON: {os.path.abspath('models/class_indices.json')}")




