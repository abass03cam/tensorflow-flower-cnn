# 🌸 Flower Classification using CNN in TensorFlow/Keras
# Author: Fodé Abass Camara
# Dataset: flower_photos (5 categories)
# Aufgabe: Bildklassifikation und Vorhersage mit Sicherheitsbewertung

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pathlib

# === Parameter ===
data_dir = pathlib.Path("c:/Users/Desktop/K_IntelligenzBigDataDataScience/29.04/elli/flower_photos")
img_height = 180
img_width = 180
batch_size = 32
seed = 123

# === Datensatz laden ===
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# === Klassen auslesen ===
class_names = train_ds.class_names
num_classes = len(class_names)

# === Performanceoptimierung (Prefetching) ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# === Modell definieren (CNN) ===
model = models.Sequential([
    Input(shape=(img_height, img_width, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# === Kompilieren ===
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# === Training starten ===
print("📊 Starte Training...")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# === Anzahl Bilder im Datensatz ===
image_count = len(list(data_dir.glob('*/*.jpg')))
print("Anzahl Bilder im Datensatz:", image_count)

# === Beispielhafte Klassifikation eines neuen Bildes ===
img_path = "c:/Users/Desktop/K_IntelligenzBigDataDataScience/29.04/elli/blume1.jpg"
img = load_img(img_path, target_size=(img_height, img_width))
img_array = img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Form: (1, height, width, 3)

# === Vorhersage durchführen ===
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# === Ergebnis anzeigen ===
predicted_class = class_names[np.argmax(score)]
confidence = 100 * np.max(score)

print("\n📷 Vorhersage:", predicted_class)
print("🔒 Sicherheit: {:.2f}%".format(confidence))

