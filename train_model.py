# ==========================================================
# Apple Disease Detection Model Training Script
# Author: Nasir Uddin
# Master's Level Project - Deep Learning
# ==========================================================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ==========================================================
# 1. Dataset Path
# ==========================================================
# Assuming dataset structure:
# dataset/
# ├── train/
# │   ├── Apple___Apple_scab/
# │   ├── Apple___Black_rot/
# │   ├── Apple___Cedar_apple_rust/
# │   └── Apple___healthy/
# ├── test/
#     ├── (same structure)

train_dir = 'data/train'
test_dir = 'data/test'

# ==========================================================
# 2. Image Preprocessing & Augmentation
# ==========================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ==========================================================
# 3. Build Model (Transfer Learning - MobileNetV2)
# ==========================================================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers for faster training

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(train_generator.num_classes, activation='softmax')
])

# ==========================================================
# 4. Compile Model
# ==========================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================================
# 5. Train Model
# ==========================================================
EPOCHS = 10

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# ==========================================================
# 6. Save Model
# ==========================================================
os.makedirs('model', exist_ok=True)
model.save('model/apple_model.h5')
print("✅ Model saved successfully as 'apple_model.h5'!")

# ==========================================================
# 7. Plot Accuracy and Loss
# ==========================================================
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
