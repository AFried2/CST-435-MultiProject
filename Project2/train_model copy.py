import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import os
import json

print("TensorFlow Version:", tf.__version__)

# --- 1. Define Paths and Constants ---
# Make sure you have downloaded the dataset and it's in this directory structure
base_dir = './Intel_Image_Classification/'
train_dir = os.path.join(base_dir, 'seg_train/seg_train')
validation_dir = os.path.join(base_dir, 'seg_test/seg_test')

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# --- 2. Data Preprocessing and Augmentation ---
# Create an ImageDataGenerator for training data with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Create an ImageDataGenerator for validation data (only rescaling)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Flow validation images in batches
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 3. Save Class Indices ---
# The app will need this to map predictions to class names
class_indices = train_generator.class_indices
# Invert the dictionary to map index to label
labels_map = {v: k for k, v in class_indices.items()}
with open('class_labels.json', 'w') as f:
    json.dump(labels_map, f)
print(f"Saved class labels to class_labels.json: {labels_map}")


# --- 4. Build the CNN Model ---
model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    
    Dense(128, activation='relu'),
    Dense(6, activation='softmax') # 6 classes
])

model.summary()

# --- 5. Compile and Train the Model ---
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Use EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5, # Increased patience slightly
    restore_best_weights=True
)

print("\nStarting model training...")
history = model.fit(
    train_generator,
    epochs=50, # Set a high number, EarlyStopping will find the best one
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# --- 6. Evaluate and Save the Final Model ---
loss, accuracy = model.evaluate(validation_generator)
print(f"\nFinal Validation Accuracy: {accuracy*100:.2f}%")
print(f"Final Validation Loss: {loss:.4f}")

# Save the trained model in the recommended .keras format
model.save('natural_scene_classifier.keras')
print("\nModel training complete and saved to natural_scene_classifier.keras")