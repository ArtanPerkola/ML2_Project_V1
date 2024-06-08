import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Datenpfade
train_dir = 'data/train'
val_dir = 'data/validation'

# Bildgrößen und Batchgröße
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Datenvorbereitung
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# ResNet50-Modell laden
base_model = ResNet50(weights='imagenet', include_top=False)

# Hinzufügen der Schichten
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Modell definieren
model = Model(inputs=base_model.input, outputs=predictions)

# Nur die oberen Schichten trainieren
for layer in base_model.layers:
    layer.trainable = False

# Kompilieren des Modells
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Checkpoints und Early Stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')

# Modelltraining
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=50,
    callbacks=[checkpoint, early_stopping]
)

# Modell speichern
model.save('final_model.keras')
