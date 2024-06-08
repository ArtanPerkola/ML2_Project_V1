from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Laden des EfficientNetB0-Modells ohne die letzten Klassifikationsschichten
base_model = EfficientNetB0(weights='imagenet', include_top=False)

# Hinzufügen neuer Schichten für die spezifische Aufgabe
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Neues Modell erstellen
model = Model(inputs=base_model.input, outputs=predictions)

# Obere Schichten einfrieren
for layer in base_model.layers:
    layer.trainable = False

# Modell kompilieren
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation für das Training
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Data Augmentation für die Validierung
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Training des Modells
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Speichern des Modells
model.save('efficientnet_model.keras')
