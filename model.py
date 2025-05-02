import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Data Generator Function
def generator(dir, gen=ImageDataGenerator(rescale=1. / 255), shuffle=True, batch_size=32, target_size=(24, 24),
              class_mode='categorical'):
    return gen.flow_from_directory(
        dir,
        batch_size=batch_size,
        shuffle=shuffle,
        color_mode='grayscale',
        class_mode=class_mode,
        target_size=target_size
    )


# Paths for training and validation datasets
train_dir = 'C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/data/train'
valid_dir = 'C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/data/valid'

# Data Generators
train_batch = generator(train_dir)
valid_batch = generator(valid_dir)

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_batch, validation_data=valid_batch, epochs=15)

# Save the model
model.save('C:/Users/ABHIMANYU.M.B/Desktop/ML PRO/Machine-Learning-Projects-main/Drowsiness detection [OPEN CV]/models/cnnCat2.h5')
