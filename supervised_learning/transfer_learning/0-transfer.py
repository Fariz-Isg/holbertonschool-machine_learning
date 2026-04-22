#!/usr/bin/env python3
"""Trains a CNN for CIFAR-10 classification using transfer learning."""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocesses CIFAR-10 data for ResNet50."""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    # Load and preprocess data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    # Input layer
    input_tensor = K.Input(shape=(32, 32, 3))

    # Scale up data to 224x224 (required for ResNet50)
    # Using UpSampling2D as a simple scaling mechanism
    # Alternatively Lambda with tf.image.resize
    upscale = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, 7, 7, "channels_last")
    )(input_tensor)

    # Base model: ResNet50 with pre-trained ImageNet weights
    base_model = K.applications.ResNet50(include_top=False,
                                         weights='imagenet',
                                         input_tensor=upscale)

    # Freeze the base model
    base_model.trainable = False

    # Create top model
    model = K.Sequential([
        base_model,
        K.layers.Flatten(),
        K.layers.BatchNormalization(),
        K.layers.Dense(256, activation='relu'),
        K.layers.Dropout(0.3),
        K.layers.BatchNormalization(),
        K.layers.Dense(128, activation='relu'),
        K.layers.Dropout(0.3),
        K.layers.Dense(10, activation='softmax')
    ])

    # Compile model
    # Note: Accuracy must be >= 87%
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Training
    # In a real scenario, this would take many epochs
    # For the script, we set it up to save the model
    model.fit(X_train, Y_train,
              batch_size=128,
              epochs=10,
              validation_data=(X_test, Y_test),
              verbose=1)

    # Save the trained model
    model.save('cifar10.h5')
