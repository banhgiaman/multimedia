import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers.experimental import preprocessing

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.15),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def model_fn(preprocess_input, input_shape=(600, 600, 3), num_class=102):
    efficientnet_model = EfficientNetB0(
        include_top=False,
        input_shape=input_shape,
        weights='imagenet'
    )
    inputs = Input(shape=input_shape)
    x = img_augmentation(inputs)
    preprocess = preprocess_input(x)
    base_model = efficientnet_model(preprocess)
    pool = GlobalAveragePooling2D()(base_model)
    outputs = Dense(num_class, activation='softmax', name='output')(pool)
    model = Model(inputs=inputs, outputs=outputs)
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    metrics = tf.metrics.CategoricalAccuracy()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


if __name__ == '__main__':
    import os
    
    epochs = 8
    img_size = (600, 600)
    batch_size = 2
    model = model_fn(preprocess_input=preprocess_input)
    model.load_weights(os.path.join('models', 'EFF_102FLOWERS', 'checkpoints'))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join('models', 'EFF_102FLOWERS', 'checkpoints'),
        save_weights_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        save_best_only=True
    )
    model.summary()
    
    data_path = os.path.join('data', 'sorted')
    train_path = os.path.join(data_path, 'train')
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='categorical'
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    validation_path = os.path.join(data_path, 'valid')
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='categorical'
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    test_path = os.path.join(data_path, 'test')
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='categorical'
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    # train
    model.fit(
        train_dataset, epochs=epochs, validation_data=validation_dataset,
        callbacks=[model_checkpoint_callback]
    )
    model.save(os.path.join('models', 'EFF_102FLOWERS'))
    