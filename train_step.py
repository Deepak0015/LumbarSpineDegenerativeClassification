import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split



def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = LearningRateScheduler(scheduler)


def train_step(segmentation_model , CustomDataGenerator , final_df ):
    train_size  = 90000
    val_size  = 7384
    train_df, val_df = train_test_split(final_df, test_size=val_size, train_size=train_size)

    train_generator = CustomDataGenerator(train_df, batch_size=32)
    val_generator = CustomDataGenerator(val_df, batch_size=32)
    segmentation_model.compile(optimizer='adam', 
              loss={'mask_output': 'binary_crossentropy', 'label_output': 'sparse_categorical_crossentropy'},
              metrics={'mask_output': 'accuracy', 'label_output': 'accuracy'})
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpoint = ModelCheckpoint('segmentation_model_checkpoint.keras', save_best_only=True)


    print("Starting model training...")
    history = segmentation_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        callbacks=[early_stopping, checkpoint , lr_scheduler],
        verbose=1
    )
    print("Model training completed.")
    segmentation_model.save("segmentation")
