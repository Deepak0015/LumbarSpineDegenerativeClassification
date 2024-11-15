import tensorflow as tf 
from tensorflow.keras.utils import Sequence
from preproces.medical_image_process import medical_preprocess
import numpy as np 



class CustomDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, new_shape=(224, 224)):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.new_shape = new_shape
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_index:end_index]

        batch_images, batch_descriptions, batch_labels, batch_masks = [], [], [], []

        for idx in batch_indexes:
            row = self.dataframe.iloc[idx]
            image_path = row['file_path']
            description = row['series_description_encode']
            label = row['label_encoder']
            x = row['x']
            y = row['y']

            if np.isnan(x) or np.isnan(y):
                x = None
                y = None

            image, description, mask, label = medical_preprocess(image_path, description, label, x, y, self.new_shape)

            batch_images.append(image)
            batch_descriptions.append(description)
            batch_labels.append(label)
            batch_masks.append(mask)

        batch_images = np.array(batch_images, dtype=np.float32)
        batch_descriptions = np.array(batch_descriptions, dtype=np.float32)
        batch_labels = np.array(batch_labels, dtype=np.int32)
        batch_masks = np.array(batch_masks, dtype=np.float32)

        inputs = (tf.convert_to_tensor(batch_images, dtype=tf.float32), tf.convert_to_tensor(batch_descriptions, dtype=tf.float32))
        targets = (tf.convert_to_tensor(batch_masks, dtype=tf.float32), tf.convert_to_tensor(batch_labels, dtype=tf.int32))

        return inputs, targets

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
