import tensorflow as tf 
import cv2 
import pydicom
from tensorflow.keras.utils import Sequence
import numpy as np 

class CustomSeverityDataGenerator(Sequence):
    def __init__(self, dataframe, batch_size=32, shuffle=True, new_shape=(224, 224)):
        self.data = dataframe 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.new_shape = new_shape
        self.indexes = np.arange(len(self.data))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indexes = self.indexes[start_index:end_index]

        batch_descriptions, cropped_images, labels, conditions = [], [], [], []

        for idx in batch_indexes:
            row = self.data.iloc[idx]
            image_path = row['file_path']
            description = row['series_description_encode']
            x = row['x']
            y = row['y']
            condition = row['condition_encode']
            label = row['severity_encoder']
            
            if np.isnan(x) or np.isnan(y):
                x = None
                y = None


            image_data, mask = self.__medical_preprocess_cropped(image_path, x, y)

            cropped_image, batch_description, condition, label = self.__cropped_image(image_data, description, condition, mask, label)

            cropped_images.append(cropped_image)
            batch_descriptions.append(batch_description)
            labels.append(label)
            conditions.append(condition)

        cropped_images = np.array(cropped_images, dtype=np.float32)
        batch_descriptions = np.array(batch_descriptions, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)
        conditions = np.array(conditions, dtype=np.int32)

        inputs = (tf.convert_to_tensor(cropped_images, dtype=tf.float32), tf.convert_to_tensor(batch_descriptions, dtype=tf.float32), tf.convert_to_tensor(conditions, dtype=tf.int32))
        targets = (tf.convert_to_tensor(labels, dtype=tf.int32))

        return inputs, targets

    def __cropped_image(self, image_data, description,condition,  mask, label, new_shape=(224 , 224)):
        xmin, ymin, xmax, ymax = self.__get_bounding_box(mask)
        cropped_image = image_data[ymin:ymax + 1, xmin:xmax + 1]
        cropped_image = cv2.resize(cropped_image, new_shape)
        cropped_image = cropped_image / 255.0
        return cropped_image, description,condition , label

    def __get_bounding_box(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max

    def __generate_mask_from_coordinates(self, image_shape, x, y, mask_size=25):
        xmin = int(np.floor(x - mask_size / 2))
        xmax = int(np.ceil(x + mask_size / 2))
        ymin = int(np.floor(y - mask_size / 2))
        ymax = int(np.ceil(y + mask_size / 2))

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_shape[1], xmax)
        ymax = min(image_shape[0], ymax)

        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        mask[ymin:ymax, xmin:xmax] = 255

        return mask, (xmin, ymin, xmax, ymax)

    def __adjust_coordinates(self, x, y, original_shape, new_shape):
        original_height, original_width = original_shape[:2]
        new_height, new_width = new_shape[:2]
        x_new = x * new_width / original_width
        y_new = y * new_height / original_height
        return x_new, y_new

    def __medical_preprocess_cropped(self, image_path, original_x=None, original_y=None, new_shape=(224, 224)):
        try:
            if isinstance(image_path, np.ndarray):
                image_path = image_path.item()
            if isinstance(image_path, bytes):
                image_path = image_path.decode()

            meta_data = pydicom.dcmread(image_path)
            pixel_array = meta_data.pixel_array

            image = cv2.resize(pixel_array, new_shape)
            image = image / np.max(image)
            image = np.stack((image,)*3, axis=-1)

            original_shape = pixel_array.shape

            if original_x is not None and original_y is not None:
                new_x, new_y = self.__adjust_coordinates(original_x, original_y, original_shape, new_shape)
                mask, bbox = self.__generate_mask_from_coordinates(image.shape, new_x, new_y)
            else:
                mask = np.zeros(new_shape[:2], dtype=np.uint8)

            return image, mask
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return np.zeros(new_shape[:2], dtype=np.uint8), np.zeros(new_shape[:2], dtype=np.uint8)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
