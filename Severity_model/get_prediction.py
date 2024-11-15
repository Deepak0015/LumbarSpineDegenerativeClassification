import pydicom
import cv2
import tensorflow as tf 
import numpy as np 


def severity_prediction(sevrity_model ,image_path, description, condition, input_shape=(224, 224)):
    
    meta_data = pydicom.dcmread(image_path)
    pixel_array = meta_data.pixel_array
    image_data = cv2.resize(pixel_array, input_shape)
    image_data = image_data / np.max(image_data)
    image_data = np.stack((image_data,) * 3, axis=-1)
    image_data = np.expand_dims(image_data, axis=0)
    
    description = np.expand_dims(np.array(description), axis=0)
    condition = np.expand_dims(np.array(condition), axis=0)
 
    prediction = sevrity_model.predict([image_data, description, condition])
    probabilities = tf.nn.softmax(prediction).numpy()
    normal_mild, moderate, severe = probabilities[0]
    
    print(f"normal/mild: {normal_mild}\nmoderate: {moderate}\nsevere: {severe}")
    return normal_mild, moderate, severe
