import cv2 
import numpy as np 
import pydicm 
import tensorflow as tf 
import pandas as pd 

def cropped_image_data(image_data, description,condition,  mask, new_shape=(224 , 224)):
        xmin, ymin, xmax, ymax = get_bounding_box(mask)
        cropped_image = image_data[ymin:ymax + 1, xmin:xmax + 1]
        cropped_image = cv2.resize(cropped_image, new_shape)
        cropped_image = cropped_image / np.max(cropped_image)
        return cropped_image, description,condition 

def get_bounding_box(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max
def overall_prediction(dataframe, segmentation_model, severity_model ,reverse_label_encoder):
    row_ids = []
    normal_milds = []
    moderates = []
    severes = []
    
    for index, row in dataframe.iterrows():  # Use dataframe.iterrows() to iterate over rows
        file_path = row['file_path']
        series_desc = row['series_encoder']
        image_shape = (224 ,224)
        meta_data = pydicom.dcmread(file_path)
        pixel_array = meta_data.pixel_array
        image_data = cv2.resize(pixel_array, image_shape)
        image_data = image_data / np.max(image_data)
        image_data = np.stack((image_data,)*3, axis=-1)
        image_data_pred  = np.expand_dims(image_data, axis=0)
        description= np.expand_dims(np.array(series_desc), axis=0)
        mask_data, label = segmentation_model.predict([image_data_pred, description])
        label = np.argmax(label)
        label_encode  = reverse_label_encoder[label]
        if label_encode !='No condition':

            cropped_image, series_desc, condition = cropped_image_data(image_data, series_desc, label, mask_data)
            description= np.expand_dims(np.array(series_desc), axis=0)

            # Prepare inputs for prediction
            cropped_image = np.expand_dims(cropped_image, axis=0)
            condition = np.expand_dims(np.array(condition), axis=0)

            # Perform prediction with severity model
            prediction = severity_model.predict([cropped_image,description,condition])
            probabilities = tf.nn.softmax(prediction).numpy()
            normal_mild, moderate, severe = probabilities[0]

            # Generate row_id
            row_id = f"44036939_{label_encode}"

            # Append results to lists
            row_ids.append(row_id)
            normal_milds.append(normal_mild)
            moderates.append(moderate)
            severes.append(severe)

        # Create submission dataframe
        submission_df = pd.DataFrame({
            'row_id': row_ids,
            'normal_mild': normal_milds,
            "moderate":moderates , 
            'severe': severes
        })

        # Save submission dataframe to CSV
        submission_df.to_csv("submission.csv", index=False)
