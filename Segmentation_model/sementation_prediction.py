import numpy as np
import pydicom 
import cv2 
import matplotlib.pyplot as plt 

def get_prediction(segmentation_model ,image_path, description,reverse_label_encoder , image_shape=(224, 224)):
    meta_data = pydicom.dcmread(image_path)
    pixel_array = meta_data.pixel_array
    image_data = cv2.resize(pixel_array, image_shape)
    image_data = image_data / np.max(image_data)
    image_data = np.stack((image_data,)*3, axis=-1)
    image_data = np.expand_dims(image_data, axis=0)
    description = np.expand_dims(np.array(description), axis=0)
    mask_data, label = segmentation_model.predict([image_data, description])
    label = np.argmax(label)
    label_encode  = reverse_label_encoder[label]
    
    return image_data , mask_data, label , label_encode 


def visualize_cropped_image(original_image, cropped_image, mask, label):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.5) 
    plt.title('Cropped Image with Mask Overlay')
    plt.axis('off')
    plt.suptitle(f'Label: {label}', fontsize=16)
    plt.tight_layout()
    plt.show()
