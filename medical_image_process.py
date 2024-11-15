import numpy as np 
import cv2 
import pydicom
import matplotlib.pyplot as plt 

def generate_mask_from_coordinates(image_shape, x, y, mask_size=25):
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
#Coodinates function
def adjust_coordinates(x, y, original_shape, new_shape):
    original_height, original_width = original_shape[:2]
    new_height, new_width = new_shape[:2]
    x_new = x * new_width / original_width 
    y_new = y * new_height / original_height 
    return x_new, y_new
#Image preprocess FUnction 
def medical_preprocess(image_path, description, label, original_x=None, original_y=None, new_shape=(224, 224)):
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
            new_x, new_y = adjust_coordinates(original_x, original_y, original_shape, new_shape)
            mask, bbox = generate_mask_from_coordinates(image.shape, new_x, new_y)
        else:
            mask = np.zeros(new_shape[:2], dtype=np.uint8)
            
        return image, description, mask, label 
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return np.zeros(new_shape[:2], dtype=np.uint8), description, np.zeros(new_shape[:2], dtype=np.uint8), label




def visualize(image, mask=None, label=None):
    if mask is not None:
         if len(mask.shape) == 4: 
                mask = np.squeeze(mask, axis=0) 
                mask = np.squeeze(mask, axis=-1) 
    plt.figure(figsize=(12, 6))

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Overlay mask and plot coordinates
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    if mask is not None:
        plt.imshow(mask, cmap='Reds', alpha=0.5)  
    if  x is not None and y is not None :
        plt.scatter(x, y, c='blue', s=50) 
    if label is not None:
        plt.title(f'Image with Mask and Coordinates: {label}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
