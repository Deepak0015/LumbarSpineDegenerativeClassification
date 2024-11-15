import numpy as np
import tensorflow as tf 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input, GlobalAveragePooling2D, Dense,Dropout, Flatten , concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model


def create_model(input_shape=(224, 224, 3), num_classes=26):
    # Define image input
    image_input = Input(shape=input_shape, name='image_input')
    
    # Define description input
    desc_input = Input(shape=(1,), dtype=tf.float32, name='desc_input')
    base_model = EfficientNetB0(include_top=False, input_tensor=image_input, weights=None)
    
    # Load the pretrained weights
    base_model.load_weights("/kaggle/input/pretrained-model/efficientnetb0_notop.h5")
    
    # Freeze the base model layers
    base_model.trainable = False


    # Encoder network
    conv1 = base_model.get_layer('block2a_expand_activation').output
    conv2 = base_model.get_layer('block3a_expand_activation').output
    conv3 = base_model.get_layer('block4a_expand_activation').output
    conv4 = base_model.get_layer('block6a_expand_activation').output
    conv5 = base_model.output

    # Decoder network
    up6 = UpSampling2D((2, 2))(conv5)
    up6 = Conv2D(512, (2, 2), padding='same',kernel_regularizer=l2(0.01))(up6)
    merge6 = Concatenate()([conv4, up6])

    up7 = UpSampling2D((2, 2))(merge6)
    up7 = Conv2D(256, (2, 2), padding='same',kernel_regularizer=l2(0.01))(up7)
    merge7 = Concatenate()([conv3, up7])

    up8 = UpSampling2D((2, 2))(merge7)
    up8 = Conv2D(128, (2, 2), padding='same',kernel_regularizer=l2(0.01))(up8)
    merge8 = Concatenate()([conv2, up8])

    up9 = UpSampling2D((2, 2))(merge8)
    up9 = Conv2D(64, (2, 2), padding='same',kernel_regularizer=l2(0.01))(up9)
    up9 = Dropout(0.5)(up9) 
    merge9 = Concatenate()([conv1, up9])

    up10 = UpSampling2D((2, 2))(merge9)
    mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(up10)
    
    desc_flatten = Flatten()(desc_input)
#     desc_dense = Dense(128, activation='relu')(desc_flatten)
    desc_dense = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(desc_flatten)
    global_features = GlobalAveragePooling2D()(conv5)
    combined = Concatenate()([global_features, desc_dense])

    label_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01) ,  name='label_output')(combined)
#     label_output = Dense(num_classes, activation='softmax', name='label_output')(combined)

    model = Model(inputs=[image_input, desc_input], outputs=[mask_output, label_output])
    
    return model
