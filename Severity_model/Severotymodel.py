from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Input, GlobalAveragePooling2D, Dense,Dropout, Flatten , concatenate
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model


# input_shape = (224, 224, 3)

def create_severity_model(input_shape):
    image_input = Input(shape=input_shape, name='image_input')
    series_description_input = Input(shape=(1,), name='series_description')
    condition_level_input = Input(shape=(1,), name='condition_level')
    
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=image_input)
    base_model.load_weights("/kaggle/input/pretrained-model/efficientnetb0_notop.h5")
    
    x = base_model.output
    x = Flatten()(x)
    metadata_concatenated = concatenate([series_description_input, condition_level_input])
    concatenated = concatenate([x, metadata_concatenated])

    x = Dense(256, activation='relu')(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    
    model = Model(inputs=[image_input, series_description_input, condition_level_input], outputs=output)
    return model

