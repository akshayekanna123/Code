from tensorflow.keras.layers import Input, Flatten, Dense, Concatenate, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset for demonstration
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10

# Preprocess the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Create VGG16 model
def create_vgg_model(input_shape):
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model_vgg.layers:
        layer.trainable = False
    flat1 = Flatten()(base_model_vgg.output)
    dense1 = Dense(256, activation='relu')(flat1)
    model = Model(inputs=base_model_vgg.input, outputs=dense1)
    return model

# Create ResNet50 model
def create_resnet_model(input_shape):
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model_resnet.layers:
        layer.trainable = False
    flat1 = Flatten()(base_model_resnet.output)
    dense1 = Dense(256, activation='relu')(flat1)
    model = Model(inputs=base_model_resnet.input, outputs=dense1)
    return model

# Input shape for CIFAR-10 images
input_shape = (32, 32, 3)

# Create VGG16 and ResNet50 models
vgg_model = create_vgg_model(input_shape)
resnet_model = create_resnet_model(input_shape)

# Concatenate the outputs of the two models
concatenated = Concatenate()([vgg_model.output, resnet_model.output])

# Additional convolutional layers
# Reshape to (batch_size, height, width, channels)
reshape_layer = Reshape((1, 1, -1))(concatenated)

# Check spatial dimensions before applying pooling
if reshape_layer.shape[1] > 1 and reshape_layer.shape[2] > 1:
    pool1 = MaxPooling2D(pool_size=(2, 2))(reshape_layer)
else:
    pool1 = GlobalAveragePooling2D()(reshape_layer)

# Additional dense layers
dense_layer = Dense(256, activation='relu')(pool1)
output_layer = Dense(num_classes, activation='softmax')(dense_layer)

# Create the final combined model
final_combined_model = Model(inputs=[vgg_model.input, resnet_model.input], outputs=output_layer)

# Compile the final combined model
final_combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print summary of the final combined model
final_combined_model.summary()

# Train the final combined model (modify this based on your training data)
final_combined_model.fit([x_train, x_train], y_train, epochs=5, batch_size=32, validation_data=([x_test, x_test], y_test))
