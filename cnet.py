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

############################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.figure(figsize=(10, 6))
data = pd.DataFrame({
    'model': ['CNN','LSTM','A-CLSTM'],
    'y': [89.3,91.1,96.2]})
custom_palette = sns.color_palette("Set1")
sns.barplot(x='model', y='y', data=data, palette=custom_palette, width=0.4)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(20,40)
plt.ylabel('Accuracy %', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig('et',dpi=1500,bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
data = pd.DataFrame({
    'model': ['CNN','LSTM','A-CLSTM'],
    'y': [88.5,90.6,95.7]})
custom_palette = sns.color_palette("Set1")
sns.barplot(x='model', y='y', data=data, palette=custom_palette, width=0.4)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(20,40)
plt.ylabel('Precision %', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig('dt',dpi=1500,bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
data = pd.DataFrame({
    'model': ['CNN','LSTM','A-CLSTM'],
    'y': [87.2,89.9,96.8]})
custom_palette = sns.color_palette("Set1")
sns.barplot(x='model', y='y', data=data, palette=custom_palette, width=0.4)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(20,40)
plt.ylabel('Recall %', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig('sl',dpi=1500,bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
data = pd.DataFrame({
    'model': ['CNN','LSTM','A-CLSTM'],
    'y': [87.8,90.2,96.2]})
custom_palette = sns.color_palette("Set1")
sns.barplot(x='model', y='y', data=data, palette=custom_palette, width=0.4)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
#plt.ylim(20,40)
plt.ylabel('F1-score %', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig('sl',dpi=1500,bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
x=[0,10,15,20]
y=[80,90,93,94.8]
plt.plot(x,y,'-b')
plt.ylabel('Accuracy %',fontweight='bold')
plt.xlabel('Epochs',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim(80,100)
plt.grid('True')
plt.show()

plt.figure(figsize=(10,6))
x=[0,10,15,20]
y=[80,90,93,94.1]
plt.plot(x,y,'-b')
plt.ylabel('Precision %',fontweight='bold')
plt.xlabel('Epochs',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim(80,100)
plt.grid('True')
plt.show()

plt.figure(figsize=(10,6))
x=[0,10,15,20]
y=[80,90,93,95.4]
plt.plot(x,y,'-b')
plt.ylabel('Recall %',fontweight='bold')
plt.xlabel('Epochs',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim(80,100)
plt.grid('True')
plt.show()

plt.figure(figsize=(10,6))
x=[0,10,15,20]
y=[80,90,93,94.7]
plt.plot(x,y,'-b')
plt.ylabel('F1-score %',fontweight='bold')
plt.xlabel('Epochs',fontweight='bold')
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim(80,100)
plt.grid('True')
plt.show()

############################################################################

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# Define the folder containing your dataset
data_folder = "\\Satellite datas"

# Define the list of class names
class_names = ['forest','ship','water']

# Function to load and preprocess the images
def load_and_preprocess_data(data_folder, class_names):
    X = []  # Features
    y = []  # Labels

    for class_index, class_name in enumerate(class_names):
        class_folder = os.path.join(data_folder, class_name)
        for image_filename in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            X.append(image)
            y.append(class_index)
    X = np.array(X)
    y = np.array(y)

    # Normalize pixel values to be between 0 and 1
    X = X / 255.0

    return X, y

X, y = load_and_preprocess_data(data_folder, class_names)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(class_names))
y_test = to_categorical(y_test, num_classes=len(class_names))

# Create a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Load a single test image
# Load a single test image
test_image = X_test[2]  # Change the index to the desired test image
print(test_image.shape)
# Reshape the image to match the model's input shape
test_image = cv2.resize(test_image, (224, 224))  # Resize the image to (224, 224)
test_image = test_image.reshape((1, 224, 224, 3))  # Reshape to (1, 224, 224, 3)

# Make predictions on the single test image
predictions = model.predict(test_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Map the indices to class names
predicted_class_name = class_names[predicted_class_index]

# Display the predicted and ground truth class names
print("Predicted class:", predicted_class_name)

# Display the test image
plt.imshow(test_image[0])
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()

test_image = "\\Satellite datas\\water\\water_body_8825.jpg"
test_image = cv2.imread(test_image)
test_image = cv2.resize(test_image, (224, 224))
test_image = test_image.reshape((1, 224, 224, 3))

# Make predictions on the single test image
predictions = model.predict(test_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Map the indices to class names
predicted_class_name = class_names[predicted_class_index]

# Display the predicted and ground truth class names
print("Predicted class:", predicted_class_name)

# Display the test image
plt.imshow(test_image[0])
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()

test_image = "\\Satellite datas\\forest\\7906_sat_17.jpg"
test_image = cv2.imread(test_image)
test_image = cv2.resize(test_image, (224, 224))
test_image = test_image.reshape((1, 224, 224, 3))

# Make predictions on the single test image
predictions = model.predict(test_image)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Map the indices to class names
predicted_class_name = class_names[predicted_class_index]

# Display the predicted and ground truth class names
print("Predicted class:", predicted_class_name)

# Display the test image
plt.imshow(test_image[0])
plt.title(f"Predicted: {predicted_class_name}")
plt.axis('off')
plt.show()
