# Import necessary libraries
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Reshape, Flatten, Dense, Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Check if TensorFlow is using GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is using the CPU.")

base_dir = os.path.dirname(os.path.realpath(__file__))  # Get the current working directory
dataset_path = os.path.join(base_dir, 'data', 'fer2013.csv')

# Load the dataset
print("Loading the dataset from", dataset_path)
df = pd.read_csv(dataset_path)

# Define constants
num_classes = 7
width = 48
height = 48
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
classes = np.array(emotion_labels)

# Initialize data lists
X_train = []
y_train = []
X_test = []
y_test = []

# Process the dataset
print("Processing the dataset...")
for index, row in df.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])

X_train = np.array(X_train, dtype='uint8')
y_train = np.array(y_train, dtype='uint8')
X_test = np.array(X_test, dtype='uint8')
y_test = np.array(y_test, dtype='uint8')

# Reshape the data to fit the model input
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# One-hot encode the labels
print("One-hot encoding the labels...")
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Initialize data generators for augmentation
print("Initializing data generators for augmentation...")
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

testgen = ImageDataGenerator(rescale=1./255)

# Fit the training data
datagen.fit(X_train)
batch_size = 64

# Create training and testing data generators
train_flow = datagen.flow(X_train, y_train, batch_size=batch_size)
test_flow = testgen.flow(X_test, y_test, batch_size=batch_size)

# Define the spatial attention block
def spatial_attention_block(input_tensor):
    avg_pool = GlobalAveragePooling2D()(input_tensor)
    avg_pool = Reshape((1, 1, input_tensor.shape[-1]))(avg_pool)
    attention_map = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(avg_pool)
    attended_output = Multiply()([input_tensor, attention_map])
    return attended_output

# Build the model
def FER_Model(input_shape=(48, 48, 1)):
    visible = Input(shape=input_shape, name='input')

    # Block 1
    conv1_1 = Conv2D(64, 3, activation='relu', padding='same')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='same')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1 = MaxPooling2D(2)(conv1_2)
    drop1 = Dropout(0.3)(pool1)
    drop1 = spatial_attention_block(drop1)  # Apply spatial attention

    # Block 2
    conv2_1 = Conv2D(128, 3, activation='relu', padding='same')(drop1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='same')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    conv2_3 = Conv2D(128, 3, activation='relu', padding='same')(conv2_2)
    conv2_3 = BatchNormalization()(conv2_3)
    pool2 = MaxPooling2D(2)(conv2_3)
    drop2 = Dropout(0.3)(pool2)
    drop2 = spatial_attention_block(drop2)  # Apply spatial attention

    # Block 3
    conv3_1 = Conv2D(256, 3, activation='relu', padding='same')(drop2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='same')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    conv3_3 = Conv2D(256, 3, activation='relu', padding='same')(conv3_2)
    conv3_3 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(256, 3, activation='relu', padding='same')(conv3_3)
    conv3_4 = BatchNormalization()(conv3_4)
    pool3 = MaxPooling2D(2)(conv3_4)
    drop3 = Dropout(0.3)(pool3)
    drop3 = spatial_attention_block(drop3)  # Apply spatial attention

    # Block 4
    conv4_1 = Conv2D(256, 3, activation='relu', padding='same')(drop3)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, 3, activation='relu', padding='same')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    conv4_3 = Conv2D(256, 3, activation='relu', padding='same')(conv4_2)
    conv4_3 = BatchNormalization()(conv4_3)
    conv4_4 = Conv2D(256, 3, activation='relu', padding='same')(conv4_3)
    conv4_4 = BatchNormalization()(conv4_4)
    pool4 = MaxPooling2D(2)(conv4_4)
    drop4 = Dropout(0.3)(pool4)
    drop4 = spatial_attention_block(drop4)  # Apply spatial attention

    # Block 5
    conv5_1 = Conv2D(512, 3, activation='relu', padding='same')(drop4)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, 3, activation='relu', padding='same')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    conv5_3 = Conv2D(512, 3, activation='relu', padding='same')(conv5_2)
    conv5_3 = BatchNormalization()(conv5_3)
    conv5_4 = Conv2D(512, 3, activation='relu', padding='same')(conv5_3)
    conv5_4 = BatchNormalization()(conv5_4)
    pool5 = MaxPooling2D(2)(conv5_4)
    drop5 = Dropout(0.3)(pool5)
    drop5 = spatial_attention_block(drop5)  # Apply spatial attention

    flatten = Flatten()(drop5)
    output = Dense(num_classes, activation='softmax')(flatten)

    model = Model(inputs=visible, outputs=output)
    print(model.summary())
    return model

# Create and compile the model
print("Compiling the model...")
model = FER_Model()
opt = Adam(learning_rate=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Set up callbacks
print("Setting up callbacks...")
filepath = "weights_min_loss.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train the model
print("Training the model...")
num_epochs = 100
history = model.fit(
    train_flow, 
    steps_per_epoch=len(X_train) / batch_size, 
    epochs=num_epochs,  
    verbose=2,  
    callbacks=callbacks_list,
    validation_data=test_flow,  
    validation_steps=len(X_test) / batch_size
)

# Evaluate the model
print("Evaluating the model...")
loss = model.evaluate(X_test / 255., y_test) 
print("Test Loss: ", loss[0])
print("Test Accuracy: ", loss[1])

# Save the model and weights to the models/ directory
print("Saving the model to disk...")
models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True) 

model_json_path = os.path.join(models_dir, "model.json")
model_h5_path = os.path.join(models_dir, "model.h5")

# Save the model architecture as JSON
with open(model_json_path, "w") as json_file:
    json_file.write(model.to_json())

# Save the model weights
model.save_weights(model_h5_path)
print("Model and weights saved successfully.")
