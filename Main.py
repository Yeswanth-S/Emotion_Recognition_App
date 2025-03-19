# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

# Data manipulation libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Image processing libraries
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# Check for GPU availability
print("Checking for GPU availability...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the GPU.")
else:
    print("TensorFlow is using the CPU.")

# Dataset loading and checking info
print("Loading dataset and checking information...")
df = pd.read_csv(r"data/raw/fer2013.csv")  
print(df.head())
print(df.info())

# Mapping emotion labels
print("Mapping emotion labels...")
label_map = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
df['emotion_label'] = df['emotion'].map(label_map)

# Plot Emotion Distribution
print("Plotting emotion distribution...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emotion_label', order=label_map.values())
plt.title('Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

# Convert pixel strings to arrays
print("Converting pixel strings to arrays...")
df['pixels'] = df['pixels'].apply(lambda x: np.array(list(map(int, x.split()))))

# Emotion Distribution per Usage
print("Plotting emotion distribution across dataset splits...")
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='emotion_label', hue='Usage', order=label_map.values())
plt.title('Emotion Distribution Across Dataset Splits')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.legend(title='Dataset Split')
plt.show()

# Plot Sample Images from the Dataset
print("Plotting sample images from the dataset...")
sample_data = df.sample(5)
plt.figure(figsize=(10, 10))
for i, (index, row) in enumerate(sample_data.iterrows()):
    image = row['pixels'].reshape(48, 48)
    plt.subplot(1, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(label_map[row['emotion']]) 
    plt.axis('off')
plt.tight_layout()
plt.show()

# Preparing data for training/testing
print("Preparing data for training and testing...")
num_classes = 7
width = 48
height = 48
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
classes = np.array(("Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"))
print(df.Usage.value_counts())

X_train = []
y_train = []
X_test = []
y_test = []
print("Splitting data into training and testing sets...")
for index, row in df.iterrows():
    k = row['pixels']
    if row['Usage'] == 'Training':
        X_train.append(k)
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(k)
        y_test.append(row['emotion'])

# Dataset training/testing split
print("Reshaping the training and testing data...")
X_train = np.array(X_train, dtype='uint8')
y_train = np.array(y_train, dtype='uint8')
X_test = np.array(X_test, dtype='uint8')
y_test = np.array(y_test, dtype='uint8')

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Show the shapes and number of samples in the training and test sets
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

print(f"Number of training samples: {len(X_train)}")
print(f"Number of test samples: {len(X_test)}")

# Apply data augmentation
print("Applying data augmentation...")

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest')

testgen = ImageDataGenerator(rescale=1./255)
datagen.fit(X_train)
batch_size = 64

train_flow = datagen.flow(X_train, y_train, batch_size=batch_size)

# Manually split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create data generators for validation
val_flow = testgen.flow(X_val, y_val, batch_size=batch_size)

print("Displaying augmented images...")

# Displaying augmented images
plt.figure(figsize=(10, 10))
for i, (X_batch, y_batch) in enumerate(datagen.flow(X_train, y_train, batch_size=9)):
    for j in range(9):
        plt.subplot(3, 3, j + 1)
        plt.imshow(X_batch[j].reshape(48, 48), cmap='gray')
        plt.title(label_map[np.argmax(y_batch[j])])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    break

# Model Implementation
print("Building the Fer model...")

# Define the spatial attention mechanism
def spatial_attention(input_feature):
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    attention = Conv2D(1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(concat)
    return Multiply()([input_feature, attention])


def FER_Model(input_shape=(48,48,1)):
    # Input layer
    visible = Input(shape=input_shape, name='input')
    num_classes = 7

    # 1st Block
    conv1_1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_1')(visible)
    conv1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, kernel_size=3, activation='relu', padding='same', name='conv1_2')(conv1_1)
    conv1_2 = BatchNormalization()(conv1_2)
    pool1_1 = MaxPooling2D(pool_size=(2, 2), name='pool1_1')(conv1_2)
    drop1_1 = Dropout(0.3, name='drop1_1')(pool1_1)

    # 2nd Block
    conv2_1 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_1')(drop1_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', name='conv2_2')(conv2_1)
    conv2_2 = BatchNormalization()(conv2_2)
    pool2_1 = MaxPooling2D(pool_size=(2, 2), name='pool2_1')(conv2_2)
    drop2_1 = Dropout(0.3, name='drop2_1')(pool2_1)

    # Adding spatial attention after the second block
    attention_2 = spatial_attention(drop2_1)

    # 3rd Block
    conv3_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_1')(attention_2)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_2 = BatchNormalization()(conv3_2)
    pool3_1 = MaxPooling2D(pool_size=(2, 2), name='pool3_1')(conv3_2)
    drop3_1 = Dropout(0.3, name='drop3_1')(pool3_1)

    # 4th Block
    conv4_1 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_1')(drop3_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Conv2D(256, kernel_size=3, activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_2 = BatchNormalization()(conv4_2)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name='pool4_1')(conv4_2)
    drop4_1 = Dropout(0.3, name='drop4_1')(pool4_1)

    # 5th Block
    conv5_1 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_1')(drop4_1)
    conv5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(512, kernel_size=3, activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_2 = BatchNormalization()(conv5_2)
    pool5_1 = MaxPooling2D(pool_size=(2, 2), name='pool5_1')(conv5_2)
    drop5_1 = Dropout(0.3, name='drop5_1')(pool5_1)

    # Flatten and Output Layer
    flatten = Flatten(name='flatten')(drop5_1)
    output = Dense(num_classes, activation='softmax', name='output')(flatten)

    # Create Model
    model = Model(inputs=visible, outputs=output)
    
    return model

# Initialize the model
model = FER_Model()

# Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping, model checkpoint, and reducing learning rate on plateau
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(train_flow,
                    validation_data=val_flow,
                    epochs=20,
                    batch_size=64,
                    callbacks=[early_stop, checkpoint, lr_scheduler])

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(testgen.flow(X_test, y_test, batch_size=batch_size), steps=len(X_test) // batch_size)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Log performance metrics at the final epoch
final_epoch = len(history.history['accuracy']) - 1
print(f"\nMetrics for the final epoch ({final_epoch + 1}):")
print(f"Training Accuracy: {history.history['accuracy'][final_epoch]:.4f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][final_epoch]:.4f}")
print(f"Training Loss: {history.history['loss'][final_epoch]:.4f}")
print(f"Validation Loss: {history.history['val_loss'][final_epoch]:.4f}")

print("\nPlotting accuracy and loss curves...")


# Plotting accuracy and loss curves
plt.figure(figsize=(14, 7))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Model Accuracy Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot the confusion matrix
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_class)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Save the model architecture as .json file and weights as .h5 file
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")