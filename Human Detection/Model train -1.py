import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# Define hyperparameters
num_classes = 7  # Number of action classes in your dataset
input_shape = (320, 240, 3)  # Input shape for your video frames
batch_size = 32
epochs = 10

# Define data augmentation and preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Define data generators for training only
train_data_loader = train_datagen.flow_from_directory(
    'D:/Python Codes/Library/Train_frames',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained I3D model from TensorFlow Hub
i3d_model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=False),
    GlobalAveragePooling2D(),
    Dense(num_classes, activation='softmax')
])

# Compile the model with appropriate optimizer and loss function
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
i3d_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model without validation
history = i3d_model.fit(
    train_data_loader,
    epochs=epochs
)

# Save the trained model for later use
i3d_model.save('i3d_action_recognition_model.h5')

# You can also evaluate the model on a test dataset if available
# test_data_loader = train_datagen.flow_from_directory(
#     'D:/Python Codes/Library/Test',
#     target_size=input_shape[:2],
#     batch_size=batch_size,
#     class_mode='categorical'
# )

# test_loss, test_accuracy = i3d_model.evaluate(test_data_loader)
# print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
