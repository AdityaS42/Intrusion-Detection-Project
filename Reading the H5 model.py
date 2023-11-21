import h5py
from tensorflow import keras
from tensorflow.keras.utils import plot_model

# Open the HDF5 file for reading
h5_file = h5py.File('D:/Python Codes/Test/Yolo/action_recognition_model.h5', 'r')

# Load the model architecture and weights
model = keras.models.load_model('action_recognition_model.h5')

# Visualize the model architecture as a diagram and save it to a file
#plot_model(model, to_file='model.png', show_shapes=True)
model.summary()


# Close the HDF5 file
h5_file.close()

# Now you can use the 'model' object for predictions or other tasks
