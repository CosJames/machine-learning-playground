import tensorboard
import tensorflow as tf;
import numpy as np;
import os;
import cv2; # type: ignore
import imghdr;
import matplotlib.pyplot as plt; # type: ignore
import matplotlib.image as mpimg; # type: ignore
# Disable Tensorflow to use all memory on gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
# Get cpus
cpus = tf.config.experimental.list_physical_devices('CPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
        
# Remove dodgy images
data_dir = 'data'
dogs_directory = os.listdir(os.path.join(data_dir, "puppies"))
cats_directory = os.listdir(os.path.join(data_dir, "cats"))
image_exts = ['jpeg', 'jpg', 'bmp', 'png']


# Loop on every image on data directory
i = cv2.imread(os.path.join('data', 'puppies', '0.jpg'))
# Remove dodgy images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            image = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print("Image not in ext list {}".format(image_class))
                os.remove(image_path)
        except Exception as e:
            print("Issues".format(image_class))

# Build an image dataset for you
# You can adjust batch size for GPU optimization on the second argument
data = tf.keras.utils.image_dataset_from_directory('data', batch_size=16)
# A numpy is a multidimensional array of object that is used for scientific calculation in python
data = data.map(lambda x,y: (x / 255, y)) # type: ignore
data_iterator = data.as_numpy_iterator() # type: ignore
batch = data_iterator.next()

batch[0].min()

train_size = int(len(data)* .7)
val_size = int(len(data)* .2) + 1
test_size = int(len(data)* .1) + 1

sum = train_size + val_size + test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Keras API
models = tf.keras.models
layers = tf.keras.layers

model = models.Sequential([
    layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), 1, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, (3, 3), 1, activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(tf.optimizers.Adam(), loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
summary = model.summary()
print(summary)

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
# print(hist.history)