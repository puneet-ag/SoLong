import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Activation
from keras.layers.normalization import BatchNormalization


from keras.optimizers import SGD, RMSprop


import wandb
from wandb.keras import WandbCallback

run = wandb.init()
config = run.config

config.img_width = 299
config.img_height = 299
config.epochs = 10
config.batch_size = 64
config.num_classes = 8

train_dir = "../train"
test_dir = "../valid"

anno_classes = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT', 'NoF']

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

nb_train_samples = get_nb_files(train_dir)
nb_classes = len(glob.glob(train_dir + "/*"))
nb_valid_smaples = get_nb_files(test_dir)

def generators(anno_classes, preprocessing_function, img_width, img_height, batch_size=32, binary=False, shuffle=True,
               train_dir="../train", val_dir="../valid"):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for validation:
    # only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        classes = anno_classes,
        class_mode = 'categorical')

    validation_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        classes = anno_classes,
        class_mode = 'categorical')
    return train_generator, validation_generator

train_generator, validation_generator = generators(anno_classes, preprocess_input, config.img_width, config.img_height, config.batch_size)

conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

for layer in conv_base.layers[:277]:
    layer.trainable = False

for layer in conv_base.layers[277:]:
    layer.trainable = True
    
    
model = Sequential()
model.add(conv_base)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.6))
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(config.num_classes, activation="softmax"))

print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-5),
              metrics=['acc'])


model.fit_generator(
    train_generator,
    epochs=config.epochs,
    steps_per_epoch=nb_train_samples // config.batch_size,
    validation_data=validation_generator,
    validation_steps=nb_valid_smaples // config.batch_size,
    callbacks=[WandbCallback()])


#model.save('transfered.h5')