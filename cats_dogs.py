from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import TensorBoard
import sys
from matplotlib import pyplot
import datetime

def define_model():
    # model = VGG16(include_top=False, input_shape=(224,224,3))
    # for layer in model.layers:
    #     layer.trainable = False
    # flat1 = layers.Flatten()(model.layers[-1].output)
    # class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    # output = layers.Dense(1, activation='sigmoid')(class1)
    # model = keras.models.Model(inputs=model.inputs, outputs=output)

    model = ResNet50(include_top=False, input_shape=(224,224,3))
    for layer in model.layers:
        layer.trainable = False
    avgpool = layers.AveragePooling2D((7,7))(model.layers[-1].output)
    flat1 = layers.Flatten()(avgpool)
    # class1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = layers.Dense(1, activation='sigmoid')(flat1)
    model = keras.models.Model(inputs=model.inputs, outputs=output)
    
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def run():
    model = define_model()
    datagen = ImageDataGenerator(featurewise_center=True)
    datagen.mean = [123.68, 116.779, 103.939]
    train_it = datagen.flow_from_directory('dataset_cats_dogs/train/', class_mode='binary', batch_size=64, target_size=(224,224))
    test_it = datagen.flow_from_directory('dataset_cats_dogs/test/', class_mode='binary', batch_size=64, target_size=(224,224))

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps =len(test_it), epochs=10, verbose = 1, callbacks=[tensorboard_callback])

    _,acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
    summarize_diagnostics(history)

def summarize_diagnostics(history):
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')

    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

run();