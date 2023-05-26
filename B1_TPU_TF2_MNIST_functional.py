'''
Data Engineering
'''

'''
D1. Import Libraries for Data Engineering
'''
import tensorflow as tf

'''
D2. Load MNIST data / Only for Toy Project
'''

# print(tf.__version__)
## MNIST Dataset #########################################################
mnist = tf.keras.datasets.mnist
# class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
##########################################################################

## Fashion MNIST Dataset #################################################
# mnist = tf.keras.datasets.fashion_mnist
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
##########################################################################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Change data type as float. If it is int type, it might cause error
'''
D3. Data Preprocessing
'''
# Normalizing
X_train, X_test = X_train / 255.0, X_test / 255.0

print(Y_train[0:10])
print(X_train.shape)

# One-Hot Encoding
from tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

'''
D4. EDA(? / Exploratory data analysis)
'''
import matplotlib.pyplot as plt

# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    # if you want to invert color, you can use 'gray_r'. this can be used only for MNIST, Fashion MNIST not cifar10
    # pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray_r'))
    
# show the figure
plt.show()

'''
D5. Build dataset
'''
batch_size = 100
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
# if it is RGB type, type shall be [image_size, image_size, 3]
# For MNIST or Fashion MNIST, it need to reshape

import numpy as np
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# It fills data as much as the input buffer_size and randomly samples and replaces it with new data.
# Perfect shuffling requires a buffer size greater than or equal to the total size of the data set.
# If you use a buffer_size smaller than the small number of data, 
# random shuffling occurs within the data as much as the initially set buffer_size.

shuffle_size = 100000

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train)).shuffle(shuffle_size).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, Y_test)).batch(batch_size)

'''
Model Engineering
'''

'''
M1. Import Libraries for Model Engineering
'''

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import Input
import numpy as np

'''
M2. Initialize Colab TPU
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print(tf.__version__)

import distutils
if distutils.version.LooseVersion(tf.__version__) <= '2.0':
    raise Exception('This notebook is compatible with TensorFlow 1.14 or higher, for TensorFlow 1.13 or lower please use the previous version at https://github.com/tensorflow/tpu/blob/r1.13/tools/colab/fashion_mnist.ipynb')

resolver = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
tf.config.experimental_connect_to_cluster(resolver)

# This is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.experimental.TPUStrategy(resolver)

'''
M3. Set Hyperparameters
'''

hidden_size = 256
output_dim = 10      # output layer dimensionality = num_classes
EPOCHS = 30
learning_rate = 0.001

'''
M4. Build NN model
'''
# in the case of Keras or TF2, type shall be [image_size, image_size, 1]
def create_model():
    inputs = Input(shape=(28, 28, 1))
    conv1      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(inputs)
    pool1      = MaxPool2D(padding='SAME')(conv1)
    conv2      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)(pool1)
    pool2      = MaxPool2D(padding='SAME')(conv2)
    pool3_flat = Flatten()(pool2)
    dense4     = Dense(units=128, activation=tf.nn.relu)(pool3_flat)
    drop4      = Dropout(rate=0.4)(dense4)
    
    logits = Dense(output_dim, activation='softmax')(drop4)
    
    model = Model(inputs=inputs, outputs=logits)

    return model

model = create_model()

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_shapes.png', show_shapes=True)

'''
M5. Optimizer
'''

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

'''
M6. Open "strategy.scope()"
'''
with strategy.scope():
    model = model
    
    '''
    M7. Define Loss Function
    '''
    @tf.function
    def loss_fn(model, images, labels):
        logits = model(images, training=True)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            y_pred=logits, y_true=labels, from_logits=True))    
        return loss   

    '''
    M8. Define train loop
    '''
    @tf.function
    def grad(model, images, labels):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, images, labels)
        return tape.gradient(loss, model.variables)

    @tf.function
    def train(model, images, labels):
        gradients = grad(model, images, labels)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    '''
    M9. Metrics - Accuracy
    '''

    @tf.function
    def evaluate(model, images, labels):
        logits = model(images, training=False)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

# checkpoint was not used in this implementation
checkpoint = tf.train.Checkpoint(cnn=model)

'''
M10. Define Episode / each step process
'''

import time
start_time = time.time()

for epoch in range(EPOCHS):
    train_loss     = 0.
    test_loss      = 0.
    train_accuracy = 0.
    test_accuracy  = 0.
    train_step     = 0
    test_step      = 0    
    
    for images, labels in train_ds:
        train(model, images, labels)
        #grads = grad(model, images, labels)                
        #optimizer.apply_gradients(zip(grads, model.variables))
        loss = loss_fn(model, images, labels)
        train_loss += loss
        acc = evaluate(model, images, labels)
        train_accuracy += acc
        train_step += 1
    train_loss = train_loss / train_step
    train_accuracy = train_accuracy / train_step
    template = 'epoch: {:>5,d}, loss: {:>2.4f}, accuracy: {:>2.3f} %'
    print (template.format(epoch+1,
                         train_loss,
                         train_accuracy*100))
            
'''
M11. Model evaluation
'''

for test_images, test_labels in test_ds:
    loss = loss_fn(model, test_images, test_labels)
    test_loss += loss
    acc = evaluate(model, test_images, test_labels)        
    test_accuracy += acc
    test_step += 1
test_loss = test_loss / test_step
test_accuracy = test_accuracy / test_step

"""
print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(train_loss), 
      'train accuracy = ', '{:.4f}'.format(train_accuracy), 
      'test accuracy = ', '{:.4f}'.format(test_accuracy))
"""
template = 'epoch: {:>5,d}, test loss: {:>2.4f}, test accuracy: {:>2.3f} %'
print (template.format(epoch+1,
                      test_loss,
                      test_accuracy*100))
    
finish_time = time.time()
print(int(finish_time - start_time),'sec.')
print('Learning Finished!')

