import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import Model

import numpy as np

print(tf.__version__)

learning_rate = 0.001
training_epochs = 15
batch_size = 32

## MNIST Dataset #########################################################
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    
##########################################################################

train_images = train_images.astype(np.float32) / 255.
test_images  = test_images.astype(np.float32) / 255.
train_images = np.expand_dims(train_images, axis=-1)
test_images  = np.expand_dims(test_images, axis=-1)

# train_images, test_images = train_images / 255.0, test_images / 255.0

# 채널 차원을 추가합니다.
# train_images = train_images[..., tf.newaxis]
# test_images = test_images[..., tf.newaxis]
    
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
                buffer_size=10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1      = Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool1      = MaxPool2D(padding='SAME')
        self.conv2      = Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool2      = MaxPool2D(padding='SAME')
        self.conv3      = Conv2D(filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool3      = MaxPool2D(padding='SAME')
        self.pool3_flat = Flatten()
        self.dense4     = Dense(units=256, activation=tf.nn.relu)
        self.drop4      = Dropout(rate=0.4)
        self.dense5     = Dense(units=10, activation='softmax')
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

model = MNISTModel()

loss_object    = tf.keras.losses.SparseCategoricalCrossentropy()

optimizer      = tf.keras.optimizers.Adam()

train_loss     = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss      = tf.keras.metrics.Mean(name='test_loss')
test_accuracy  = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

# EPOCHS = 5

for epoch in range(training_epochs):
    for images, labels in train_dataset:
        train_step(images, labels)

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    template = 'epoch: {}, loss: {}, accuracy: {}, test loss: {}, test accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
