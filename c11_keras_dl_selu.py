import tensorflow as tf
from   tensorflow              import keras
from   tensorflow.python.ops   import math_ops
from   tensorflow.python.keras import backend as K
import numpy as np
from   sklearn.metrics         import (precision_score, 
                                       recall_score, 
                                       confusion_matrix)

def sparse_categorical_accuracy(y_true, y_pred):
  y_true = math_ops.reduce_max(y_true, axis=-1)
  y_pred = math_ops.argmax(y_pred, axis=-1)

  # If the expected labels are float, we need to cast the int returned by
  # argmax to compare.
  if K.dtype(y_true) == K.floatx():
    y_pred = math_ops.cast(y_pred, K.floatx())

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())

def as_keras_metric(method):
  import functools
  @functools.wraps(method)
  def wrapper(self, args, **kwargs):
    """ Wrapper for turning tensorflow metrics into keras metrics """
    value, update_op = method(self, args, **kwargs)
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
      value = tf.identity(value)
    return value
  return wrapper
     
class Keras_DL_SELU(object):
  width    = 28
  height   = 28
  channels =  1
  
  def __init__(self):
    self.X_train                               = None
    self.X_test                                = None
    self.X_valid                               = None
    self.y_train                               = None
    self.y_test                                = None
    self.y_valid                               = None
    
    self.X_train_scaled                        = None
    self.X_valid_scaled                        = None
    self.X_test_scaled                         = None
    
    self.model                                 = None
    
  def run(self):
    self.preprocess_data()
    self.run_training()
    
  def preprocess_data(self):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() 
    self.X_valid = train_images[55000:,:,:]   / 255.
    self.y_valid = train_labels[55000:]
    self.X_train = train_images[0:55000,:,:]  / 255.
    self.y_train = train_labels[0:55000]
    self.X_test  = test_images                / 255.
    self.y_test  = test_labels
    
    self.X_valid = self.X_valid.reshape(-1, self.width*self.height*self.channels)
    self.X_train = self.X_train.reshape(-1, self.width*self.height*self.channels)
    self.X_test  = self.X_test.reshape(-1, self.width*self.height*self.channels)
    
    means               = self.X_train.mean(axis=0, keepdims=True)
    stds                = self.X_train.std (axis=0, keepdims=True)
    self.X_train_scaled = (self.X_train - means) / stds
    self.X_valid_scaled = (self.X_valid - means) / stds
    self.X_test_scaled  = (self.X_test  - means) / stds
    
  def setup_sim_layer(self):
    self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense( 10, activation=tf.nn.softmax)
      ])
    
  def setup_mnist_relu_layer(self):
    data_format = 'channels_first'
    self.model = keras.Sequential([
        keras.layers.Reshape(
            target_shape = (1, 28, 28),
            input_shape  = (28*28,)
          ),
        keras.layers.Conv2D(
            32,
            5,
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu
          ),
        keras.layers.MaxPooling2D(
            (2,2),
            (2,2),
            padding='same',
            data_format = data_format
          ),
        keras.layers.Conv2D(
            64,
            5,
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu
          ),
        keras.layers.MaxPooling2D(
            (2,2),
            (2,2),
            padding='same',
            data_format = data_format
          ),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10)
      ])
    
  def setup_simple_selu_layer(self):
    self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.AlphaDropout(rate=0.2),
        keras.layers.Dense(128, activation=tf.nn.selu, 
                           kernel_initializer='lecun_normal'),
        keras.layers.Dense( 10)
      ])
    
    self.X_train = self.X_train_scaled
    self.X_valid = self.X_valid_scaled
    self.X_test  = self.X_test_scaled
    
  def run_training(self):
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    
#     self.setup_sim_layer()
#     self.setup_simple_selu_layer()
    self.setup_mnist_relu_layer()
    
    self.model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])#, precision, recall])
        
    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 5 epochs
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
      ]
    
    self.model.fit(self.X_train, self.y_train, epochs=400, batch_size=50,
                   callbacks=callbacks, 
                   validation_data=(self.X_valid, self.y_valid))
    
    test_loss, test_acc = \
          self.model.evaluate(self.X_test, self.y_test, batch_size=50)
    
    y_pred = self.model.predict(self.X_test, batch_size=50)
    conf_matrix = confusion_matrix(np.asarray(self.y_test), 
                                   np.asarray(y_pred.argmax(axis=1)))
    test_prec = precision_score(np.asarray(self.y_test), 
                                np.asarray(y_pred.argmax(axis=1)), 
                                average=None)
    test_recall = recall_score (np.asarray(self.y_test), 
                                np.asarray(y_pred.argmax(axis=1)), 
                                average=None)
    

    print('')
    print('Test Loss     :', test_loss)
    print('Test accuracy :', test_acc)
    print('Test precision:', test_prec)
    print('Test recall   :', test_recall)
    print('')
    print('Confusion Matrix:')
    print(conf_matrix)
    
if __name__ == '__main__':
  keras_dl_selu = Keras_DL_SELU()
  keras_dl_selu.run()
    