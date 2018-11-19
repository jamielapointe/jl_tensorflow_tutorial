'''
Created on 4 Nov 2018

@author: jamie
'''

from   collections                import deque
from   copy                       import copy
from   datetime                   import datetime
from   enum                       import Enum
from   enum                       import unique
# import matplotlib
import matplotlib.pyplot as plt
from   my_alpha_drop              import alpha_dropout
import numpy as np
import os
import tensorflow as tf
import shutil
      
unique_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      

@unique
class LayerForms(Enum):
   rectangular = 0
   conic       = 1 # start at number of nodes and geometricaly decay to the # layers in output layer
   number      = 2
      
class Hyperparameters(object):
   # set to last best values
   _data = {'num_hidden_units'  : 541,
            'num_hidden_layers' : 7,
            'learning_rate'     : 0.001,
            'keep_prob'         : 0.9013661357285545,
            'layer_form'        : LayerForms.conic.value}
   
   @property
   def num_hidden_units(self):
      return self._data['num_hidden_units']
   
   @property
   def num_hidden_layers(self):
      return self._data['num_hidden_layers']
   
   @property
   def learning_rate(self):
      return self._data['learning_rate']
   
   @property
   def keep_prob(self):
      return self._data['keep_prob']
   
   @property
   def layer_form(self):
      return self._data['layer_form']
   
   def random_update(self):
      self._data['learning_rate']     = 10 ** np.random.randint(-6, 1)
      self._data['num_hidden_units']  = np.random.randint(100, 1025)
      self._data['num_hidden_layers'] = np.random.randint(2, 33)
      self._data['layer_form']        = np.random.randint(0, LayerForms.number.value)
      self._data['keep_prob']         = (1.0-0.5) * np.random.random_sample() + 0.5
      
   def __copy__(self):
      newtype = type(self)()
      newtype._data = self._data.copy()
      return newtype
      
   def __str__(self, *args, **kwargs):
      return object.__str__(self._data, *args, **kwargs)

def reset_graph(seed=None):
   if not seed:
      seed = np.random.randint(1,2**32)
   tf.reset_default_graph()
   tf.set_random_seed(seed)
   np.random.seed(seed)
   return seed
   
def get_num_hidden_units(max_val, min_val, num_hidden_layers, 
                         hidden_layer_idx, layer_forms):
   '''
   Get the number of hidden units depending on layer form selected
   
   Parameters:
      max_val - this is the set number of hidden units per hidden layer
                if LayerForms.rectangular or the maximum number if conic
      min_val - this is the number of hidden units in the output layer
      num_hidden_layers - this the number of hidden layers in the model
      hidden_layer_idx  - this is the 0-based index of the current hidden
                          layer
      layer_forms       - this is the enumerated type of the LayerForm - 
                          Rectangular - every hidden layer as the exact same
                                        number of units (max_val)
                          Conic       - The first hidden layer starts with max_val
                                        units and through a geometric progression
                                        goes down to the number of units in the 
                                        output layer
   '''
   if layer_forms == LayerForms.rectangular.value:
      return max_val
   elif layer_forms == LayerForms.conic.value:
      a = min_val
      n = max_val
      b = num_hidden_layers
      r = (n/a)**(1/b)
      b = hidden_layer_idx
      return np.uint32(np.round(a*(r**b)))
   else:
      raise Exception('Unexpected layer_forms {0}'.format(layer_forms))
   
def shuffle_batch(X, y, batch_size):
   rnd_idx   = np.random.permutation(len(X))
   n_batches = len(X) // batch_size
   for batch_idx in np.array_split(rnd_idx, n_batches):
      X_batch, y_batch = X[batch_idx], y[batch_idx]
      yield X_batch, y_batch
      
def get_p_labels(labels):
   p_labels = list()
   total_num_labels = labels.shape[0]
   for i in range(0, len(unique_labels)):
      p_labels.append(labels[labels == unique_labels[i]].shape[0] / total_num_labels)
   return p_labels
   
plt.rcParams['axes.labelsize' ] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

random_seed = reset_graph()

hyperparameters = Hyperparameters()

n_inputs                = 28 * 28  # MNIST
n_outputs               = 10
num_random_tuning_tries = 20

n_epochs                = 4000
batch_size              = 150
early_stop_window_len   = 10 # median filter kernel size

now           = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir   = os.path.join("tf_logs", __name__)
logdir        = "{0}/run-{1}-{2}/".format(root_logdir, 'c11_dl_selu_run0', now)

best_median_validation_accuracy_ht     = 0.
best_hyperparameters                   = copy(hyperparameters)

saved_model_dir = os.path.join(os.getcwd(), 'saved_models')
if os.path.exists(saved_model_dir):
   shutil.rmtree(saved_model_dir, ignore_errors=True)
model_builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)

export_dir = os.path.join(os.getcwd(), 'saved_checkpoints')
os.makedirs(export_dir, mode=0o775, exist_ok=True)
early_stop_model_name = os.path.join(export_dir, 'my_model.ckpt')
early_stop_graph_name = os.path.join(export_dir, 'my_model.ckpt.meta')
final_model_name = os.path.join(export_dir, 'my_model_final.ckpt')
final_graph_name = os.path.join(export_dir, 'my_model_final.ckpt.meta')
   
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train                              = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test                               = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train                              = y_train.astype(np.int32)
y_test                               = y_test.astype(np.int32)
X_valid, X_train                     = X_train[:5000], X_train[5000:]
y_valid, y_train                     = y_train[:5000], y_train[5000:]

# determine the percentage of labels in the training, validation, and test sets
p_labels_train = np.float32(get_p_labels(y_train))
p_labels_valid = np.float32(get_p_labels(y_valid))

for hyperparameter_tuning_epoch in range(0,num_random_tuning_tries):
   tf.reset_default_graph()
   
   X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
   y = tf.placeholder(tf.int64  , shape=(None          ), name="y")

   training = tf.placeholder_with_default(False, shape=(), name='training')
   X_drop   = alpha_dropout(X, hyperparameters.keep_prob, training=training, name='alpha_dropout')
   
   with tf.name_scope("dnn"):
      num_hidden_units = get_num_hidden_units(hyperparameters.num_hidden_units, 
                                              n_outputs, 
                                              hyperparameters.num_hidden_layers, 
                                              hyperparameters.num_hidden_layers, 
                                              hyperparameters.layer_form)
      input_layer = tf.layers.dense(X_drop , num_hidden_units, activation=tf.nn.selu, name="input_layer") # input layer
      
      last_hidden = input_layer
      for i in range(1, hyperparameters.num_hidden_layers):
         num_hidden_units = get_num_hidden_units(hyperparameters.num_hidden_units, 
                                                 n_outputs, 
                                                 hyperparameters.num_hidden_layers, 
                                                 hyperparameters.num_hidden_layers-i, 
                                                 hyperparameters.layer_form)
         tmp_hidden = tf.layers.dense(last_hidden, num_hidden_units, activation=tf.nn.selu, name="hidden_{0}".format(i))
         last_hidden = tmp_hidden
      logits  = tf.layers.dense(last_hidden, n_outputs,                        name="logits") # uses a linear activation
   
   with tf.name_scope("loss"):
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
      loss     = tf.reduce_mean(xentropy, name="loss") 
      
   with tf.name_scope("train"):
      optimizer   = tf.train.AdamOptimizer(hyperparameters.learning_rate)
      training_op = optimizer.minimize(loss)
      
   with tf.name_scope("eval"):
      predictions = {
          'classes':       tf.argmax(input=logits, axis=1, name='classes'),
          'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
        }
      # TODO at precision/recall for other labels
      accuracy_metric, accuracy_metric_update = \
            tf.metrics.accuracy(labels=y, predictions=predictions['classes'], name='accuracy_metric')
      precision_metric, precision_metric_update = \
            tf.metrics.precision(labels=y, predictions=predictions['classes'], name='precision_metric')
      recall_metric, recall_metric_update = \
            tf.metrics.recall(labels=y, predictions=predictions['classes'], name='recall_metric')
      confustion_op = \
            tf.confusion_matrix(labels=y, predictions=predictions['classes'], num_classes=n_outputs, name='confustion_matrix')
      
   tf.summary.scalar('accuracy_metric' , accuracy_metric)
   tf.summary.scalar('precision_metric', precision_metric)
   tf.summary.scalar('recall_metric'   , recall_metric   )
      
   init_global  = tf.global_variables_initializer()
   init_local   = tf.local_variables_initializer()
   saver        = tf.train.Saver()
   
   # scale the input for use with SELU (0 mean, 1 stdev)
   means          = X_train.mean(axis=0, keepdims=True)
   stds           = X_train.std(axis=0, keepdims=True) + 1e-10 # prevents div by 0
   X_valid_scaled = (X_valid - means) / stds
   
   merged = tf.summary.merge_all()
   
   early_stop_window    = deque(maxlen=early_stop_window_len)
   best_median_validation_accuracy     = 0.
   early_stop_count     = 0
   already_saved        = False
   
   with tf.Session() as sess:
      did_load = False
      if hyperparameter_tuning_epoch == 0:
         try:
            # load if it exists
            saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
            X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local = tf.get_collection("my_important_ops")
            tf.get_default_graph().clear_collection("my_important_ops")
            saver.restore(sess, final_model_name)
            sess.run(init_local)
            did_load = True
         except:
            pass
      if not did_load:
         init_global.run()
         init_local.run()
      did_early_stop = False
      print("Current Hyperparameters  : ", hyperparameters)
      with tf.summary.FileWriter(logdir, sess.graph) as graph_writer:
         for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
               means          = X_batch.mean(axis=0, keepdims=True)
               stds           = X_batch.std(axis=0, keepdims=True) + 1e-10 # prevents div by 0
               X_batch_scaled = (X_batch - means) / stds
               sess.run([training_op], feed_dict={X: X_batch_scaled, y: y_batch, training: True})
               
            # get validate set metrics
            [_] = \
                  sess.run([accuracy_metric_update], 
                           feed_dict={X: X_valid_scaled, y: y_valid, training: False})
            [accuracy_validation_metric, summary] = \
                  sess.run([accuracy_metric, merged])
            
            # reset metrics after running for this epoch
            sess.run(init_local)
            
            # test for early stop
            early_stop_window.append(accuracy_validation_metric)
            median_validation_accuracy = np.median(early_stop_window)
            if median_validation_accuracy > best_median_validation_accuracy:
               early_stop_count  = 0
               # save off the best model so far
               saver.save(sess, early_stop_model_name)
               did_early_stop = True
               best_median_validation_accuracy = median_validation_accuracy
            else:
               early_stop_count += 1
               
            if epoch % 5 == 0:
               print(epoch, " Validation accuracy: ", accuracy_validation_metric)
               graph_writer.add_summary(summary, epoch)
               
            if early_stop_count >= early_stop_window_len:
               print('Stopping early at epoch {}'.format(epoch))
               break
      
      if did_early_stop:
         # restore earl-stop model
         saver.restore(sess, early_stop_model_name)
            
      if best_median_validation_accuracy > best_median_validation_accuracy_ht:
         best_median_validation_accuracy_ht = best_median_validation_accuracy
         # save out our last best validation model
         # save all of the variables & ops:
         for op in (X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local):
            tf.add_to_collection("my_important_ops", op)
         saver.save(sess, final_model_name)
         tf.get_default_graph().clear_collection("my_important_ops")
         # save best model hyperparameters
         best_hyperparameters                   = copy(hyperparameters)
    
   hyperparameters.random_update()

tf.reset_default_graph()
saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local = tf.get_collection("my_important_ops")
            
with tf.Session() as sess:
   saver.restore(sess, final_model_name)
   # ideally here we would test the accuracy on the validation AND test sets
   sess.run(init_local)
   sess.run([accuracy_metric_update, precision_metric_update, recall_metric_update], 
                  feed_dict={X: X_valid_scaled, y: y_valid, training: False})
   [confusion_matrix] = sess.run([confustion_op], 
                                 feed_dict={X: X_valid_scaled, y: y_valid, training: False})
   [accuracy_metric_validation, precision_metric_validation, recall_metric_validation] = \
                  sess.run([accuracy_metric, precision_metric, recall_metric])
   f1_validation = 2 * ((precision_metric_validation*recall_metric_validation)/(precision_metric_validation + recall_metric_validation))
   print()
   print("p training labels          : ", p_labels_train)
   print("p validation labels        : ", p_labels_valid)
   print("Random Seed Used           : ", random_seed)
   print("Final Validation Accuracy  : ", accuracy_metric_validation)
   print("Final Validation Precision : ", precision_metric_validation)
   print("Final Validation Recall    : ", recall_metric_validation)
   print("Final Validation F1        : ", f1_validation)
   print()
   print("Confusion Matrix           : ")
   print(confusion_matrix)
   print()
   print("Best Hyperparameters       : ", best_hyperparameters)
   print()
   
#    # save model for export
#    model_builder.add_meta_graph_and_variables(sess,
#                                               [tf.saved_model.tag_constants.TRAINING],
#                                               strip_default_attrs=True)
# 
# model_builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
#    
# # actually save it
# model_builder.save()



