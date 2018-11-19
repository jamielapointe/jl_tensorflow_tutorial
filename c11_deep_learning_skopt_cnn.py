'''
Created on 4 Nov 2018

@author: jamie
'''

from   collections                import deque
from   enum                       import Enum
from   enum                       import unique
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import shutil
import skopt
from   skopt.utils                import use_named_args
      
unique_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      

@unique
class LayerForms(Enum):
   rectangular = 0
   conic       = 1 # start at number of nodes and geometricaly decay to the # layers in output layer
   number      = 2
   
class Model(object):
   X = None
   y = None
   training = None
   training_op = None
   accuracy_metric = None
   accuracy_metric_update = None
   precision_metric = None
   precision_metric_update = None
   recall_metric = None
   recall_metric_update = None
   confustion_op = None
   init_global = None
   init_local = None
   merged = None
   loss = None

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
   
dim_fmap0             = skopt.space.Integer(low=32  , high=128,                       name='fmap0')
dim_fmap1             = skopt.space.Integer(low=32  , high=128,                       name='fmap1')
dim_ksize             = skopt.space.Integer(low=2   , high=7,                         name='ksize')
dim_kstride           = skopt.space.Integer(low=1   , high=4,                         name='kstride')
# dim_pool_dropout      = skopt.space.Real   (low=0.0 , high=0.6 ,                      name='pool_dropout')
dim_fc_dropout        = skopt.space.Real   (low=0.0 , high=0.6 ,                      name='fc_dropout')
dim_fcunits           = skopt.space.Integer(low=128 , high=1024,                      name='fc_units')
dim_pool_padding      = skopt.space.Integer(low=0   , high=1,                         name='pool_padding')
dim_learning_rate     = skopt.space.Real   (low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dimensions = [dim_fmap0,
              dim_fmap1,
              dim_ksize,
              dim_kstride,
#               dim_pool_dropout,
              dim_fc_dropout,
              dim_fcunits,
              dim_pool_padding,
              dim_learning_rate]
   
# @use_named_args(dimensions=dimensions)
def log_dir_name(fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate):

   # The dir-name for the TensorBoard log-dir.
   # Insert all the hyper-parameters in the dir-name.
   log_dir = os.path.join("tf_logs", 
                          "c11_dl_skopt_cnn", 
                          "f0_{0}_f1_{1}_kern_{2}_stride_{3}_pdr_{4:0e}_fdr_{5:0e}_fu_{6}_pad_{7}_lr_{8:0e}".format(fmap0,
                                                                                                                    fmap1,
                                                                                                                    ksize,
                                                                                                                    kstride,
                                                                                                                    0.0,
                                                                                                                    fc_dropout,
                                                                                                                    fc_units,
                                                                                                                    pool_padding,
                                                                                                                    learning_rate))

   
   return log_dir

   
plt.rcParams['axes.labelsize' ] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

random_seed = reset_graph()

default_parameters = [32, 64, 5, 1, 0.4, 1024, 1, 0, 0.001]

hyperparameters = {'fmap0'             : default_parameters[0],
                   'fmap1'             : default_parameters[1],
                   'ksize'             : default_parameters[2],
                   'kstride'           : default_parameters[3],
                   'pool_dropout'      : default_parameters[4],
                   'fc_dropout'        : default_parameters[5],
                   'fc_units'          : default_parameters[6],
                   'do_flatten'        : default_parameters[7],
                   'pool_padding'      : default_parameters[8],
                   'learning_rate'        : default_parameters[9]}

height                  = 28
width                   = 28
channels                = 1
n_inputs                = height*width
n_outputs               = 10
num_random_tuning_tries = 50

n_epochs                    = 4000
batch_size                  = 50
early_stop_window_len       = 10 # median filter kernel size
check_interval              = 100
max_checks_without_progress = 20

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
X_valid = X_train[55000:,:,:]
y_valid = y_train[55000:]
X_train = X_train[0:55000,:,:]
y_train = y_train[0:55000]

y_train  = y_train.astype(np.int32)
y_test   = y_test .astype(np.int32)
y_valid  = y_valid.astype(np.int32)

num_training_examples = X_train.shape[0]

# scale the input for use with SELU (0 mean, 1 stdev)
means          = X_train.mean(axis=0, keepdims=True)
stds           = X_train.std (axis=0, keepdims=True) + 1e-10 # prevents div by 0
X_valid_scaled = (X_valid - means) / stds
X_test_scaled  = (X_test  - means) / stds

# determine the percentage of labels in the training, validation, and test sets
p_labels_train = np.float32(get_p_labels(y_train))
p_labels_valid = np.float32(get_p_labels(y_valid))
p_labels_test  = np.float32(get_p_labels(y_test))

# @use_named_args(dimensions=dimensions)
def create_model(fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate):
   model = Model()
   
   with tf.name_scope("inputs"):
      X = tf.placeholder(tf.float32, shape=(None, n_inputs),          name="X")
      X_reshaped = tf.reshape(X, shape=[-1, height, width, channels], name="X_respahed")
      y = tf.placeholder(tf.int64  , shape=(None          ),          name="y")
      training = tf.placeholder_with_default(False, shape=(),         name='training')
      
   with tf.name_scope("cnn"):
      conv1 = tf.layers.conv2d(X_reshaped, filters=fmap0, kernel_size=ksize,
                               strides=kstride, padding='same',
                               activation=tf.nn.relu, name="conv1")
      conv2 = tf.layers.conv2d(conv1     , filters=fmap1, kernel_size=ksize,
                               strides=kstride, padding='same',
                               activation=tf.nn.relu, name="conv2")
      
   with tf.name_scope("pool"):
      if pool_padding:
         padding='same'
      else:
         padding='valid'
      pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
      pool_flat      = tf.reshape(pool, shape=[-1, fmap1 * 14 * 14])
#       pool_flat_drop = tf.layers.dropout(pool_flat, pool_dropout, training=training)
      
   with tf.name_scope("fc1"):
      fc1      = tf.layers.dense(pool_flat, fc_units, activation=tf.nn.relu, name="fc1")
      fc1_drop = tf.layers.dropout(fc1, fc_dropout, training=training)
   
   with tf.name_scope("output"):
      logits  = tf.layers.dense(fc1_drop, n_outputs, name="output")
#       y_proba = tf.nn.softmax(logits, name="y_proba")
   
   with tf.name_scope("train"):
      xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
      loss     = tf.reduce_mean(xentropy, name="loss") 
      optimizer   = tf.train.AdamOptimizer(learning_rate, name='adam_opt')
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
   
   merged = tf.summary.merge_all()
   
   model.X = X
   model.y = y
   model.training = training
   model.training_op = training_op
   model.accuracy_metric = accuracy_metric
   model.accuracy_metric_update = accuracy_metric_update
   model.precision_metric = precision_metric
   model.precision_metric_update = precision_metric_update
   model.recall_metric = recall_metric
   model.recall_metric_update = recall_metric_update
   model.confustion_op = confustion_op
   model.init_global = init_global
   model.init_local = init_local
   model.merged = merged
   model.loss = loss
   
   return model

def get_model_params():
   gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
   return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
   gvar_names = list(model_params.keys())
   assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                 for gvar_name in gvar_names}
   init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
   feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
   tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
   
early_stop_window              = deque(maxlen=early_stop_window_len)
best_median_validation_loss_ht = np.infty
first_time_thru                = True
best_model_params              = None

@use_named_args(dimensions=dimensions)
def fitness(fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate):
   tf.reset_default_graph()

   global early_stop_window
   global best_median_validation_loss_ht
   global first_time_thru
   global best_model_params

   hyperparameters = dict(zip(hyperparameters.keys(), [fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate]))

   model   = create_model(fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate)
   logdir  = log_dir_name(fmap0, fmap1, ksize, kstride, fc_dropout, fc_units, pool_padding, learning_rate)

   best_median_validation_loss_early_stop     = np.infty
   
   with tf.Session() as sess:
      did_load = False
#       if first_time_thru:
#          try:
#             # load if it exists
#             saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
#             X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local = tf.get_collection("my_important_ops")
#             tf.get_default_graph().clear_collection("my_important_ops")
#             saver.restore(sess, final_model_name)
#             sess.run(init_local)
#             did_load = True
#             first_time_thru = False
#          except:
#             pass
      if not did_load:
         model.init_global.run()
         model.init_local.run()
         X = model.X
         y = model.y
         training = model.training
      saver   = tf.train.Saver()
      print("Current Hyperparameters  : ", hyperparameters)
      early_stop_count     = 0
      iteration            = 0
      with tf.summary.FileWriter(logdir, sess.graph) as graph_writer:
         for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
               X_batch_scaled   = (X_batch - means) / stds
               sess.run([model.training_op], feed_dict={X: X_batch_scaled, y: y_batch, training: True})
               
               # check for early stop
               if iteration % check_interval == 0:
                  loss_val = model.loss.eval(feed_dict={X: X_valid_scaled,
                                                        y: y_valid})
                  early_stop_window.append(loss_val)
                  median_loss_val = np.median(early_stop_window)
                  if median_loss_val < best_median_validation_loss_early_stop:
                     early_stop_count  = 0
                     best_median_validation_loss_early_stop = median_loss_val
                     best_model_params = get_model_params()
                  else:
                     early_stop_count += 1
               iteration += 1      
               
            # get validate set metrics
            sess.run([model.accuracy_metric_update], 
                     feed_dict={X: X_valid_scaled, y: y_valid, training: False})
            [accuracy_validation_metric, summary] = \
                  sess.run([model.accuracy_metric, model.merged])
            
            # reset metrics after running for this epoch
            sess.run(model.init_local)
               
            print("Epoch {0:5} Validation Accuracy: {1:6.4} Best Loss: {2:6.4}".format(epoch, accuracy_validation_metric, best_median_validation_loss_early_stop))
            graph_writer.add_summary(summary, epoch)
               
            if early_stop_count >= max_checks_without_progress:
               print('Stopping early at epoch {}'.format(epoch))
               break
      
      if best_model_params:
         print('Restoring early stop model')
         restore_model_params(best_model_params)
            
      if best_median_validation_loss_early_stop > best_median_validation_loss_ht:
         best_median_validation_loss_ht = best_median_validation_loss_early_stop
         # save out our last best validation model
         # save all of the variables & ops:
         for op in (model.X, model.y, model.training, model.training_op, 
                    model.accuracy_metric, model.accuracy_metric_update, 
                    model.precision_metric, model.precision_metric_update, 
                    model.recall_metric, model.recall_metric_update, 
                    model.confustion_op, model.init_local):
            tf.add_to_collection("my_important_ops", op)
         saver.save(sess, final_model_name)
         tf.get_default_graph().clear_collection("my_important_ops")
         # save best model hyperparameters
      
      return -best_median_validation_loss_early_stop # yes negative so skopt can find a minimum

# lots of options to play with here...
search_result = skopt.gp_minimize(func=fitness,
                                  dimensions=dimensions,
                                  n_calls=num_random_tuning_tries,
                                  x0=default_parameters)

tf.reset_default_graph()
saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local = tf.get_collection("my_important_ops")
            
with tf.Session() as sess:
   saver.restore(sess, final_model_name)
   
   # Test on validation set
   sess.run(init_local)
   sess.run([accuracy_metric_update, precision_metric_update, recall_metric_update], 
                  feed_dict={X: X_valid_scaled, y: y_valid, training: False})
   [confusion_matrix_validation] = sess.run([confustion_op], 
                                 feed_dict={X: X_valid_scaled, y: y_valid, training: False})
   [accuracy_metric_validation, precision_metric_validation, recall_metric_validation] = \
                  sess.run([accuracy_metric, precision_metric, recall_metric])
   f1_validation = 2 * ((precision_metric_validation*recall_metric_validation)/(precision_metric_validation + recall_metric_validation))
   
   # Test on test set
   sess.run(init_local)
   sess.run([accuracy_metric_update, precision_metric_update, recall_metric_update], 
                  feed_dict={X: X_test_scaled, y: y_test, training: False})
   [confusion_matrix_test]  = sess.run([confustion_op], 
                                       feed_dict={X: X_test_scaled, y: y_test, training: False})
   [accuracy_metric_test, precision_metric_test, recall_metric_test] = \
                  sess.run([accuracy_metric, precision_metric, recall_metric])
   f1_test = 2 * ((precision_metric_test*recall_metric_test)/(precision_metric_test + recall_metric_test))
   
   print()
   print("p training labels          : ", p_labels_train)
   print("p validation labels        : ", p_labels_valid)
   print("p test labels              : ", p_labels_test)
   print("Random Seed Used           : ", random_seed)
   print()
   print("Final Validation Accuracy  : ", accuracy_metric_validation)
   print("Final Validation Precision : ", precision_metric_validation)
   print("Final Validation Recall    : ", recall_metric_validation)
   print("Final Validation F1        : ", f1_validation)
   print()
   print("Confusion Matrix Valid     : ")
   print(confusion_matrix_validation)
   print()
   print()
   print("Final Test Accuracy        : ", accuracy_metric_test)
   print("Final Test Precision       : ", precision_metric_test)
   print("Final Test Recall          : ", recall_metric_test)
   print("Final Test F1              : ", f1_test)
   print()
   print("Confusion Matrix Test      : ")
   print(confusion_matrix_test)
   print()
   print("Best Hyperparameters       : ", dict(zip(hyperparameters.keys(), search_result.x)))
   print()
   print("Best negative fitness      : ", -search_result.fun)
   print()
   print("Optimization Results       : ")
   results_list = sorted(zip(search_result.func_vals, search_result.x_iters), reverse=True)
   for result in results_list:
      objective_func = result[0]
      hyperparm_val  = dict(zip(hyperparameters.keys(), result[1]))
      print('{0:6.4}, {1}'.format(objective_func, hyperparm_val))
   print()
   
#    # save model for export
#    model_builder.add_meta_graph_and_variables(sess,
#                                               [tf.saved_model.tag_constants.TRAINING],
#                                               strip_default_attrs=True)
#    model_builder.add_meta_graph([tf.saved_model.tag_constants.SERVING], strip_default_attrs=True)
#    
# # actually save it
# model_builder.save()



