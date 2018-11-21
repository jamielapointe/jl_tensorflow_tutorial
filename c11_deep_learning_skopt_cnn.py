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
label_dimensions = len(unique_labels)

@unique
class LayerForms(Enum):
   rectangular = 0
   conic       = 1 # start at number of nodes and geometricaly decay to the # layers in output layer
   number      = 2
   
class Model(object):
   X = None
   y = None
   y_proba = None
   training = None
   training_op = None
   accuracy_metric = None
   accuracy_metric_update = None
   precision_metric = None
   precision_metric_update = None
   recall_metric = None
   recall_metric_update = None
   loss_metric = None
   loss_metric_update = None
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
dim_pool_dropout      = skopt.space.Real   (low=0.0 , high=0.6 ,                      name='pool_dropout')
dim_fc_dropout        = skopt.space.Real   (low=0.0 , high=0.6 ,                      name='fc_dropout')
dim_fcunits           = skopt.space.Integer(low=128 , high=1024,                      name='fc_units')
dim_pool_padding      = skopt.space.Integer(low=0   , high=1,                         name='pool_padding')
dim_learning_rate     = skopt.space.Real   (low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
dimensions = [dim_fmap0,
              dim_fmap1,
              dim_ksize,
              dim_kstride,
              dim_pool_dropout,
              dim_fc_dropout,
              dim_fcunits,
              dim_pool_padding,
              dim_learning_rate]
   
default_parameters = [32, 64, 3, 1, 0.25, 0.0, 128, 1, 0.001]
hyperparameters = {'fmap0'             : default_parameters[0],
                   'fmap1'             : default_parameters[1],
                   'ksize'             : default_parameters[2],
                   'kstride'           : default_parameters[3],
                   'pool_dropout'      : default_parameters[4],
                   'fc_dropout'        : default_parameters[5],
                   'fc_units'          : default_parameters[6],
                   'pool_padding'      : default_parameters[7],
                   'learning_rate'     : default_parameters[8]}
   
# @use_named_args(dimensions=dimensions)
def log_dir_name(fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate):

   # The dir-name for the TensorBoard log-dir.
   # Insert all the hyper-parameters in the dir-name.
   log_dir = os.path.join("tf_logs", 
                          "c11_dl_skopt_cnn", 
                          "f0_{0}_f1_{1}_kern_{2}_stride_{3}_pdr_{4:0e}_fdr_{5:0e}_fu_{6}_pad_{7}_lr_{8:0e}".format(fmap0,
                                                                                                                    fmap1,
                                                                                                                    ksize,
                                                                                                                    kstride,
                                                                                                                    pool_dropout,
                                                                                                                    fc_dropout,
                                                                                                                    fc_units,
                                                                                                                    pool_padding,
                                                                                                                    learning_rate))

   
   return log_dir

   
plt.rcParams['axes.labelsize' ] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

random_seed = reset_graph()

height                  = 28
width                   = 28
channels                = 1
n_inputs                = height*width
n_outputs               = label_dimensions
num_random_tuning_tries = 50

n_epochs                    = 4000
batch_size                  = 32
early_stop_window_len       = 10 # median filter kernel size
check_interval              = 100
max_checks_without_progress = 20
worst_loss                  = 5.

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
# for this training to work right the number of training items should be a multiple of 64

X_valid = X_train[55040:,:,:]
y_valid = y_train[55040:]
X_train = X_train[0:55040,:,:]
y_train = y_train[0:55040]

X_train  = X_train.astype(np.float32).reshape(-1, width*height*channels)
X_test   = X_test .astype(np.float32).reshape(-1, width*height*channels)
X_valid  = X_valid.astype(np.float32).reshape(-1, width*height*channels)
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
def create_model(fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate):
   model = Model()
   
   with tf.name_scope("inputs"):
      X = tf.placeholder(tf.float32, shape=(None, n_inputs),          name="X")
      X_reshaped = tf.reshape(X, shape=[-1, height, width, channels], name="X_respahed")
      y = tf.placeholder(tf.int64  , shape=(None          ),          name="y")
      training = tf.placeholder_with_default(False, shape=(),         name='training')
   
   if pool_padding:
      padding='SAME'
   else:
      padding='VALID'
      
   with tf.name_scope("cnn"):
      conv1 = tf.layers.conv2d(X_reshaped, filters=fmap0, kernel_size=(ksize,ksize),
                               strides=(kstride,kstride), padding='SAME',
                               activation=tf.nn.relu, name="conv1")
      conv2 = tf.layers.conv2d(conv1     , filters=fmap1, kernel_size=(ksize,ksize),
                               strides=(kstride,kstride), padding='SAME',
                               activation=tf.nn.relu, name="conv2")
      
   with tf.name_scope("pool"):
      pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
      pool_flat      = tf.reshape(pool, shape=[-1, fmap1 * 14 * 14])
      pool_flat_drop = tf.layers.dropout(pool_flat, pool_dropout, training=training)
      
   with tf.name_scope("fc1"):
      fc1      = tf.layers.dense(pool_flat_drop, fc_units, activation=tf.nn.relu, name="fc1")
      fc1_drop = tf.layers.dropout(fc1, fc_dropout, training=training)
   
   with tf.name_scope("output"):
      logits  = tf.layers.dense(fc1_drop, n_outputs, name="output")
      y_proba = tf.nn.softmax(logits, name="y_proba")
   
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
      loss_metric, loss_metric_update = \
            tf.metrics.mean(values=loss, name='loss_metric')
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
      merged = tf.summary.merge_all()
      
   with tf.name_scope("init"):  
      init_global  = tf.global_variables_initializer()
      init_local   = tf.local_variables_initializer()
   
   model.X = X
   model.y = y
   model.y_proba = y_proba
   model.training = training
   model.training_op = training_op
   model.accuracy_metric = accuracy_metric
   model.accuracy_metric_update = accuracy_metric_update
   model.precision_metric = precision_metric
   model.precision_metric_update = precision_metric_update
   model.recall_metric = recall_metric
   model.recall_metric_update = recall_metric_update
   model.loss_metric = loss_metric
   model.loss_metric_update = loss_metric_update
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

best_median_validation_loss_ht = worst_loss
first_time_thru                = True

@use_named_args(dimensions=dimensions)
def fitness(fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate):
   tf.reset_default_graph()

   global best_median_validation_loss_ht
   global first_time_thru
   
   hyperparameters_fit = dict(zip(hyperparameters.keys(), [fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate]))

   model   = create_model(fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate)
   logdir  = log_dir_name(fmap0, fmap1, ksize, kstride, pool_dropout, fc_dropout, fc_units, pool_padding, learning_rate)
   
   with tf.Session() as sess:
      did_load = False
#       if first_time_thru:
#          try:
#             # load if it exists
#             saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
#             X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local, init_global, merged, loss = tf.get_collection("my_important_ops")
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
         X        = model.X
         y        = model.y
         training = model.training
      saver   = tf.train.Saver()
      print("Current Hyperparameters  : ", hyperparameters_fit)
      best_median_validation_loss_early_stop = worst_loss
      early_stop_count                       = 0
      best_model_params                      = None
      early_stop_window                      = deque(maxlen=early_stop_window_len)
      with tf.summary.FileWriter(logdir, sess.graph) as graph_writer:
         for epoch in range(n_epochs):
            
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
               X_batch_scaled   = (X_batch - means) / stds
               sess.run([model.training_op], feed_dict={X: X_batch_scaled, y: y_batch, training: True})
              
            for X_valid_batch, y_valid_batch in shuffle_batch(X_valid_scaled, y_valid, batch_size):
               sess.run([model.loss_metric_update, model.accuracy_metric_update], 
                     feed_dict={X: X_valid_batch, y: y_valid_batch, training: False})
            # check for early stop
            [accuracy_validation_metric, summary, loss_val] = sess.run([model.accuracy_metric, model.merged, model.loss_metric])
#             loss_val = model.loss.eval(feed_dict={X: X_valid_scaled,
#                                                      y: y_valid, training: False})
            
            if np.isinf(loss_val) or np.isnan(loss_val):
               print("Loss value is NAN - skipping this hyperparameter tuning session")
               return worst_loss
            
            early_stop_window.append(loss_val)
            median_loss_val = np.median(early_stop_window)
            if median_loss_val < best_median_validation_loss_early_stop:
               early_stop_count  = 0
               best_median_validation_loss_early_stop = median_loss_val
               best_model_params = get_model_params()
            else:
               early_stop_count += 1
            
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
            
      if best_median_validation_loss_early_stop < best_median_validation_loss_ht:
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
      
      return best_median_validation_loss_early_stop

# lots of options to play with here...
search_result = skopt.gp_minimize(func=fitness,
                                  dimensions=dimensions,
                                  n_calls=num_random_tuning_tries,
                                  x0=default_parameters)

tf.reset_default_graph()
saver           = tf.train.import_meta_graph(final_graph_name, clear_devices=True)
X, y, training, training_op, accuracy_metric, accuracy_metric_update, precision_metric, precision_metric_update, recall_metric, recall_metric_update, confustion_op, init_local, init_global, merged, loss = tf.get_collection("my_important_ops")
            
with tf.Session() as sess:
   saver.restore(sess, final_model_name)
   
   # Test on validation set
   sess.run(init_local)
   is_first = True
   for X_valid_batch, y_valid_batch in shuffle_batch(X_valid_scaled, y_valid, batch_size):
      sess.run([accuracy_metric_update, precision_metric_update, recall_metric_update], 
                     feed_dict={X: X_valid_batch, y: y_valid_batch, training: False})
      [confusion_matrix_validation_tmp] = sess.run([confustion_op], 
                                    feed_dict={X: X_valid_scaled, y: y_valid, training: False})
      if is_first:
         confusion_matrix_validation  = confusion_matrix_validation_tmp
         is_first = False
      else:
         confusion_matrix_validation += confusion_matrix_validation_tmp
   [accuracy_metric_validation, precision_metric_validation, recall_metric_validation] = \
                  sess.run([accuracy_metric, precision_metric, recall_metric])
   f1_validation = 2 * ((precision_metric_validation*recall_metric_validation)/(precision_metric_validation + recall_metric_validation))
   
   # Test on test set
   sess.run(init_local)
   is_first = True
   for X_test_batch, y_test_batch in shuffle_batch(X_test_scaled, y_test, batch_size):
      sess.run([accuracy_metric_update, precision_metric_update, recall_metric_update], 
                     feed_dict={X: X_test_scaled, y: y_test, training: False})
      [confusion_matrix_test_tmp]  = sess.run([confustion_op], 
                                          feed_dict={X: X_test_scaled, y: y_test, training: False})
      if is_first:
         confusion_matrix_test  = confusion_matrix_test_tmp
         is_first = False
      else:
         confusion_matrix_test += confusion_matrix_test_tmp
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
   results_list = sorted(zip(search_result.func_vals, search_result.x_iters))
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



