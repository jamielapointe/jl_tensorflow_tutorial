'''
Created on 4 Nov 2018

@author: jamie
'''

from   datetime import datetime
# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def reset_graph(seed=4838392):
   tf.reset_default_graph()
   tf.set_random_seed(seed)
   np.random.seed(seed)
   
def shuffle_batch(X, y, batch_size):
   rnd_idx   = np.random.permutation(len(X))
   n_batches = len(X) // batch_size
   for batch_idx in np.array_split(rnd_idx, n_batches):
      X_batch, y_batch = X[batch_idx], y[batch_idx]
      yield X_batch, y_batch
   
plt.rcParams['axes.labelsize' ] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

reset_graph()

n_inputs      = 28 * 28  # MNIST
n_hidden1     = 300
n_hidden2     = 100
n_outputs     = 10
learning_rate = 0.01
n_epochs      = 40
batch_size    = 50

now           = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir   = os.path.join("tf_logs", __name__)
logdir        = "{0}/run-{1}-{2}/".format(root_logdir, __name__, now)

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32  , shape=(None          ), name="y")

with tf.name_scope("dnn"):
   hidden1 = tf.layers.dense(X      , n_hidden1, activation=tf.nn.elu, name="hidden1")
   hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, name="hidden2")
   logits  = tf.layers.dense(hidden2, n_outputs,                        name="outputs") # uses a linear activation

with tf.name_scope("loss"):
   xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
   loss     = tf.reduce_mean(xentropy, name="loss") 
   
with tf.name_scope("train"):
   optimizer   = tf.train.GradientDescentOptimizer(learning_rate)
   training_op = optimizer.minimize(loss)
   
with tf.name_scope("eval"):
   correct  = tf.nn.in_top_k(logits, y, 1)
   accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

tf.summary.scalar('accuracy', accuracy)
   
init  = tf.global_variables_initializer()
saver = tf.train.Saver()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train                              = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test                               = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train                              = y_train.astype(np.int32)
y_test                               = y_test.astype(np.int32)
X_valid, X_train                     = X_train[:5000], X_train[5000:]
y_valid, y_train                     = y_train[:5000], y_train[5000:]

merged = tf.summary.merge_all()

with tf.Session() as sess:
   with tf.summary.FileWriter(logdir, sess.graph) as graph_writer:
      init.run()
      for epoch in range(n_epochs):
         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            summary, _ = sess.run([merged, training_op], feed_dict={X: X_batch, y: y_batch})
         if epoch % 5 == 0:
            acc_batch   = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_valid   = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            summary_str = "{0:3} Batch accuracy: {1:5.2} Validation accuracy: {2:5.2}".format(epoch, acc_batch, acc_valid)
            print(summary_str)
            graph_writer.add_summary(summary, epoch)
   
      save_path = saver.save(sess, "./my_model_final.ckpt")
      
      
      
      
