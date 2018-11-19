'''
Created on 20 Oct 2018

@author: jamie
'''

from   datetime              import datetime
import matplotlib
import matplotlib.pyplot     as plt
import numpy                 as np
from   sklearn.datasets      import fetch_california_housing
from   sklearn.datasets      import make_moons
from   sklearn.preprocessing import StandardScaler
import tensorflow            as tf

def linear_regression_00():
   housing = fetch_california_housing()
   m, n = housing.data.shape
   housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
   
   X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
   y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
   XT = tf.transpose(X)
   theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
   
   with tf.Session() as sess:
      theta_value = theta.eval()
      
   print('Finished Linear Regression 00')
   
def gradient_descent_02():
   # get the data
   housing = fetch_california_housing()
   m, n = housing.data.shape
   
   # First scale the inputs - could use TF; however, just use scikit-learn for now
   scaler = StandardScaler()
   scaled_housing_data = scaler.fit_transform(housing.data)
   scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
   
   n_epochs = 1000
   learning_rate = 0.01
   
   X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
   y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
   theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
   y_pred = tf.matmul(X, theta, name="predictions")
   error = y_pred - y
   mse = tf.reduce_mean(tf.square(error), name="mse")
   
   optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=0.9)
   training_op = optimizer.minimize(mse)
   
   init = tf.global_variables_initializer()

   with tf.Session() as sess:
      sess.run(init)
   
      for epoch in range(n_epochs):
         if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
         sess.run(training_op)
       
      best_theta = theta.eval()
   
   print("Best theta:")
   print(best_theta)
   
def gradient_descent_03():
   now         = datetime.utcnow().strftime("%Y%m%d%H%M%S")
   root_logdir = "tf_logs"
   logdir      = "{}/run-{}/".format(root_logdir, now)
   
   # get the data
   housing = fetch_california_housing()
   m, n    = housing.data.shape
   
   # First scale the inputs - could use TF; however, just use scikit-learn for now
   scaler                        = StandardScaler()
   scaled_housing_data           = scaler.fit_transform(housing.data)
   scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
   
   X      = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
   y      = tf.placeholder(tf.float32, shape=(None, 1  ), name="y")
   theta  = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
   y_pred = tf.matmul(X, theta, name="predictions")
   error  = y_pred - y
   mse    = tf.reduce_mean(tf.square(error), name="mse")
   
   learning_rate = 0.01
   optimizer     = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
   training_op   = optimizer.minimize(mse)
   
   init          = tf.global_variables_initializer()
   
   mse_summary   = tf.summary.scalar('MSE', mse)
   file_writer   = tf.summary.FileWriter(logdir, tf.get_default_graph())
   
   n_epochs      = 10
   batch_size    = 100
   n_batches     = int(np.ceil(m / batch_size))
   
   def fetch_batch(epoch, batch_index, batch_size):
      np.random.seed(epoch * n_batches + batch_index)
      
      indices = np.random.randint(m, size=batch_size)  
      X_batch = scaled_housing_data_plus_bias[indices] 
      y_batch = housing.target.reshape(-1, 1)[indices] 
      
      return X_batch, y_batch

   with tf.Session() as sess:
      sess.run(init)
   
      for epoch in range(n_epochs):
         for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
               summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
               step = epoch * n_batches + batch_index
               file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
       
      best_theta = theta.eval()
   
   file_writer.close()
   
   print("Best theta:")
   print(best_theta)
   
def gradient_descent_04():
   now         = datetime.utcnow().strftime("%Y%m%d%H%M%S")
   root_logdir = "tf_logs"
   logdir      = "{}/run-{}/".format(root_logdir, now)
   
   # get the data
   housing = fetch_california_housing()
   m, n    = housing.data.shape
   
   # First scale the inputs - could use TF; however, just use scikit-learn for now
   scaler                        = StandardScaler()
   scaled_housing_data           = scaler.fit_transform(housing.data)
   scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
   
   X      = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
   y      = tf.placeholder(tf.float32, shape=(None, 1  ), name="y")
   theta  = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
   y_pred = tf.matmul(X, theta, name="predictions")
   
   with tf.name_scope('loss') as scope:
      error  = y_pred - y
      mse    = tf.reduce_mean(tf.square(error), name="mse")
   
   learning_rate = 0.01
   optimizer     = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
   training_op   = optimizer.minimize(mse)
   
   init          = tf.global_variables_initializer()
   
   mse_summary   = tf.summary.scalar('MSE', mse)
   file_writer   = tf.summary.FileWriter(logdir, tf.get_default_graph())
   
   n_epochs      = 10
   batch_size    = 100
   n_batches     = int(np.ceil(m / batch_size))
   
   def fetch_batch(epoch, batch_index, batch_size):
      np.random.seed(epoch * n_batches + batch_index)
      
      indices = np.random.randint(m, size=batch_size)  
      X_batch = scaled_housing_data_plus_bias[indices] 
      y_batch = housing.target.reshape(-1, 1)[indices] 
      
      return X_batch, y_batch

   with tf.Session() as sess:
      sess.run(init)
   
      for epoch in range(n_epochs):
         for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
               summary_str   = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
               step          = epoch * n_batches + batch_index
               file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
       
      best_theta = theta.eval()
   
   file_writer.flush()
   file_writer.close()
   print("Best theta:")
   print(best_theta)
   print(error.op.name)
   print(mse.op.name)
   
   
def logistics_regression():
   m                = 1000
   X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
   
   plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
   plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
   plt.legend()
   plt.show()
   
   X_moons_with_bias     = np.c_[np.ones((m, 1)), X_moons]
   y_moons_column_vector = y_moons.reshape(-1, 1)
   
   test_ratio = 0.2
   test_size  = int(m * test_ratio)
   X_train    = X_moons_with_bias[:-test_size]
   X_test     = X_moons_with_bias[-test_size:]
   y_train    = y_moons_column_vector[:-test_size]
   y_test     = y_moons_column_vector[-test_size:]
   
   def random_batch(X_train, y_train, batch_size):
      rnd_indices = np.random.randint(0, len(X_train), batch_size)
      X_batch = X_train[rnd_indices]
      y_batch = y_train[rnd_indices]
      return X_batch, y_batch
   
      
if __name__ == '__main__':
   logistics_regression()
