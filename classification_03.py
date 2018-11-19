'''
Created on 1 Jun 2018

@author: jamie
'''

from   fetch_mnist             import fetch_mnist
import matplotlib
import matplotlib.pyplot       as plt
import numpy as np
import os
from   sklearn.base            import (BaseEstimator, clone)
from   sklearn.datasets        import fetch_mldata
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.linear_model    import SGDClassifier
from   sklearn.metrics         import (precision_recall_curve, roc_curve,
                                       roc_auc_score)
from   sklearn.multiclass      import OneVsOneClassifier
from   sklearn.model_selection import (cross_val_predict, StratifiedKFold)

# Set to None to be psuedo-random - enter an unsigned long to be repeatable
random_seed = 38749290
np.random.seed(random_seed)

project_root_dir         = os.path.abspath(os.path.dirname(__file__))
chap_title               = 'classification'
classification_data_path = os.path.join(project_root_dir, 'datasets', chap_title)
classification_img_path  = os.path.join(project_root_dir, 'images', chap_title) 

# set some defaults for matplotlib
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
   plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
   plt.plot(thresholds, recalls[:-1],    "g-",  label="Recall",    linewidth=2)
   plt.xlabel("Threshold",      fontsize=16)
   plt.legend(loc="upper left", fontsize=16)
   plt.ylim([0, 1])
   
def plot_precision_vs_recall(precisions, recalls):
   plt.plot(recalls, precisions, "b-", linewidth=2)
   plt.xlabel("Recall", fontsize=16)
   plt.ylabel("Precision", fontsize=16)
   plt.axis([0, 1, 0, 1])
   
def plot_roc_curve(fpr, tpr, label=None):
   plt.plot(fpr, tpr, linewidth=2, label=label)
   plt.plot([0, 1], [0, 1], 'k--')
   plt.axis([0, 1, 0, 1])
   plt.xlabel('False Positive Rate', fontsize=16)
   plt.ylabel('True Positive Rate', fontsize=16)
   
def plot_confusion_matrix(matrix):
   """If you prefer color and a colorbar"""
   fig = plt.figure(figsize=(8,8))
   ax = fig.add_subplot(111)
   cax = ax.matshow(matrix)
   fig.colorbar(cax)

class Never5Classifier(BaseEstimator):
   def fit(self, X, y=None):
      pass
   def predct(self, X):
      return np.zeros((len(X), 1), dtype=bool)

class Classification03(object):
   def __init__(self):
      self.mnist   = None
      self.X       = None
      self.Y       = None
      self.X_train = None
      self.X_test  = None
      self.y_train = None
      self.y_test  = None
   
   def fetch_minst_data(self, mnist_dest_path=classification_data_path):
      fetch_mnist(mnist_dest_path)
      self.mnist = fetch_mldata('MNIST original', data_home=mnist_dest_path)
      
   def process_mnist(self):
      self.X       = self.mnist["data"]
      self.y       = self.mnist["target"]
      self.X_train = self.X[:60000]
      self.y_train = self.y[:60000]
      self.X_test  = self.X[60000:]
      self.y_test  = self.y[60000:]
      
      # shuffle the training set
      shuffle_index = np.random.permutation(60000)
      self.X_train  = self.X_train[shuffle_index]
      self.y_train  = self.y_train[shuffle_index]
      
   def save_image(self, fig_id, tight_layout=True):
      path = os.path.join(classification_img_path, fig_id + ".png")
      print("Saving Figure {0}".format(fig_id))
      if tight_layout:
         plt.tight_layout()
      plt.savefig(path, format='png', dpi=300)
      
   def plot_digits(self, instances, images_per_row=10, **options):
      size = 28
      images_per_row = min(len(instances), images_per_row)
      images = [instance.reshape(size,size) for instance in instances]
      n_rows = (len(instances)-1) // images_per_row + 1
      row_images = []
      n_empty = n_rows * images_per_row - len(instances)
      images.append(np.zeros((size, size*n_empty)))
      for row in range(n_rows):
         rimages = images[row * images_per_row : (row + 1) * images_per_row]
         row_images.append(np.concatenate(rimages, axis=1))
      image = np.concatenate(row_images, axis=0)
      plt.imshow(image, cmap=matplotlib.cm.binary, **options)
      plt.axis('off')
      
   def binary_classifier(self):
      y_train_5 = (self.y_train == 5)
      y_test_5  = (self.y_test  == 5)
      
      sgd_classifier = SGDClassifier(max_iter=5, random_state=random_seed)
      sgd_classifier.fit(self.X_train, y_train_5)
      
      skfolds = StratifiedKFold(n_splits=3, random_state = random_seed)
      
      for train_index, test_index in skfolds.split(self.X_train, y_train_5):
         clone_classifier = clone(sgd_classifier)
         X_train_folds    = self.X_train[train_index]
         y_train_folds    = (y_train_5[train_index])
         X_test_folds     = self.X_train[test_index]
         y_test_folds     = (y_train_5[test_index])
         
         clone_classifier.fit(X_train_folds, y_train_folds)
         y_pred = clone_classifier.predict(X_test_folds)
         n_correct = sum(y_pred == y_test_folds)
         
      y_scores = cross_val_predict(sgd_classifier, self.X_train, y_train_5, cv=3,
                                   method="decision_function")
      precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
      
      plt.figure(figsize=(8, 4))
      plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
      plt.xlim([-700000, 700000])
      self.save_image("precision_recall_vs_threshold_plot")
      plt.show()
      
      plt.figure(figsize=(8, 6))
      plot_precision_vs_recall(precisions, recalls)
      self.save_image("precision_vs_recall_plot")
      plt.show()
      
      fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
      
      plt.figure(figsize=(8, 6))
      plot_roc_curve(fpr, tpr)
      self.save_image("roc_curve_plot")
      plt.show()
      
      forest_clf = RandomForestClassifier(random_state=random_seed)
      y_probas_forest = cross_val_predict(forest_clf, self.X_train, y_train_5, cv=3,
                                          method="predict_proba")
      y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
      fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)
      
      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
      plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
      plt.legend(loc="lower right", fontsize=16)
      self.save_image("roc_curve_comparison_plot")
      plt.show()
      
   def multiclass_classifier(self):
      sgd_classifier = SGDClassifier(max_iter=5, random_state=random_seed)
      sgd_classifier.fit(self.X_train, self.y_train)
      ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
      ovo_clf.fit(self.X_train, self.y_train)
      forest_clf = RandomForestClassifier(random_state=random_seed)
      forest_clf.fit(self.X_train, self.y_train)
      
   def test_run(self):
      self.fetch_minst_data()
      self.process_mnist()
      
      plt.figure(figsize=(9,9))
      example_images = np.r_[self.X[:12000:600], self.X[13000:30600:600], self.X[30600:60000:590]]
      self.plot_digits(example_images, images_per_row=10)
      self.save_image("more_digits_plot")
      
      plt.show()
      
   def test_binary(self):   
      self.fetch_minst_data()
      self.process_mnist()
      self.binary_classifier()
      
   def test_multiclass(self):
      self.fetch_minst_data()
      self.process_mnist()
      self.multiclass_classifier()
      
def test_classification():
   classification = Classification03()
#    classification.test_run()
   classification.test_binary()
      
if __name__ == "__main__":
   test_classification()
      
