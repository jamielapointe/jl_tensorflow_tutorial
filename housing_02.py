'''
Created on 26 May 2018

@author: jamie
'''

from   future_encoders import OneHotEncoder
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from   multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
from   pandas.plotting import scatter_matrix
from   scipy import stats
from   scipy.stats import randint
from   six.moves import urllib
from   shutil import unpack_archive
from   sklearn.base import (BaseEstimator, TransformerMixin)
from   sklearn.ensemble.forest import RandomForestRegressor
from   sklearn.externals import joblib
from   sklearn.metrics.regression import mean_squared_error
from   sklearn.model_selection import (StratifiedShuffleSplit, 
                                       GridSearchCV,
                                       RandomizedSearchCV)
from   sklearn.pipeline import (Pipeline, FeatureUnion)
from   sklearn.preprocessing.data import (StandardScaler)
from   sklearn.preprocessing.imputation import Imputer
import shutil
from   time import time
from sklearn.cross_validation import cross_val_score

# Set to None to be psuedo-random - enter an unsigned long to be repeatable
random_seed = 38749290
np.random.seed(random_seed)

pct_test_data = 0.2
n_iter_search = 1000
n_jobs        = cpu_count()
project_root_dir = os.path.abspath(os.path.dirname(__file__))
download_root = "https://github.com/ageron/handson-ml/raw/master"
housing_path  = os.path.join(project_root_dir, "datasets", "housing")
housing_url   = download_root + "/datasets/housing/housing.tgz"

# column indicies
rooms_ix      = 3 
bedrooms_ix   = 4 
population_ix = 5 
household_ix  = 6 

# Combined attributes adder transformation class
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
   def __init__(self, add_bedrooms_per_room = True):
      self.add_bedrooms_per_room = add_bedrooms_per_room
   
   def fit(self, X, y=None):
      return self # nothing to do here
   
   def transform(self, X, y=None):
      rooms_per_household      = X[:, rooms_ix     ] / X[:, household_ix]
      population_per_household = X[:, population_ix] / X[:, household_ix]
      
      if self.add_bedrooms_per_room:
         bedrooms_per_room     = X[:, bedrooms_ix]   / X[:, bedrooms_ix ]
         return np.c_[X, rooms_per_household, population_per_household, 
                      bedrooms_per_room]
      
      return np.c_[X, rooms_per_household, population_per_household]

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
   def __init__(self, attribute_names):
      self.attribute_names = attribute_names
   
   def fit(self, X, y=None):
      return self # nothing to do here
   
   def transform(self, X, y=None):
      return X[self.attribute_names].values

class Housing02(object):
   def __init__(self):      
      self.housing_data        = None
      self.num_attribs         = None
      self.cat_attribs         = None
      self.housing_num         = None
      self.num_attribs         = None
      self.num_pipeline        = None
      self.cat_pipeline        = None
      self.full_pipeline       = None
      self.final_model         = None

   # functions for housing data machine learning tutorial
   def fetch_housing_data(self, housing_url=housing_url, housing_path=housing_path, override=True):
      if os.path.isdir(housing_path):
         if not override:
            return
         shutil.rmtree(housing_path, ignore_errors=False)
      os.makedirs(housing_path,exist_ok=True)
      tgz_path = os.path.join(housing_path, "housing.tgz")
      urllib.request.urlretrieve(housing_url, tgz_path)
      unpack_archive(tgz_path, housing_path)
      
   def load_housing_data(self, housing_path=housing_path):
      csv_path = os.path.join(housing_path, "housing.csv")
      self.housing_data = pd.read_csv(csv_path)
      
   def load_saved_model(self, model_path=os.path.join(housing_path, "housing_model_pkl")):
      self.final_model = joblib.load(filename=model_path)
   
   def split_train_test(self, housing_data, test_ratio=pct_test_data):
      shuffled_indices = np.random.permutation(len(housing_data))
      test_set_size   = int(len(housing_data)*test_ratio)
      test_indices    = shuffled_indices[:test_set_size]
      train_indices   = shuffled_indices[test_set_size:]
      return housing_data.iloc[train_indices], housing_data.iloc[test_indices]
   
   def strat_split_train_test(self, test_ratio=pct_test_data):
      # create a temporary category which will represent discretized median_income to be used as strata
      # divide by 1.5 to limit the number of income categories for strata
      self.housing_data["income_cat"] = np.ceil(self.housing_data["median_income"] / 1.5)
      # Label those above 5 as 5 - this says leave alone income_cat <5.0; anything >= 5.0 set to 5.0 with changes made in-place
      self.housing_data["income_cat"].where(self.housing_data["income_cat"] < 5.0, 5.0, inplace=True)
      split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
      # there is only one split... so that is why this wierd looking logic actually works
      for train_index, test_index in split.split(self.housing_data, self.housing_data["income_cat"]):
         self.strat_train_set = self.housing_data.loc[train_index]
         self.strat_test_set  = self.housing_data.loc[test_index]
      # remove the temporary category
      for set_ in (self.strat_train_set, self.strat_test_set):
         set_.drop("income_cat", axis=1, inplace=True)
   
   def dataVisualization(self):
      # Show median house value vs population vs location scatter plot on top of a map of California
      ca_img = mpimg.imread(project_root_dir + '/images/end_to_end_project/california.png')
      self.housing_data.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7), 
              s=self.housing_data["population"]/100, label="Population", c="median_house_value",
              cmap=plt.get_cmap("jet"), colorbar=False, alpha=0.4)
      plt.imshow(ca_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))
      plt.ylabel("Latitude", fontsize=14)
      plt.xlabel("Longitude", fontsize=14)
      prices = self.housing_data["median_house_value"]
      tick_values = np.linspace(prices.min(), prices.max(), 11)
      cbar = plt.colorbar()
      cbar.ax.set_yticklabels(["${0}k".format(round(v/1000)) for v in tick_values], fontsize=14)
      cbar.set_label('Median House Value', fontsize=16)
      plt.legend(fontsize=16)
      plt.savefig("housing_prices_scatterplot")
      
      # Get the correlation matrix - can only do this since the dataset is fairly small
      corr_matrix = self.housing_data.corr()
      corr_matrix["median_house_value"].sort_values(ascending=False)
      attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
      scatter_matrix(self.housing_data[attributes], figsize=(12,8))
      plt.savefig("scatter_matrix_plot")
      
   def transform_data(self, housing_data):
      data             = housing_data.drop('median_house_value', axis=1)
      self.housing_num = data.select_dtypes(include=[np.number])
      self.num_attribs = list(self.housing_num)
      self.cat_attribs = list(data.select_dtypes(include=[np.object]))
      
      self.num_pipeline = Pipeline([
            ('selector'     , DataFrameSelector      (self.num_attribs )),
            ('imputer'      , Imputer                (strategy="median")),
            ('attribs_adder', CombinedAttributesAdder(                 )),
            ('std_caller'   , StandardScaler         (                 ))
         ])
      
      self.cat_pipeline = Pipeline([
            ('selector'     , DataFrameSelector      (self.cat_attribs )),
            ('cat_encoder'  , OneHotEncoder          (sparse=False     ))
         ])
      
      self.full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", self.num_pipeline),
            ("cat_pipeline", self.cat_pipeline)
         ])
   
   def train_data(self):
      housing_data = self.strat_train_set
      X = housing_data.drop("median_house_value", axis=1)
      y = housing_data["median_house_value"].values
      X_prepared = self.full_pipeline.fit_transform(X)
      
      # Note: you can play with different models here to select an algorithm
      #       here we are going to go with Random Forest and use a Randomized Search
      #       to optimize hyperparameters
      
      param_distributions ={
            'bootstrap'   : randint(0,   1),
            'n_estimators': randint(1, 100),
            'max_features': randint(1,   8)
         }
      
      forest_reg = RandomForestRegressor(random_state=random_seed)
      random_search = RandomizedSearchCV(estimator=forest_reg, param_distributions=param_distributions,
                                         n_iter=n_iter_search, scoring='neg_mean_squared_error',
                                         cv=5, n_jobs=n_jobs, random_state=random_seed)
      print('Training data...')
      start = time()
      random_search.fit(X_prepared, y)
      print("RandomizedSearchCV took %.2f seconds for %d candidates"
            " parameter settings." % ((time() - start), n_iter_search))
      negative_mse = random_search.best_score_
      rmse = np.sqrt(-negative_mse)
      print('Best RMSE for training set: {0}'.format(rmse))
      print('\n')
      print('Optimized Parameters:\n{0}\n'.format(random_search.best_params_))
      feature_importances = random_search.best_estimator_.feature_importances_
      extra_attribs       = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
      cat_encoder         = self.cat_pipeline.named_steps["cat_encoder"]
      cat_one_hot_attribs = list(cat_encoder.categories_[0])
      attributes          = self.num_attribs + extra_attribs + cat_one_hot_attribs
      sorted_feature_importances_labeld = sorted(zip(feature_importances, attributes), reverse=True)
      print(sorted_feature_importances_labeld)
      
      # Let's look at cross validation scores
      scores = cross_val_score(forest_reg, X_prepared, y, scoring="neg_mean_squared_error", cv=10)
      pd.Series(np.sqrt(-scores)).describe()
      
      self.final_model = random_search.best_estimator_
      
   def test_data(self):
      housing_data      = self.strat_test_set
      X                 = housing_data.drop("median_house_value", axis=1)
      y                 = housing_data["median_house_value"].copy()
      X_prepared        = self.full_pipeline.transform(X)
      final_predictions = self.final_model.predict(X_prepared)
      
      final_mse  = mean_squared_error(y, final_predictions)
      final_rmse = np.sqrt(final_mse)
      
      print('\n')
      print('final root mean squared error (RMSE):\n{0}\n'.format(final_rmse))
      
      confidence     = 0.95
      squared_errors = (final_predictions - y) ** 2
      mean           = squared_errors.mean()
      scale          = stats.sem(squared_errors)
      m              = len(squared_errors)
      interval95     = np.sqrt(stats.t.interval(confidence, m-1, loc=mean,scale=scale))
      print('95% confidence interval for the RMSE:\n{0}\n'.format(interval95))
      
   def run_training(self):
      self.fetch_housing_data(override=False)
      self.load_housing_data()
      self.dataVisualization()
      
      self.strat_split_train_test()
      self.transform_data(self.strat_train_set)
      
      self.train_data()
      self.test_data()
      
      # save off the best model
      joblib.dump(self.final_model, os.path.join(housing_path, "housing_model.pkl"))
      
      # show all plots
      plt.show()
      
   def run_model(self):
      # load additional data
      self.load_housing_data()
      self.load_saved_model()
      self.production_data()
    
   
   
def test_housing02():
   housing = Housing02()
   housing.run_training()

if __name__ == "__main__":
   test_housing02()
   
   
   
   