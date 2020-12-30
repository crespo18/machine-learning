#!/usr/bin/env python

import numpy as np
import xgboost as xgb
from sklearn.model_selection import  train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


class XgboostLinearModel():
    def __init__(self, model_name):
        self.model_name = model_name
        print('model name: ', self.model_name)

    def load_train_data(self, file_name):
        with open(file_name) as fp:
            data = np.loadtxt(fp, delimiter = ',', dtype = np.float32)
            return(data)

    def generate_xgb_train_data(self, data, train_y_idx, train_x_start_idx, train_x_end_idx):
        train_y = data[:,train_y_idx]
        train_x = data[:,train_x_start_idx:train_x_end_idx]
        train_x = preprocessing.scale(train_x)
        return([train_x, train_y])

    def train_xgb_model(self, train_x, train_y):
        clf = xgb.XGBClassifier(silent=0, learning_rate=0.3, min_child_weight=5, max_depth=6, gamma=0, subsample=1, reg_lambda=1, n_estimator=100, seed=1000)
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(train_x, train_y)],eval_metric='auc', verbose=True)
        return(clf)

    def generate_xgb_leaf_data(self, xgb_clf,train_x, file_name):
         xgb_leaf_feature = xgb_clf.apply(train_x)
         np.savetxt(file_name, xgb_leaf_feature, delimiter = ',')

    def load_leaf_data(self, file_name):
        with open(file_name) as fp:
            leaf_data = np.loadtxt(fp, delimiter = ',')
            return(leaf_data)

    def split_leaf_data(self, train_y, leaf_data):
        train_x,test_x,train_y,test_y = train_test_split(leaf_data, train_y, test_size = 0.3, random_state = 33)
        return([train_x,test_x,train_y,test_y])

    def train_linear_model(self, train_x, train_y):
        clf = linear_model.LogisticRegression(C=1, verbose = 1)
        clf.fit(train_x, train_y)
        return(clf)

    def evaluate_linear_model(self, clf, train_x, test_x, train_y, test_y):
        print('train score: ', clf.score(train_x, train_y))
        print('test score: ', clf.score(test_x, test_y))
        
 

if __name__ == '__main__':
    my_xgb_linear_model = XgboostLinearModel('xgb + linear model')
    data = my_xgb_linear_model.load_train_data('train_data.csv')
    train_x,train_y = my_xgb_linear_model.generate_xgb_train_data(data, 0, 2, 142)
    xgb_clf = my_xgb_linear_model.train_xgb_model(train_x, train_y)
    my_xgb_linear_model.generate_xgb_leaf_data(xgb_clf,train_x,'boost_leaf_feature.csv')
    leaf_data = my_xgb_linear_model.load_leaf_data('boost_leaf_feature.csv')
    train_x,test_x,train_y,test_y = my_xgb_linear_model.split_leaf_data(train_y, leaf_data)
    clf = my_xgb_linear_model.train_linear_model(train_x, train_y)
    my_xgb_linear_model.evaluate_linear_model(clf, train_x, test_x, train_y, test_y) 
