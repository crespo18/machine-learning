#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing

class GbdtModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def load_data(self, file_name):
        with open(file_name) as fp:
            data = np.loadtxt(fp, delimiter = ',', dtype=np.float32)
            return(data)

    def split_data(self, data, train_y_idx, train_x_start_idx, train_x_end_idx):
        train_y = data[:,train_y_idx]
        train_x = data[:,train_x_start_idx:train_x_end_idx]
        train_x = preprocessing.scale(train_x)
        train_x,test_x,train_y,test_y = train_test_split(train_x, train_y, test_size = 0.3, random_state = 33)
        return([train_x,test_x,train_y,test_y])


    def train_gbdt_model(self, train_x, train_y, test_x, test_y):
        clf = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=0.8, max_depth=6, min_samples_leaf=10, random_state=10)
        clf.fit(train_x, train_y)
        return(clf)


    def evaluate_gbdt_model(self, clf, train_x, train_y, test_x, test_y):
        print('train score: ', clf.score(train_x, train_y))
        print('test score: ', clf.score(test_x, test_y))


if __name__ == '__main__':
    my_gbdt_model = GbdtModel('gbdt model')
    data = my_gbdt_model.load_data('train_data.csv')
    train_x,test_x,train_y,test_y = my_gbdt_model.split_data(data, 0, 2, 142)
    clf = my_gbdt_model.train_gbdt_model(train_x,train_y,test_x,test_y)
    my_gbdt_model.evaluate_gbdt_model(clf, train_x, train_y, test_x, test_y)
