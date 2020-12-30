#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing

class LinearModel():
    def __init__(self, model_name):
        self.model_name = model_name
        print('model start: ', self.model_name)

    def load_train_csv_data(self, file_name):
        with open(file_name) as fp:
            data = np.loadtxt(fp, delimiter = ',', dtype=np.float32)
            return(data)

    def split_data(self, data, label_idx, train_start_idx, train_end_idx):
        train_y = data[:,label_idx]
        train_x = data[:,train_start_idx:train_end_idx]
        train_x = preprocessing.scale(train_x)

        train_x,test_x,train_y,test_y = train_test_split(train_x, train_y, test_size = 0.3, random_state = 33)
        return([train_x,test_x,train_y,test_y])


    def train_linear_model(self, train_x, test_x, train_y, test_y):
        clf = linear_model.LogisticRegression(C=1, verbose = 1)
        clf.fit(train_x, train_y)
        return(clf)

    def evaluate_linear_model(self, clf, train_x, test_x, train_y, test_y):
        print('train score: ', clf.score(train_x, train_y))
        print('test score: ', clf.score(test_x, test_y))
   
if __name__ == '__main__':
    my_linear_model = LinearModel('logistic')
    data = my_linear_model.load_train_csv_data('train_data.csv')
    train_x,test_x,train_y,test_y = my_linear_model.split_data(data, 0, 2, 142)
    print('train_y: ',train_y.shape, train_y[:20])
    print('train_x: ',train_x.shape, train_x[:20])
    print('test_y: ',test_y.shape, test_y[:20])
    print('test_x: ',test_x.shape, test_x[:20])

    clf = my_linear_model.train_linear_model(train_x, test_x, train_y, test_y)
    my_linear_model.evaluate_linear_model(clf, train_x, test_x, train_y, test_y)  
   
