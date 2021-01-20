#!/usr/bin/env python

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from sklearn import preprocessing

class XgboostModel():
    def __init__(self, user_name):
        self.user_name = user_name
        print('model start: ', self.user_name)

    def load_train_data(self, file_name):
        with open(file_name) as fp:
            data = np.loadtxt(fp, delimiter = ',', dtype = np.float32)
            return(data)

    def split_data(self, data, label_idx, train_start_idx, train_end_idx):
        train_y = data[:,label_idx]
        train_x = data[:,train_start_idx:train_end_idx]
        train_x = preprocessing.scale(train_x)
        train_x,test_x,train_y,test_y = train_test_split(train_x, train_y, test_size = 0.3, random_state = 33)
        return([train_x,test_x,train_y,test_y])

    def train_model(self, train_x, test_x, train_y, test_y):
        clf = xgb.XGBClassifier(silent = 0, learning_rate = 0.3, min_child_weight = 5, max_depth = 6, gamma = 0, subsample = 1, max_delta_step = 0, colsample_bytree = 1, reg_lambda = 1, n_estimators = 100, seed = 1000)
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(test_x, test_y)],eval_metric = 'auc', verbose = True)
        
        #保存模型
        clf.save_model('MODEL_FILE/xgb.model')
        return(clf)

    def evaluate_model(self, clf, train_x, train_y, test_x, test_y):
        evaluate_result = clf.evals_result()
        print('train: ', clf.score(train_x, train_y))
        print('test: ', clf.score(test_x, test_y))
        return(evaluate_result)
   
    #加载模型
    def load_model(self, file, test_x):
        clf = xgb.XGBClassifier()
        clf.load_model(file)
        predict_result = clf.predict_proba(test_x)
        print('predict_result.shape:\n', predict_result.shape)
        print(predict_result)


if __name__ == '__main__':
    my_xgboost_model = XgboostModel('xgboost')
    data = my_xgboost_model.load_train_data('train_data.csv')
    train_x,test_x,train_y,test_y = my_xgboost_model.split_data(data, 0, 2, 142)
    print('data.shape: ','\ntrain_x: ',train_x.shape,'\ntest_x: ',test_x.shape, '\ntrain_y: ',train_y.shape,'\ntest_y: ',test_y.shape,'\n')
    clf = my_xgboost_model.train_model(train_x, test_x, train_y, test_y)
    evals_result = my_xgboost_model.evaluate_model(clf, train_x, train_y, test_x, test_y)
    print('evaluate_result:\n', evals_result)
    my_xgboost_model.load_model('MODEL_FILE/xgb.model',test_x)
