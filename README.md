# machine-learning
传统机器学习算法的应用案例

train_data.csv
第一列是标签，黑标签为1，白标签为0，每行是一个训练样本，从2-142列是每个特征的值（事前加工过）

1、Logistic_model.py执行情况

train_y:  (350,) [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
train_x:  (350, 140) [[-0.16366175 -0.16503525 -0.15327023 ... -0.16947891 -0.19388726
  -0.16255617]
 [-0.18296148 -0.14946736 -0.04868458 ... -0.14793953 -0.19388726
  -0.16255617]
 [-0.10474677 -0.07941185 -0.10097741 ... -0.16947891 -0.19388726
  -0.16255617]
 ...
 [-0.21546629 -0.22730683 -0.20556305 ... -0.15152942 -0.12891161
  -0.16255617]
 [-0.21343474 -0.22730683 -0.20556305 ... -0.14793953 -0.19388726
   0.12981096]
 [-0.21445052 -0.21952288 -0.20556305 ... -0.16947891 -0.19388726
  -0.16255617]]
test_y:  (150,) [0. 0. 1. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
test_x:  (150, 140) [[-0.21343474 -0.22730683 -0.20556305 ... -0.11563046 -0.19388726
  -0.16255617]
 [-0.12099917 -0.09497975 -0.07483099 ... -0.16947891 -0.19388726
  -0.16255617]
 [ 0.19389124  0.08405101 -0.08790421 ...  0.22899951 -0.19388726
  -0.16255617]
 ...
 [-0.09966788 -0.05606001 -0.17941664 ... -0.16588901 -0.19388726
  -0.16255617]
 [-0.11287297 -0.09497975 -0.11405062 ... -0.16947891 -0.19388726
  -0.16255617]
 [-0.20530854 -0.21952288 -0.17941664 ...  0.8464616   0.52084476
   1.0069124 ]]
 
 iter  1 act 9.453e+01 pre 8.751e+01 delta 1.527e+00 f 2.426e+02 |g| 1.620e+02 CG   5
iter  2 act 8.545e+00 pre 7.371e+00 delta 1.527e+00 f 1.481e+02 |g| 3.958e+01 CG  10
iter  3 act 1.579e+00 pre 1.345e+00 delta 1.527e+00 f 1.395e+02 |g| 1.700e+01 CG   9
iter  4 act 2.620e-01 pre 2.305e-01 delta 1.527e+00 f 1.380e+02 |g| 5.605e+00 CG  10
iter  5 act 3.748e-02 pre 3.259e-02 delta 1.527e+00 f 1.377e+02 |g| 1.929e+00 CG  10
iter  6 act 6.415e-03 pre 5.421e-03 delta 1.527e+00 f 1.377e+02 |g| 5.323e-01 CG  10
iter  7 act 8.747e-04 pre 8.104e-04 delta 1.527e+00 f 1.376e+02 |g| 1.420e-01 CG  11
iter  8 act 2.226e-05 pre 2.211e-05 delta 1.527e+00 f 1.376e+02 |g| 2.132e-02 CG  11
[LibLinear]train score:  0.8457142857142858
test score:  0.8666666666666667

