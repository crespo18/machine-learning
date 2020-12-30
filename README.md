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

2、gbdt_model.py执行情况

train score:  0.92

test score:  0.8066666666666666


3、xgboost_model.py执行情况
train_x:  (350, 140)

test_x:  (150, 140)

train_y:  (350,)

test_y:  (150,)


[0]     validation_0-auc:0.681743       validation_1-auc:0.469066

[1]     validation_0-auc:0.787878       validation_1-auc:0.437079

[2]     validation_0-auc:0.844001       validation_1-auc:0.418771

[3]     validation_0-auc:0.885097       validation_1-auc:0.48085

[4]     validation_0-auc:0.895577       validation_1-auc:0.445076

[5]     validation_0-auc:0.905615       validation_1-auc:0.470118

[6]     validation_0-auc:0.913852       validation_1-auc:0.485901

[7]     validation_0-auc:0.915181       validation_1-auc:0.491582

[8]     validation_0-auc:0.923713       validation_1-auc:0.503788

[9]     validation_0-auc:0.928053       validation_1-auc:0.480008

[10]    validation_0-auc:0.930355       validation_1-auc:0.495581

[11]    validation_0-auc:0.936673       validation_1-auc:0.498527

[12]    validation_0-auc:0.940039       validation_1-auc:0.479588

[13]    validation_0-auc:0.944586       validation_1-auc:0.482534

[14]    validation_0-auc:0.943965       validation_1-auc:0.482955

[15]    validation_0-auc:0.948453       validation_1-auc:0.485059

[16]    validation_0-auc:0.94928        validation_1-auc:0.477904

[17]    validation_0-auc:0.951641       validation_1-auc:0.482534

[18]    validation_0-auc:0.953177       validation_1-auc:0.491793

[19]    validation_0-auc:0.955952       validation_1-auc:0.489268

[20]    validation_0-auc:0.955834       validation_1-auc:0.4758

[21]    validation_0-auc:0.957133       validation_1-auc:0.472012

[22]    validation_0-auc:0.958786       validation_1-auc:0.474958

[23]    validation_0-auc:0.958077       validation_1-auc:0.484217

[24]    validation_0-auc:0.960262       validation_1-auc:0.494318

[25]    validation_0-auc:0.961384       validation_1-auc:0.480008

[26]    validation_0-auc:0.963805       validation_1-auc:0.485059

[27]    validation_0-auc:0.964159       validation_1-auc:0.483375

[28]    validation_0-auc:0.964277       validation_1-auc:0.481692

[29]    validation_0-auc:0.965163       validation_1-auc:0.48085

[30]    validation_0-auc:0.966403       validation_1-auc:0.49053

[31]    validation_0-auc:0.966993       validation_1-auc:0.492635

[32]    validation_0-auc:0.966285       validation_1-auc:0.492214

[33]    validation_0-auc:0.967466       validation_1-auc:0.49979

[34]    validation_0-auc:0.967702       validation_1-auc:0.493897

[35]    validation_0-auc:0.968233       validation_1-auc:0.492635

[36]    validation_0-auc:0.968351       validation_1-auc:0.503577

[37]    validation_0-auc:0.96841        validation_1-auc:0.501473

[38]    validation_0-auc:0.969946       validation_1-auc:0.493897

[39]    validation_0-auc:0.969178       validation_1-auc:0.49516

[40]    validation_0-auc:0.969296       validation_1-auc:0.494739

[41]    validation_0-auc:0.970182       validation_1-auc:0.495581

[42]    validation_0-auc:0.970713       validation_1-auc:0.493476

[43]    validation_0-auc:0.970595       validation_1-auc:0.48548


[44]    validation_0-auc:0.971127       validation_1-auc:0.489689

[45]    validation_0-auc:0.971363       validation_1-auc:0.491793

[46]    validation_0-auc:0.971422       validation_1-auc:0.496843

[47]    validation_0-auc:0.971658       validation_1-auc:0.511574

[48]    validation_0-auc:0.971894       validation_1-auc:0.505682

[49]    validation_0-auc:0.971599       validation_1-auc:0.501473

[50]    validation_0-auc:0.97213        validation_1-auc:0.496002

[51]    validation_0-auc:0.972839       validation_1-auc:0.501052

[52]    validation_0-auc:0.973607       validation_1-auc:0.50021

[53]    validation_0-auc:0.973961       validation_1-auc:0.496423

[54]    validation_0-auc:0.973547       validation_1-auc:0.488005

[55]    validation_0-auc:0.973075       validation_1-auc:0.487584

[56]    validation_0-auc:0.973016       validation_1-auc:0.485059

[57]    validation_0-auc:0.97337        validation_1-auc:0.487584

[58]    validation_0-auc:0.973016       validation_1-auc:0.488426

[59]    validation_0-auc:0.973075       validation_1-auc:0.489268

[60]    validation_0-auc:0.973311       validation_1-auc:0.487584

[61]    validation_0-auc:0.974374       validation_1-auc:0.491372

[62]    validation_0-auc:0.974079       validation_1-auc:0.486322

[63]    validation_0-auc:0.974551       validation_1-auc:0.489268

[64]    validation_0-auc:0.97461        validation_1-auc:0.493897

[65]    validation_0-auc:0.974728       validation_1-auc:0.496002

[66]    validation_0-auc:0.974965       validation_1-auc:0.49516

[67]    validation_0-auc:0.974728       validation_1-auc:0.493897

[68]    validation_0-auc:0.974846       validation_1-auc:0.49053

[69]    validation_0-auc:0.974846       validation_1-auc:0.488005

[70]    validation_0-auc:0.974965       validation_1-auc:0.492214

[71]    validation_0-auc:0.974846       validation_1-auc:0.488426

[72]    validation_0-auc:0.975024       validation_1-auc:0.488005

[73]    validation_0-auc:0.975437       validation_1-auc:0.490109

[74]    validation_0-auc:0.975555       validation_1-auc:0.490951

[75]    validation_0-auc:0.975555       validation_1-auc:0.487163

[76]    validation_0-auc:0.975496       validation_1-auc:0.489268

[77]    validation_0-auc:0.975614       validation_1-auc:0.488005

[78]    validation_0-auc:0.976027       validation_1-auc:0.488005

[79]    validation_0-auc:0.975968       validation_1-auc:0.488005

[80]    validation_0-auc:0.976027       validation_1-auc:0.487584

[81]    validation_0-auc:0.976027       validation_1-auc:0.489268

[82]    validation_0-auc:0.976264       validation_1-auc:0.483375

[83]    validation_0-auc:0.976264       validation_1-auc:0.487163

[84]    validation_0-auc:0.976205       validation_1-auc:0.488005

[85]    validation_0-auc:0.976264       validation_1-auc:0.486742

[86]    validation_0-auc:0.976323       validation_1-auc:0.479588

[87]    validation_0-auc:0.976382       validation_1-auc:0.479588

[88]    validation_0-auc:0.976264       validation_1-auc:0.48085

[89]    validation_0-auc:0.976323       validation_1-auc:0.480008

[90]    validation_0-auc:0.976441       validation_1-auc:0.482955

[91]    validation_0-auc:0.9765 validation_1-auc:0.483375

[92]    validation_0-auc:0.9765 validation_1-auc:0.486322

[93]    validation_0-auc:0.9765 validation_1-auc:0.486742

[94]    validation_0-auc:0.976441       validation_1-auc:0.487584

[95]    validation_0-auc:0.976382       validation_1-auc:0.486322

[96]    validation_0-auc:0.976441       validation_1-auc:0.486322

[97]    validation_0-auc:0.9765 validation_1-auc:0.483375

[98]    validation_0-auc:0.976529       validation_1-auc:0.487584

[99]    validation_0-auc:0.976529       validation_1-auc:0.483796

train:  0.92

test:  0.82

evaluate_result:
 
 {'validation_0': {'auc': [0.681743, 0.787878, 0.844001, 0.885097, 0.895577, 0.905615, 0.913852, 0.915181, 0.923713, 0.928053, 0.930355, 0.936673, 0.940039, 0.944586, 0.943965, 0.948453, 0.94928, 0.951641, 0.953177, 0.955952, 0.955834, 0.957133, 0.958786, 0.958077, 0.960262, 0.961384, 0.963805, 0.964159, 0.964277, 0.965163, 0.966403, 0.966993, 0.966285, 0.967466, 0.967702, 0.968233, 0.968351, 0.96841, 0.969946, 0.969178, 0.969296, 0.970182, 0.970713, 0.970595, 0.971127, 0.971363, 0.971422, 0.971658, 0.971894, 0.971599, 0.97213, 0.972839, 0.973607, 0.973961, 0.973547, 0.973075, 0.973016, 0.97337, 0.973016, 0.973075, 0.973311, 0.974374, 0.974079, 0.974551, 0.97461, 0.974728, 0.974965, 0.974728, 0.974846, 0.974846, 0.974965, 0.974846, 0.975024, 0.975437, 0.975555, 0.975555, 0.975496, 0.975614, 0.976027, 0.975968, 0.976027, 0.976027, 0.976264, 0.976264, 0.976205, 0.976264, 0.976323, 0.976382, 0.976264, 0.976323, 0.976441, 0.9765, 0.9765, 0.9765, 0.976441, 0.976382, 0.976441, 0.9765, 0.976529, 0.976529]}, 'validation_1': {'auc': [0.469066, 0.437079, 0.418771, 0.48085, 0.445076, 0.470118, 0.485901, 0.491582, 0.503788, 0.480008, 0.495581, 0.498527, 0.479588, 0.482534, 0.482955, 0.485059, 0.477904, 0.482534, 0.491793, 0.489268, 0.4758, 0.472012, 0.474958, 0.484217, 0.494318, 0.480008, 0.485059, 0.483375, 0.481692, 0.48085, 0.49053, 0.492635, 0.492214, 0.49979, 0.493897, 0.492635, 0.503577, 0.501473, 0.493897, 0.49516, 0.494739, 0.495581, 0.493476, 0.48548, 0.489689, 0.491793, 0.496843, 0.511574, 0.505682, 0.501473, 0.496002, 0.501052, 0.50021, 0.496423, 0.488005, 0.487584, 0.485059, 0.487584, 0.488426, 0.489268, 0.487584, 0.491372, 0.486322, 0.489268, 0.493897, 0.496002, 0.49516, 0.493897, 0.49053, 0.488005, 0.492214, 0.488426, 0.488005, 0.490109, 0.490951, 0.487163, 0.489268, 0.488005, 0.488005, 0.488005, 0.487584, 0.489268, 0.483375, 0.487163, 0.488005, 0.486742,0.479588, 0.479588, 0.48085, 0.480008, 0.482955, 0.483375, 0.486322, 0.486742, 0.487584, 0.486322, 0.486322, 0.483375, 0.487584, 0.483796]}}


4、xgboost_logistic.py执行情况
[0]     validation_0-auc:0.77118        validation_1-auc:0.77118
[1]     validation_0-auc:0.820243       validation_1-auc:0.820243
[2]     validation_0-auc:0.841717       validation_1-auc:0.841717
[3]     validation_0-auc:0.878352       validation_1-auc:0.878352
[4]     validation_0-auc:0.889741       validation_1-auc:0.889741
[5]     validation_0-auc:0.910889       validation_1-auc:0.910889
[6]     validation_0-auc:0.913217       validation_1-auc:0.913217
[7]     validation_0-auc:0.92186        validation_1-auc:0.92186
[8]     validation_0-auc:0.930719       validation_1-auc:0.930719
[9]     validation_0-auc:0.934319       validation_1-auc:0.934319
[10]    validation_0-auc:0.934055       validation_1-auc:0.934055
[11]    validation_0-auc:0.939114       validation_1-auc:0.939114
[12]    validation_0-auc:0.943365       validation_1-auc:0.943365
[13]    validation_0-auc:0.944762       validation_1-auc:0.944762
[14]    validation_0-auc:0.946841       validation_1-auc:0.946841
[15]    validation_0-auc:0.949168       validation_1-auc:0.949168
[16]    validation_0-auc:0.950317       validation_1-auc:0.950317
[17]    validation_0-auc:0.950782       validation_1-auc:0.950782
[18]    validation_0-auc:0.952147       validation_1-auc:0.952147
[19]    validation_0-auc:0.953451       validation_1-auc:0.953451
[20]    validation_0-auc:0.954816       validation_1-auc:0.954816
[21]    validation_0-auc:0.956089       validation_1-auc:0.956089
[22]    validation_0-auc:0.956616       validation_1-auc:0.956616
[23]    validation_0-auc:0.957764       validation_1-auc:0.957764
[24]    validation_0-auc:0.958447       validation_1-auc:0.958447
[25]    validation_0-auc:0.958695       validation_1-auc:0.958695
[26]    validation_0-auc:0.959192       validation_1-auc:0.959192
[27]    validation_0-auc:0.959782       validation_1-auc:0.959782
[28]    validation_0-auc:0.959906       validation_1-auc:0.959906
[29]    validation_0-auc:0.959968       validation_1-auc:0.959968
[30]    validation_0-auc:0.961116       validation_1-auc:0.961116
[31]    validation_0-auc:0.960775       validation_1-auc:0.960775
[32]    validation_0-auc:0.961116       validation_1-auc:0.961116
[33]    validation_0-auc:0.961271       validation_1-auc:0.961271
[34]    validation_0-auc:0.961302       validation_1-auc:0.961302
[35]    validation_0-auc:0.960868       validation_1-auc:0.960868
[36]    validation_0-auc:0.961147       validation_1-auc:0.961147
[37]    validation_0-auc:0.961147       validation_1-auc:0.961147
[38]    validation_0-auc:0.961209       validation_1-auc:0.961209
[39]    validation_0-auc:0.96124        validation_1-auc:0.96124
[40]    validation_0-auc:0.961581       validation_1-auc:0.961581
[41]    validation_0-auc:0.961395       validation_1-auc:0.961395
[42]    validation_0-auc:0.961737       validation_1-auc:0.961737
[43]    validation_0-auc:0.961737       validation_1-auc:0.961737
[44]    validation_0-auc:0.961954       validation_1-auc:0.961954
[45]    validation_0-auc:0.962109       validation_1-auc:0.962109
[46]    validation_0-auc:0.962419       validation_1-auc:0.962419
[47]    validation_0-auc:0.962388       validation_1-auc:0.962388
[48]    validation_0-auc:0.962295       validation_1-auc:0.962295
[49]    validation_0-auc:0.962357       validation_1-auc:0.962357
[50]    validation_0-auc:0.962171       validation_1-auc:0.962171
[51]    validation_0-auc:0.96214        validation_1-auc:0.96214
[52]    validation_0-auc:0.962233       validation_1-auc:0.962233
[53]    validation_0-auc:0.962512       validation_1-auc:0.962512
[54]    validation_0-auc:0.962668       validation_1-auc:0.962668
[55]    validation_0-auc:0.962761       validation_1-auc:0.962761
[56]    validation_0-auc:0.962699       validation_1-auc:0.962699
[57]    validation_0-auc:0.962606       validation_1-auc:0.962606
[58]    validation_0-auc:0.962823       validation_1-auc:0.962823
[59]    validation_0-auc:0.963009       validation_1-auc:0.963009
[60]    validation_0-auc:0.963009       validation_1-auc:0.963009
[61]    validation_0-auc:0.963071       validation_1-auc:0.963071
[62]    validation_0-auc:0.963195       validation_1-auc:0.963195
[63]    validation_0-auc:0.963102       validation_1-auc:0.963102
[64]    validation_0-auc:0.963102       validation_1-auc:0.963102
[65]    validation_0-auc:0.963133       validation_1-auc:0.963133
[66]    validation_0-auc:0.96304        validation_1-auc:0.96304
[67]    validation_0-auc:0.963226       validation_1-auc:0.963226
[68]    validation_0-auc:0.963319       validation_1-auc:0.963319
[69]    validation_0-auc:0.963443       validation_1-auc:0.963443
[70]    validation_0-auc:0.963568       validation_1-auc:0.963568
[71]    validation_0-auc:0.963537       validation_1-auc:0.963537
[72]    validation_0-auc:0.963505       validation_1-auc:0.963505
[73]    validation_0-auc:0.963568       validation_1-auc:0.963568
[74]    validation_0-auc:0.96363        validation_1-auc:0.96363
[75]    validation_0-auc:0.963785       validation_1-auc:0.963785
[76]    validation_0-auc:0.963661       validation_1-auc:0.963661
[77]    validation_0-auc:0.963847       validation_1-auc:0.963847
[78]    validation_0-auc:0.96394        validation_1-auc:0.96394
[79]    validation_0-auc:0.964033       validation_1-auc:0.964033
[80]    validation_0-auc:0.964064       validation_1-auc:0.964064
[81]    validation_0-auc:0.964126       validation_1-auc:0.964126
[82]    validation_0-auc:0.964033       validation_1-auc:0.964033
[83]    validation_0-auc:0.964157       validation_1-auc:0.964157
[84]    validation_0-auc:0.964188       validation_1-auc:0.964188
[85]    validation_0-auc:0.964188       validation_1-auc:0.964188
[86]    validation_0-auc:0.964157       validation_1-auc:0.964157
[87]    validation_0-auc:0.964188       validation_1-auc:0.964188
[88]    validation_0-auc:0.964219       validation_1-auc:0.964219
[89]    validation_0-auc:0.964188       validation_1-auc:0.964188
[90]    validation_0-auc:0.964157       validation_1-auc:0.964157
[91]    validation_0-auc:0.964219       validation_1-auc:0.964219
[92]    validation_0-auc:0.964188       validation_1-auc:0.964188
[93]    validation_0-auc:0.964219       validation_1-auc:0.964219
[94]    validation_0-auc:0.964219       validation_1-auc:0.964219
[95]    validation_0-auc:0.964219       validation_1-auc:0.964219
[96]    validation_0-auc:0.964219       validation_1-auc:0.964219
[97]    validation_0-auc:0.964219       validation_1-auc:0.964219
[98]    validation_0-auc:0.964219       validation_1-auc:0.964219
[99]    validation_0-auc:0.964219       validation_1-auc:0.964219
/opt/conda/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
iter  1 act 8.119e+01 pre 7.596e+01 delta 1.770e-02 f 2.426e+02 |g| 9.216e+03 CG   1
cg reaches trust region boundary
iter  2 act 7.253e+00 pre 7.023e+00 delta 3.102e-02 f 1.614e+02 |g| 1.236e+03 CG   2
cg reaches trust region boundary
iter  3 act 5.733e+00 pre 5.438e+00 delta 5.011e-02 f 1.542e+02 |g| 2.688e+02 CG   2
cg reaches trust region boundary
iter  4 act 9.427e+00 pre 9.682e+00 delta 1.127e-01 f 1.484e+02 |g| 7.770e+02 CG   2
cg reaches trust region boundary
iter  5 act 1.184e+01 pre 1.145e+01 delta 1.691e-01 f 1.390e+02 |g| 2.330e+02 CG   2
cg reaches trust region boundary
iter  6 act 1.443e+01 pre 1.420e+01 delta 2.461e-01 f 1.272e+02 |g| 2.889e+02 CG   3
cg reaches trust region boundary
iter  7 act 1.199e+01 pre 1.095e+01 delta 2.961e-01 f 1.127e+02 |g| 3.011e+02 CG   4
cg reaches trust region boundary
iter  8 act 1.023e+01 pre 9.914e+00 delta 3.365e-01 f 1.007e+02 |g| 3.135e+02 CG   5
cg reaches trust region boundary
iter  9 act 9.182e+00 pre 8.601e+00 delta 4.802e-01 f 9.051e+01 |g| 2.932e+02 CG   6
cg reaches trust region boundary
iter 10 act 7.010e+00 pre 6.614e+00 delta 6.197e-01 f 8.132e+01 |g| 1.577e+02 CG   7
cg reaches trust region boundary
iter 11 act 5.904e+00 pre 5.535e+00 delta 7.283e-01 f 7.432e+01 |g| 1.622e+02 CG  10
cg reaches trust region boundary
iter 12 act 4.308e+00 pre 3.803e+00 delta 8.502e-01 f 6.841e+01 |g| 9.611e+01 CG  12
iter 13 act 9.961e-01 pre 9.684e-01 delta 8.502e-01 f 6.410e+01 |g| 9.415e+01 CG   9
cg reaches trust region boundary
iter 14 act 1.933e+00 pre 1.806e+00 delta 9.685e-01 f 6.311e+01 |g| 1.645e+01 CG  12
iter 15 act 3.503e-01 pre 3.491e-01 delta 9.685e-01 f 6.118e+01 |g| 3.627e+01 CG  12
iter 16 act 4.056e-01 pre 3.929e-01 delta 9.685e-01 f 6.082e+01 |g| 8.452e+00 CG  22
iter 17 act 1.253e-02 pre 1.245e-02 delta 9.685e-01 f 6.042e+01 |g| 6.444e+00 CG  16
iter 18 act 7.428e-03 pre 7.445e-03 delta 9.685e-01 f 6.041e+01 |g| 6.520e-01 CG  36
iter 19 act 1.377e-04 pre 1.375e-04 delta 9.685e-01 f 6.040e+01 |g| 1.786e-01 CG  31
[LibLinear]train score:  0.92
test score:  0.8466666666666667
