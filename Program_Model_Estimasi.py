import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy import integrate
from scipy import optimize

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn import linear_model

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#%matplotlib qt

dc = pd.read_csv('DATA_FITUR3.csv')
dc = dc[dc.f1 > 0]
dc = dc[dc.f5 < 1]
dc = dc[dc.f19 < 3.5]
dc = dc[dc.f22 > -2.25]
dc = dc[dc.f25 < 4.5]

dbf = dc[['f1', 'f2', 'f3', 'f4','f5', 'f6', 'f7', 'f8', 'f9', 
    'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
    'f20','f21', 'f22', 'f23', 'f24', 'f25', 'tc', 'ts', 'td']]

ls_ps = dc['sistole'].values.tolist()
ls_pd = dc['diastole'].values.tolist()
ls_tc = dc['tc'].values.tolist()
ls_ts = dc['ts'].values.tolist()

ls_qo = []
ls_cat = []

#menghitung nilai Qo
for i in range(len(ls_tc)):
    integran = lambda x: np.sin(np.pi*x / ls_ts[i])
    integ,_ = integrate.quad(integran, 0, ls_ts[i])

    ls_qo.append(round(1000*5*ls_tc[i] / (60*(integ)),2))

dc['qo'] = ls_qo
ls_io = dc['qo'].values.tolist()

#Menghitung nilai R dan C untuk 219 data

ls_R = []
ls_C = []

for i in range(len(ls_tc)):
    tc = ls_tc[i]
    ts = ls_ts[i]
    td = tc - ts
    io = ls_io[i]
    ps = ls_ps[i]
    pd = ls_pd[i]
    
    def fungsi(z):
        R = z[0]
        C = z[1]

        F = np.empty((2))

        F[0] = pd * np.exp(-ts/(R*C)) + ( (io*ts*C*np.pi*R*R)/(ts*ts + C*C * np.pi*
                                                              np.pi * R*R) ) * (1+np.exp(-ts/(R*C))) - ps

        F[1] = ps * np.exp(-td/(R*C)) - pd

        return(F)
    zGuess = np.array([1,1])

    z = optimize.fsolve(fungsi, zGuess)
    R = round(z[0],2)
    C = round(z[1],2)

    ls_R.append(R)
    ls_C.append(C)

dc['R'] = ls_R
dc['C'] = ls_C

#Cross Validation

fitur_R = ['f1', 'f2', 'f3', 'f4','f5', 'f6', 'f7', 'f8', 'f9', 
    'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
    'f20','f21', 'f22', 'f23', 'f24', 'f25']

fitur_C = ['f1', 'f2', 'f3', 'f4','f5', 'f6', 'f7', 'f8', 'f9', 
    'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
    'f20','f21', 'f22', 'f23', 'f24', 'f25']

##Pendefinisian fitur dan target

XR = dc[fitur_R]
XC = dc[fitur_C]

yR = dc.R
yC = dc.C 

### Pendefinisian Model Estimasi dan Hyperparameter ###
rf_R = RandomForestRegressor(n_estimators=800 , max_features=8)
rf_C = RandomForestRegressor(n_estimators=800 , max_features=8)

gb_R = GradientBoostingRegressor(n_estimators=800 , max_features=8)
gb_C = GradientBoostingRegressor(n_estimators=800 , max_features=8)

### Melakukan Cross Validation ###
cv_rf_R = cross_val_score (rf_R, XR,yR, cv=10, scoring = 'neg_mean_absolute_error' )
cv_rf_C = cross_val_score (rf_C, XC,yC, cv=10, scoring = 'neg_mean_absolute_error' )

cv_gb_R = cross_val_score (gb_R, XR,yR, cv=10, scoring = 'neg_mean_absolute_error' )
cv_gb_C = cross_val_score (gb_C, XC,yC, cv=10, scoring = 'neg_mean_absolute_error' )

### Menampilkan Hasil Cross Validation ###

print('Hasil Cross Validation')
print('RF_R =', np.mean(-cv_rf_R))
print('GB_R =', np.mean(-cv_gb_R))
print()
print('RF_C =', np.mean(-cv_rf_C))
print('GB_C =', np.mean(-cv_gb_C))

#Melatih model akhir dan menyimpan hasil latih

XR = dc[fitur_R]
XC = dc[fitur_C]

yR = dc.R
yC = dc.C 

#Melatih model regresi
rf_R = RandomForestRegressor(n_estimators=200 , max_features=2)
rf_C = RandomForestRegressor(n_estimators=200 , max_features=8)

#gb_R = GradientBoostingRegressor(n_estimators=200 , max_features=2)
#gb_C = GradientBoostingRegressor(n_estimators=200 , max_features=4)

rf_R.fit(XR, yR)
#gb_R.fit(XR, yR)

rf_C.fit(XC, yC)
#gb_C.fit(XC, yC)

Pkl_Filename_RF_R = 'Rf_Wind_R2.pkl'
Pkl_Filename_RF_C = 'Rf_Wind_C2.pkl'

#Pkl_Filename_GB_R = 'Gb_Wind_R.pkl'
#Pkl_Filename_GB_C = 'Gb_Wind_C.pkl'

with open(Pkl_Filename_RF_R, 'wb') as file:
    pickle.dump(rf_R, file)
with open(Pkl_Filename_RF_C, 'wb') as file:
    pickle.dump(rf_C, file)

#with open(Pkl_Filename_GB_R, 'wb') as file:
    #pickle.dump(gb_R, file)
#with open(Pkl_Filename_GB_C, 'wb') as file:
    #pickle.dump(gb_C, file)



