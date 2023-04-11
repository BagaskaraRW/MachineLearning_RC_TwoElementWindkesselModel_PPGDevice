import csv
import pstats
from sqlite3 import Row
from scipy import signal
from scipy import integrate
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def filter_bp(sig):         #fungsi untuk filter sinyal
    srate = 1000            #Sampling rate
    fcl= 10                 #frekuensi cuttoff
    fch= 0.5
    wnl = fcl/(srate/2)     #frekuensi angular
    wnh = fch/(srate/2)

    b,a = signal.cheby2(4, 20, wnh, 'high')
    fil = signal.filtfilt(b,a, sig)

    b,a = signal.cheby2(4, 20, wnl, 'low')
    fil = signal.filtfilt(b,a, fil)

    return(fil)

def normalisasi_bp(ppg):    #fungsi untuk normalisasi standar deviasi
    mean = np.mean(ppg)
    std = np.std(ppg)
    ppg = [((i - mean)/std) for i in ppg]
    return(ppg)

def normalisasi_max(fil):   #fungsi untuk normalisasi min - max
    maks = max(fil)
    mins = min(fil)
    fil = [((i - mins)/(maks - mins)) for i in fil]
    return(fil)

def cari_batas(ppg, vpg, apg): #fungsu untuk mencari batas sinyal
    apg = normalisasi_bp(apg)

    try:
        bt = signal.find_peaks(apg, prominence = 0.5, height = [1.5,3.5])[0]
        bt_ki, bt_ka = bt[-2]-30, bt[-1]-30
        pjg = bt_ka - bt_ki

        if pjg<500:
            bt_ki, bt_ka = bt[-2]-30, bt[-1]-30
    
    except:
        bt = signal.find_peaks(apg, prominence = 0.5, height = [1,3])[0]
        bt_ki, bt_ka = bt[-2]-30, bt[-1]-30
    
    return([bt_ki, bt_ka])

def potong(batas, ppg, vpg, apg):   #Fungsi untuk memotong siklus sinyal
    ppg = ppg[batas[0]: batas[1]]
    vpg = vpg[batas[0]: batas[1]]
    apg = apg[batas[0]: batas[1]]
    return(ppg, vpg, apg)

def deteksi_wave(ppg, vpg, apg):    #Fungsi untuk mendeteksi karakter sinyal
    global indx, indx_O, indx_S, indx_O2, indx_wyz, indx_y, indx_a, indx_b, indx_ab, indx_iapg, indx_de, indx_cde, indx_c 
    global indx_e, indx_d, indx_D, indx_w, indx_y, indx_z, O, S, D, O2, w, y, z, a, b, c, d, e
    
    ppg_raw = ppg

    ppg = normalisasi_bp(ppg)
    vpg = normalisasi_bp(vpg)
    apg = normalisasi_bp(apg)

    indx = [i for i in range(len(ppg))]

    indx_O = indx[0]
    indx_O2 = indx[-1]
    indx_S = signal.find_peaks(ppg, prominence = 0.01)[0][0]
    indx_wyz = signal.find_peaks(vpg, prominence = 0.01)[0]

    vpg_inv = [-i for i in vpg]
    indx_y = signal.find_peaks(vpg_inv)[0][0]

    apg_abs = [abs(i) for i in apg]
    indx_a = signal.find_peaks(apg_abs, prominence = 0)[0][0]
    indx_b = signal.find_peaks(apg_abs, prominence = 0)[0][1]

    apg_inv = [-i for i in apg]
    indx_ab = signal.find_peaks(apg)[0]
    indx_iapg = signal.find_peaks(apg_inv)[0]
    indx_de = [i for i in indx_iapg if i > indx_b and i < 400]
    indx_cde = [i for i in indx_ab if i > indx_b and i < 400]

    if len(indx_cde)>1:
        indx_c = indx_cde[0]
        indx_e = indx_cde[-1]
        try:
            indx_d = int(np.mean(indx_de))
        except:
            indx_d = indx_cde[0]
        
    else:
        indx_c = indx_cde[0]
        indx_d = indx_cde[0]
        indx_e = indx_cde[0]

    indx_D = [i for i in indx_iapg if i > indx_e][0]

    if len(indx_wyz)>1:
        indx_w = indx_wyz[0]
        indx_z = indx_wyz[1]
    
    else:
        indx_w = indx_wyz[0]
        indx_z = int(round((indx_D + indx_e)/2))
    
    ppg = normalisasi_max(ppg_raw)
    vpg = (filter_bp(np.gradient(ppg)))
    apg = (filter_bp(np.gradient(vpg)))

    O = (indx_O, ppg[indx_O], vpg[indx_O], apg[indx_O])
    S = (indx_S, ppg[indx_S], vpg[indx_S], apg[indx_S])
    D = (indx_D, ppg[indx_D], vpg[indx_D], apg[indx_D])
    O2 = (indx_O2, ppg[indx_O2], vpg[indx_O2], apg[indx_O2])

    w = (indx_w, ppg[indx_w], vpg[indx_w], apg[indx_w])
    y = (indx_w, ppg[indx_y], vpg[indx_y], apg[indx_y])
    z = (indx_w, ppg[indx_z], vpg[indx_z], apg[indx_z])

    a = (indx_a, ppg[indx_a], vpg[indx_a], apg[indx_a])
    b = (indx_b, ppg[indx_b], vpg[indx_b], apg[indx_b])
    c = (indx_c, ppg[indx_c], vpg[indx_c], apg[indx_c])
    d = (indx_d, ppg[indx_d], vpg[indx_d], apg[indx_d])
    e = (indx_e, ppg[indx_e], vpg[indx_e], apg[indx_e])

    return(O, S, D, O2, w, y, z, a, b, c, d, e)

def ekstrak_wave(wave, ppg, vpg, apg):  #Fungsi untuk mengekstrasi fitur sinyal
    
    global f1, f2, f3, f4,f5, f6, f7, f8, f9, f10
    global f11, f12, f13, f14, f15, f16, f17, f18, f19, f20
    global f21, f22, f23, f24, f25, tc, ts, td
    
    def time_span(o,s): #fungsi menghitung timespan
        ts = (s[0]-o[0]) / 1000
        return(ts)

    def slope(o,s,p):   #Fungsi untuk menghitung slope
        m = (s[p] - o[p]) / time_span(o,s)
        return(m)
    
    def wave_area(o,s,p):   #Fungsi untuk menghitung wave area
        if p == 1:
            sig = ppg
        elif p == 2:
            sig = vpg
        else:
            sig = apg
        
        if o[0] < s[0]:
            sig = sig[o[0]:s[0]]
            area = integrate.simps(sig, dx = 0.001)
            return(area)
        else:
            sig = sig[s[0]:o[0]]
            area = integrate.simps(sig, dx = 0.001)
            return(-area)
    
    def power_area(o,s,p):  #Fungsi untuk menghitung power area
        if p == 1:
            sig = ppg
        elif p == 2:
            sig = vpg
        else:
            sig = apg
        
        if o[0] < s[0]:
            sig = sig[o[0]:s[0]]
            sig = [i*i for i in sig]
            area = integrate.simps(sig, dx = 0.001)
            return(area)
        else:
            sig = sig[s[0]:o[0]]
            sig = [i*i for i in sig]
            area = integrate.simps(sig, dx = 0.001)
            return(-area)
        
    O, S, D, O2, w, y, z, a, b, c, d, e = wave

    f1 = (abs(b[3]) - abs(c[3]) - abs(d[3])) / a[3]
    f5 = abs(c[1])/S[1]
    f8 = abs(b[1]) / S[1]

    f2 = slope(b, d, 1)
    f3 = slope(S, c, 1)
    f22 = slope(D, O2, 1)

    f6 = power_area(S, c, 2)
    f7 = power_area(S, c, 2) / power_area(O, O2, 2)
    f10 = power_area(w, S, 2) / power_area(O, O2, 2)
    
    tc = time_span(O, O2)
    ts = time_span(O, c)
    td = time_span(c, O2)
    f4 = time_span(O,S)
    f9 = time_span(S,c)
    f11 = time_span(O,O2)
    f12 = time_span(S,O2)
    f13 = time_span(O,D)
    f14 = time_span(D,O2)
    f15 = time_span(O,e)
    f16 = time_span(e,O2)
    f21 = time_span(w,y)

    f23 = wave_area(O, e, 1)
    f24 = wave_area(e, O2, 1)
    
    f17 = S[1] / D[1]
    f18 = (S[1] / D[1])/ S[1]
    f19 = S[1] / e[1]
    f20 = D[1] / e[1]
    f25 = f23/f24

    return([f1, f2, f3, f4,f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, tc, ts, td])

with open('DATA_FITUR2.csv','w',newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['id','f1', 'f2', 'f3', 'f4','f5', 'f6', 'f7', 'f8', 'f9', 
    'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
    'f20','f21', 'f22', 'f23', 'f24', 'f25', 'tc', 'ts', 'td',
    'sistole','diastole'])
    
df = pd.read_csv('PPG-BP_sqi.csv')

ID_subjek = df['subject_ID']
ls_sistole = df['Systolic Blood Pressure(mmHg)']
ls_diastole = df['Diastolic Blood Pressure(mmHg)']

ls_sqi_1 = df['segment 1']
ls_sqi_2 = df['segment 2']
ls_sqi_3 = df['Segment 3']

###############################

for i in range(len(ID_subjek)):
    sistole = ls_sistole[i]
    diastole = ls_diastole[i]
    subjek = ID_subjek[i]

    sqi = [ls_sqi_1[i], ls_sqi_2[i], ls_sqi_3[i]]

    sampel = np.argmax(sqi)+1

    nama = str(subjek)+'_'+str(sampel)
    namacsv = str(subjek)+'_'+str(sampel)+'.txt'

    with open(namacsv, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter="\t")
        for row in spamreader:
            sig = row
    
    sig = [float(i) for i in sig if len(i)>0]
    
    ppg = filter_bp(sig)
    vpg = filter_bp(np.gradient(ppg))
    apg = filter_bp(np.gradient(vpg))

    try:
        raw = ppg
        ppg = normalisasi_max(ppg)
        vpg = filter_bp(np.gradient(ppg))
        apg = filter_bp(np.gradient(vpg))
            
        batas = cari_batas(ppg, vpg, apg)
        ppg, vpg, apg = potong(batas, ppg, vpg, apg)
            
        wave = deteksi_wave(ppg, vpg, apg)

        fitur = ekstrak_wave(wave, ppg, vpg, apg)
    
    except:
        fitur = [0 for i in range(28)]

    ls_write = []
    ls_write.append(nama)
    ls_write.extend(fitur)
    ls_write.append(sistole)
    ls_write.append(diastole)

    with open('DATA_FITUR2.csv','a',newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(ls_write)