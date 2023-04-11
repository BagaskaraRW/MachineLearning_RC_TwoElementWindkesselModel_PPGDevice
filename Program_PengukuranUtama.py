##### Program Pengukuran Utama BP.py#####
# Import libraries

#For serial communicatio
import serial
import serial.tools.list_ports

#For mathematics
import numpy as np

#For plot data
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph
import sys
from PyQt5.QtCore import (QThread, Qt, pyqtSignal , QCoreApplication)

#For signal processing
from scipy import signal
from scipy . signal import find_peaks
from scipy import integrate

#For file reading
import csv
from datetime import datetime
import time
import os
import pickle

# Layout ui
import ppgbp
import identitas

##################### Here's the Code #####################






com = ''
n = 2000 #n = jumlah sampel yang diambil
w = 4
iterasi = 2
data_olah = [] # data yang diolah
hr = 0
hrv = 0

lsis = [] # list berisi sampel tekanan darah sistole
ldia = [] # list berisi sampel tekanan darah sistole
i = 0 # index iterasi
sig = [] # data sinyal ppg
fil = [] # sinyal ppg terfilter
nor = [] # sinyal ppg ternormalisasi
fitur = [] # fitur sinyal ppg
cyc_a = 0 # area siklus
cyc_t = 0 # waktu siklus
sis_t = 0 # waktu sistolik
dia_t = 0 # waktu diastolik
rsd = 0 # rasio sistolik diastolik
td_sis = [] # sampel tekanan darah sistole
td_dia = [] # sampel tekanan darah diastole
sis = 0 # tekanan darah sistole
dia = 0 # tekanan darah diastole
pts = 70 # Nilai diastole awal
pts0= 70 # Nilai diastole awal

nama = ''
jenis_kelamin = ''
tanggal_lahir = ''
th_lahir = 0
b_lahir = 0
date_lahir = 0

umur = 0
tb = ''
bb = ''
operator = ''

#untuk penamaan file
dateTimeObj = datetime.now()
date = dateTimeObj. strftime ("%d - %m - %Y")

path = ''
filesimpan = ''
filehraw = ''

##Membaca file model estimasi
with open("Rf_Wind_R.pkl", "rb") as file:
    model_R = pickle.load(file)
with open("Gb_Wind_C.pkl", "rb") as file:
    model_C = pickle.load(file)




# Thread untuk mengolah data
class Thread(QThread):
    signal_sis = pyqtSignal(float)
    signal_dia = pyqtSignal(float)

    def run (self):
        sinyal = self.akuisisi_sinyal()

        global sis, dia

        sis, dia = self.tekanan_darah(sinyal, model_R, model_C)

        # simpan semua data
        self.simpan_data(sis, dia, filesimpan)
        
        # simpan raw
        for raw in sinyal :
            self.simpan_raw(raw, filehraw)

        #Emit nilai sis dan dia
        self.signal_sis.emit(sis)
        self.signal_dia.emit(dia)

    def akuisisi_sinyal(self):
        global com, n, listdata

        listdata = []

        for i in range(n):
            while(com.inWaiting()==0):
                pass
            serdata = str(com.readline(),"utf-8")
            data_str = serdata.replace ('\r\n','')
            #print(data_str)
            if len(data_str)<6 and len(data_str)>0:
                data_int = int(data_str)
                listdata.append(data_int)

        sinyal = listdata
        return(sinyal)
    
    def filter_bp(self, sig):     #fungsi untuk filter sinyal
        srate = 150            #Sampling rate
        fcl= 10                 #frekuensi cuttoff
        fch= 0.5
        wnl = fcl/(srate/2)     #frekuensi angular
        wnh = fch/(srate/2)

        b,a = signal.cheby2(4, 20, wnh, 'high')
        fil = signal.filtfilt(b,a, sig)

        b,a = signal.cheby2(4, 20, wnl, 'low')
        fil = signal.filtfilt(b,a, fil)

        return(fil)
    
    def normalisasi_bp(self, ppg):    #fungsi untuk normalisasi standar deviasi
        mean = np.mean(ppg)
        std = np.std(ppg)
        ppg = [((i - mean)/std) for i in ppg]
        return(ppg)

    def normalisasi_max_bp(self, fil):   #fungsi untuk normalisasi min - max
        maks = max(fil)
        mins = min(fil)
        fil = [((i - mins)/(maks - mins)) for i in fil]
        return(fil)
    
    def cari_batas(self, ppg, vpg, apg): #fungsu untuk mencari batas sinyal
        apg = self.normalisasi_bp(apg)
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
    
    def potong_bp(self, batas, ppg, vpg, apg):   #Fungsi untuk memotong siklus sinyal
        ppg = ppg[batas[0]: batas[1]]
        vpg = vpg[batas[0]: batas[1]]
        apg = apg[batas[0]: batas[1]]
        return(ppg, vpg, apg)
    
    def deteksi_wave(self, ppg, vpg, apg):    #Fungsi untuk mendeteksi karakter sinyal
        global indx, indx_O, indx_S, indx_O2, indx_wyz, indx_y, indx_a, indx_b, indx_ab, indx_iapg, indx_de, indx_cde, indx_c 
        global indx_e, indx_d, indx_D, indx_w, indx_y, indx_z, O, S, D, O2, w, y, z, a, b, c, d, e
        
        ppg_raw = ppg

        ppg = self.normalisasi_bp(ppg)
        vpg = self.normalisasi_bp(vpg)
        apg = self.normalisasi_bp(apg)
        vpg_nor = self.normalisasi_bp(vpg)
        apg_nor = self.normalisasi_bp(apg)

        indx = [i for i in range(len(ppg))]

        indx_O = indx[0]
        indx_O2 = indx[-1]
        indx_S = signal.find_peaks(ppg, prominence = 0.01)[0][0]
        indx_wyz = signal.find_peaks(vpg, prominence = 0.01)[0]

        vpg_inv = [-i for i in vpg_nor]
        indx_y = signal.find_peaks(vpg_inv)[0][0]

        apg_abs = [abs(i) for i in apg_nor]
        indx_a = signal.find_peaks(apg_abs, prominence = 0)[0][0]
        indx_b = signal.find_peaks(apg_abs, prominence = 0)[0][1]

        apg_inv = [-i for i in apg_nor]
        indx_ab = signal.find_peaks(apg_nor)[0]
        indx_iapg = signal.find_peaks(apg_inv)[0]
        indx_de = [i for i in indx_iapg if i > indx_b and i < 400]
        indx_cde = [i for i in indx_ab if i > indx_b and i < 400]
        indx_w = indx_wyz[0]
        indx_z = indx_wyz[1]

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

        ppg = self.normalisasi_max_bp(ppg_raw)
        vpg = (self.filter_bp(np.gradient(ppg)))
        apg = (self.filter_bp(np.gradient(vpg)))

        if len(indx_wyz)>1:
            indx_w = indx_wyz[0]
            indx_z = indx_wyz[1]
        
        else:
            indx_w = indx_wyz[0]
            indx_z = int(round((indx_D + indx_e)/2))

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

    def ekstrak_fitur(wave, ppg, vpg, apg):  #Fungsi untuk mengekstrasi fitur sinyal
    
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
        
    def estimasi(self, fitur, model_R, model_C): # fungsi untuk estimasi tekanan darah dari fitur
        global pts
        tc = fitur [-3]
        ts = fitur [-2]
        td = fitur [-1]
        X = [ fitur [0:25]]
        integran = lambda x: np.sin(np.pi*x / ts)
        integ, _ = integrate.quad(integran, 0, ts)
        CO = 5
        io = (round(1000*CO*tc / (60*(integ ) ) ,2) )
        R = model_R.predict(X)[0]
        C = model_C.predict(X)[0]
        ps = pts*np.exp(-ts/(R*C)) + ( (io*ts*C*np.pi*R*R)/(ts*ts + C*C *np.pi* np.pi * R*R) ) * (1+np.exp(-ts/(R*C)))
        ptd = ps
        pd = ptd*np.exp(-td/(R*C))
        pts = pd
        if pts < -1 or pts >200:
            pts = pts0
            ps = pts*np.exp(-ts/(R*C)) + ( (io*ts*C*np.pi*R*R)/(ts*ts + C*C *np.pi* np.pi * R*R) ) * (1+np.exp(-ts/(R*C)))
            ptd = ps
            pd = ptd*np.exp(-td/(R*C))
        return(ps, pd)

    def tekanan_darah(self, sinyal, model_R, model_C):
        lsis = [] 
        ldia = []
        errtd = 0

        sig = signal.resample(sinyal, len(sinyal)*10)

        ppg = self.filter_bp(sig)
        vpg = self.filter_bp(np.gradient(ppg))
        apg = self.filter_bp(np.gradient(vpg))

        try:
                    
            batas = self.cari_batas(ppg, vpg, apg)
            ppg, vpg, apg = self.potong_bp(batas, ppg, vpg, apg)

            wave = self.deteksi_wave(ppg, vpg, apg)

            raw = ppg
            ppg = self.normalisasi_max_bp(ppg)
            vpg = self.filter_bp(np.gradient(ppg))
            apg = self.filter_bp(np.gradient(vpg))

            fitur = self.ekstrak_fitur(wave, ppg, vpg, apg)
            ps, pd = self.estimasi(fitur, model_R, model_C)
    
        except:
            sis = 0
            dia = 0

        if sis<0 or dia<0:
            sis = 0
            dia = 0

        sis = round(sis)
        dia = round(dia)
        return(sis, dia)
    
    def simpan_data(self, dat, dat2, file):
        dateTime = datetime.now()
        time = dateTime.strftime("%H:%M:%S")

        with open(path+"/_DATA"+file,'a',newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")
            writer.writerow([time, dat, dat2])
    
    def simpan_raw(self, dat, file):
        dateTime = datetime.now()
        time = dateTime.strftime("%H:%M:%S")
        with open(path+"/_RAW"+file,'a',newline="") as outfile:
            writer = csv.writer(outfile, delimiter=",")
            writer.writerow([time, dat])

class App(QtGui.QMainWindow, ppgbp.Ui_MainWindow):
    global n, com, dia, sis, path

    path = os.getcwd()

    def buka_port(self):
        ports =  serial.tools.list_ports.comports(include_links =False)
        for port in ports:
            print("Find Port"+port.device)

        com = serial.Serial(port.device)
        
        if com.isOpen():
            com.close()
        com = serial.Serial(port.device, 115200) #timeout=1)
        com.flushInput()
        com.flushOutput()
        com.flushInput()

        return(com)

    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'k')
        super(App, self).__init__(parent)
        self.setupUi(self)
        self.tombol_mulai.clicked.connect(self.mulai)
        self.tombol_selesai.clicked.connect(self.tutup)
        self.grafik_sistole.plotItem.showGrid(True, True, 0.7)
        self.grafik_sistole.plotItem.setYRange(0, 200, padding=0)
        self.grafik_sistole.plotItem.setTitle ('Tekanan Darah')
        self.grafik_sistole.plotItem.setLabel('left', 'Tekanan Darah', units='mmHg')
        self.grafik_sistole.plotItem.setLabel('bottom', 'Waktu', units='s')
        self.axistd = self.grafik_sistole.plotItem.getAxis('bottom')
        self.axistd.setTickSpacing(20,2)
        self.legendtd = self.grafik_sistole.plotItem.addLegend(size=(1,1), offset=(55,30))
        self.legendtd.setParentItem(self.grafik_sistole.plotItem)
        self.grafik_sistole.plot([0], [0], pen=pyqtgraph.mkPen(color = '#DC143C', width = 0.3), symbol = 'o', symbolBrush=pyqtgraph.mkBrush(color = '#DC143C'), name = '-sistole')
        self.grafik_sistole.plot([0], [0], pen=pyqtgraph.mkPen(color = '#4169E1', width = 0.3), symbol = 'o', symbolBrush=pyqtgraph.mkBrush(color = '#4169E1'), name = '-diastole')

        self.xs = [0]
        self.ys = [0]
        self.xd = [0]
        self.yd = [0]

    def mulai(self):
        global com, filesimpan, filehraw, date

        com = self.buka_port()

        for i in range(5):
            serdata = str(com.readline(),"utf-8")
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(2000)
        self.timerwaktu = QtCore.QTimer()
        self.timewaktu = QtCore.QTime(0, 0, 0)
        self.timerwaktu.timeout.connect(self.timerwaktuEvent)
        self.timerwaktu.start(1000)
        time.sleep(2)
        self.timer_BP = QtCore.QTimer()
        self.timer_BP.timeout.connect(self.timerEvent_BP)
        self.timer_BP.start(2000)
        timeObj = datetime.now()
        jam = timeObj.strftime("%H - %M")

        # File penyimpanan data
        filesimpan = nomor_show +''+ date +''+ jam +'.csv'
        filehraw = nomor_show +''+ date +'' + jam +"-RAW" + '.csv'

        with open(path+"/_DATA/" + filesimpan, 'a', newline='' ) as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow ([nama,str(umur),str(tb),str(bb), operator,'-'])
            writer.writerow (["Waktu","Sistole (mmHg)","Diastole(mmHg)"])
        
        # tampilan identitas
        #self.tanggal.setText(date)
        #self.nomor_id.setText(nomor_show)
        #self.nama_klien.setText(nama)
        #self.kelamin.setText(jenis_kelamin)
        #self.usia.setText(str(umur))
        #self.tinggi.setText(str(tb))
        #self.berat.setText(str(bb))
        #self.nama_operator.setText(operator)

    def timerwaktuEvent(self) :
        self.timewaktu = self.timewaktu.addSecs(1)
        self.tampilan_timer = self.timewaktu.toString("hh:mm:ss")
        self.waktu.setText(self.tampilan_timer)

    def tutup (self):
        sys.exit(app.exec_())
        sys.exit(identity.exec_())
        com.close()

    def closeEvent (self, event) :
        print ("User has clicked the red x on the main window")
        event.accept()
        self.tutup()

    def timerEvent_BP(self) :
        global n,w,hr, hrv, est , sis , dia
        self.sis = sis
        self.dia = dia

        # append tekanan
        self.xs.append(self.xs[-1] + 2)
        self.ys.append(self.sis)
        self.xd.append(self.xd[-1] + 2)
        self.yd.append(self.dia)

        #############################################################
        
        # cut tekanan
        self.cutxs = self.xs[-30:]
        self.cutys = self.ys[-30:]
        self.cutxd = self.xd[-30:]
        self.cutyd = self.yd[-30:]
        self.grafik_sistole.clear()
        self.grafik_sistole.plot(self.cutxs, self.cutys, pen=pyqtgraph.mkPen(color = '#DC143C', width = 0.3),symbol = 'o', symbolSize=5,symbolBrush=pyqtgraph.mkBrush(color = '#DC143C'))
        self.grafik_sistole.plot(self.cutxs, self.cutyd, pen=pyqtgraph.mkPen(color = '#4169E1', width = 0.3), symbol = 'o', symbolSize=5, symbolBrush=pyqtgraph.mkBrush(color = '#4169E1'))
        
        # tampilan angka tekanan
        labels = (str(round(self.ys[-1])) )
        labeld = (str(round(self.yd[-1])))
        self.label_sistole.setText(labels)
        self.label_diastole.setText(labeld)

    def timerEvent(self):
        global n, w, hr, hrv, est, sis, dia

        thread = Thread(self)
        thread.start()

        try:
            #Raw
            self.raw = (listdata)
            # append x raw
            self.x_raw = []
            for i in range(len(self.raw)):
                self.x_raw.append(i/100)
        except KeyboardInterrupt :
            com.close()
        
# Thread input data form

class Identity (QtGui.QMainWindow, identitas.Ui_Input):
    def __init__ (self , parent=None):
        pyqtgraph.setConfigOption('background', 'k') #before loading widget
        super(Identity, self). __init__(parent)
        #Hilangkan Tombol Close
        self.setWindowFlags(
        QtCore.Qt.Window |
        QtCore.Qt.CustomizeWindowHint |
        QtCore.Qt.WindowTitleHint |
        QtCore.Qt.WindowStaysOnTopHint)
        self.setupUi(self)
        self.app = App()
        #pencet tombol oke
        self.tombol_ok_baru.clicked.connect(self.oke_baru)
        self.tombol_ok_lama.clicked.connect(self.oke_lama)
        #Check pasien lama/baru
        self.groupBox_klienlama.toggled.connect(self.checklama)
        self.groupBox_klienbaru.toggled.connect(self.checkbaru)
    def checklama(self):
        if self.groupBox_klienlama.isChecked and self.groupBox_klienbaru.isChecked:
            self.groupBox_klienbaru.setChecked(False)
        else:
            self.groupBox_klienbaru.setChecked(True)
    def checkbaru(self):
        if self.groupBox_klienlama.isChecked and self.groupBox_klienbaru.isChecked:
            self.groupBox_klienlama.setChecked(False)
        else:
            self.groupBox_klienlama.setChecked(True)
    
    def oke_lama(self ) :
        global date, nama, jenis_kelamin, tanggal_lahir, th_lahir, b_lahir, date_lahir, umur, tb, bb, operator, nomor_show
        global filesimpan, filehraw
        nomor = int( self.input_id.text())
        tb = self.input_tb_lama.text()
        bb = self.input_bb_lama.text()
        operator = self.input_op_lama.text()
        operator = operator.capitalize()

        with open("database.csv", newline ='') as csvfile :
            spamreader = csv.reader(csvfile, delimiter = ',')
            list_baris =[]

        for row in spamreader:
            list_baris.append(row)

        datanya = list_baris[nomor]
        nomor_show = str ("{:06d}".format(nomor))
        nama = datanya[1]
        jenis_kelamin = datanya[2]
        tanggal_lahir = datanya[3] # ukur lagi
        th_lahir = int(tanggal_lahir[0:4])
        b_lahir = int(tanggal_lahir[5:6])
        date_lahir = int(tanggal_lahir[7:8])
        t_now = date.split('-')
        th_now = int(t_now[2])
        b_now = int(t_now[1])
        date_now = int(t_now[0])
        umur = th_now - th_lahir

        if b_now >= b_lahir and date_now >= date_lahir :
            umur = umur

        else :
            umur = umur - 1
            
        form2.close()
        form.show()
    
    def oke_baru(self):
        global date, nama, jenis_kelamin, tanggal_lahir, th_lahir, b_lahir, date_lahir, umur, tb, bb, operator, nomor_show
        global filesimpan, filehraw
        print('OKE')
        nama = self.input_nama.text()
        nama = nama.capitalize()
        # print (nama)
        jenis_kelamin = str (self.input_jk.currentText())
        # print ( jenis_kelamin )

        #generate nomor ID
        with open("database.csv", newline ='') as csvfile :
            spamreader = csv.reader(csvfile, delimiter= ',')
            list_baris = []
            for row in spamreader:
                list_baris.append(row)

        nomor = int(list_baris[-1:][0][0]) + 1
        nomor_show = str("{:06d}".format(nomor))

        # lahir
        ttl = self.input_tl.date()
        tanggal_lahir = str(ttl.toPyDate()).split('-')
        th_lahir = int(tanggal_lahir[0])
        b_lahir = int(tanggal_lahir[1])
        date_lahir = int(tanggal_lahir[2])
        tgl_lahir = ''
        for i in tanggal_lahir:
            tgl_lahir = tgl_lahir + i

        # sekarang
        t_now = date.split('-')
        th_now = int(t_now[2])
        b_now = int(t_now[1])
        date_now = int(t_now[0])

        umur = th_now - th_lahir
        if b_now >= b_lahir and date_now >= date_lahir :
            umur = umur
        else :
            umur = umur - 1
            tb = self.input_tb.text()
            bb = self.input_bb.text()
            operator = self.input_op.text()
            operator = operator.capitalize()

        # Membuat file database
        with open('database .csv', 'a', newline='' ) as outfile:
            writer = csv.writer(outfile, delimiter =",")
            writer.writerow([nomor,nama,jenis_kelamin,tgl_lahir])
        form2.close()
        #form.show()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    #identity = QtGui.QApplication(sys.argv)
    form = App()
    form.show()
    #form2 = Identity()
    #form2.show()
    app.exec_()
    #identity.exec_()







