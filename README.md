# RC_TwoElementWindkesselModel_PPGDevice

### Rakha Wisnu Bagaskara_Physics Engineering'18

## Description About Repository
This is my final project at physics engineering major, Gadjah Mada University. The title of my final Project is Planning and Design Toe Blood Pressure Measurement System - Based on Photoplethysmography for Peripheral Arterial Disease Early Detection. This repository This repository contains programs used on personal computers with the main programming language, Python. The main purpose of this repository is showing the uses of machine learning algorithm i.e. Random Forest Regressor for prediction the value of R and C in Two Element Windkessel Model. The input for this program is the raw signal of PPG (digital data with ADC in 10bit) and below is the output from this program:
* MAE value for R and C in Two Element Windkessel Model
* Prediction of actual value the Toe Blood Pressure
* User Interface for the blood pressure value

## System Insight
In addition, the following is a block diagram of this system.

<p align="center">
  <img src="https://github.com/BagaskaraRW/RC_TwoElementWindkesselModel_PPGDevice/blob/main/Picture1.png" />
</p>

The Controller used is i.e. Arduino Nano. Basically, data acquisition using microcontroller for acquire the raw data from PPG Sensor (MAX30100). The program i used for the microcontroller and aquire the raw data from the sensor can be accessed via the following link [ArduinoMAX30100 by oxullo](https://github.com/oxullo/Arduino-MAX30100.git). Credit for Oxullo, thanks for your help in my final project.

## Dataset Source and Data Processing

Before make a prediction for blood pressure value, i used dataset that contains PPG raw signal and the value from 219 subject. The dataset can be accessed at [PPG-BP Dataset](https://figshare.com/articles/dataset/PPG-BP_Database_zip/5459299). This dataset used for training and testing blood pressure value and get the minimum MAE (Mean Absolute Error) with tunning hyperparameter i.e n_estimator and max_features. For get this value, there are several file and program that is used for, i.e as follows:
* database.csv
* sqi.csv
* PPG-BP_sqi.csv (the fusion of database.csv and sqi.csv and sqi stand for signal quality index)
* Program_Olah_Sinyal_Dataset.py (this program is used for raw signal processing)
* Program_Model_Estimasi.py (this program is used for build the estimation model from Random Forest Regression Algorithm)

The first step is data acquisition with PPG Sensor that connected to microcontroller. At this step, Arduino Nano has programmed with the acquisition program using Arduino IDE. The raw data obtained is 16 bit data which is the limit of the Analog to Digital Converter (ADC) from the PPG sensor. Arduino Nano had a processor function at this project (not controller), because this item just process the raw data from the sensor to the Main Control Unit (MCU / PC). Data processing occurs in the main control unit. The program that used in MCU for this time as follows:
* Program_PengukuranUtama.py (the main program for prediction BP value)
* identitas.py (the GUI program for patient demographic)
* ppgbp.py (the GUI program for show the actual BP value every 2 second)

All the BP Value is recorded and stored in an internal database.

## Two Element Windkessel Model
The Windkessel Model is a mathematical model that describes the cardiovascular system in humans in the form of electrical circuits. The model can mathematically relate blood flow and blood pressure in the arteries. In this analogy, the flow of arterial blood is described as the flow of fluid through a pipe. In the simplest form of the Windkessel model (two – element Windkessel model), total peripheral resistance or Systemic Vascular Resistance (SVR) and arterial compliance are modeled as Resistance (R in mmHg.s/mL) and Capacitance (C in mL/mmHg).

<p align="center">
  <img src="https://github.com/BagaskaraRW/RC_TwoElementWindkesselModel_PPGDevice/blob/main/PictureTWEM.png" />
</p>

This model is used for build training model and get the minimum MAE value of R and C. After formula derivation and machine learning model building with Random Forest Regression algorithm in Program_Model_Estimasi.py, the following results were obtained.

<p align="center">
  <img src="https://github.com/BagaskaraRW/RC_TwoElementWindkesselModel_PPGDevice/blob/main/PictureR.png" />
</p>
<p align="center">
  <img src="https://github.com/BagaskaraRW/RC_TwoElementWindkesselModel_PPGDevice/blob/main/PictureC.png" />
</p>

The result show the minimum value for is 0.154897 for R and 0.142881 for C with RFR Machine Learning Algorithm.
