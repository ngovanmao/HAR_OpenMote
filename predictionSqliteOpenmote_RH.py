#!/usr/bin/python

import sqlite3
import numpy as np
import scipy.stats
import time
import pygame
import sys, getopt
import math

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import svm

VERSION = 'PA03'
" A1 is sitting A2 is standing A3 is walking"
" A4 is lying A5 is Running A6 is sitting and standing"
noActivities = 6
dRate = 8
SamplingRate = 64
startIndex = 1
endIndex = 25

Width=512
Height = 512 
pygame.init()
gameDisplay = pygame.display.set_mode((Width, Height))
myFont = pygame.font.SysFont("monospace", 25)
pygame.display.set_caption('Activity Prediction')
clock = pygame.time.Clock()

img = ['sitting.png', 'standing.png', 'walking.png', 'lying.png', "running.png", 'sittingStanding.png', 'idle.jpg']
white = (255,255,255)

"========================================================================="
" Function: Display the activity image                                    "
" Input: activity index                                                   "
"========================================================================="
def show(activity):
    activityImg = pygame.image.load(img[activity])
    label = myFont.render(img[activity], 1, (255,0,0))
    x = Width * 0
    y = Height * 0
    gameDisplay.fill(white)
    gameDisplay.blit(activityImg, (x,y))
    gameDisplay.blit(label, (x, y))
    pygame.display.update()
    

# Obtain data from database for training purpose
CID = 42337
LHID = 42486
RHID = 42509
LFID = 42440
RFID = 42545
ID = np.array([CID, LHID, RHID, LFID, RFID])

"========================================================================="
" Function: Stack matrix add to data with horizontal direction"
" Output: data with added matrix"
"========================================================================="
def hStackArray(data, add):
    if len(data) > 0:
        data = np.hstack((data, add))
    else:
        data = add
    return data

"========================================================================="
" Function: Stack matrix add to data with vertical direction"
" Output: data with added matrix"
"========================================================================="
def vStackArray(data, add):
    if len(data) > 0:
        data = np.vstack((data, add))
    else:
        data = add
    return data

"========================================================================="
" Function: Each window = 2second, that means we have two timestamps in data"
"           This function is to round up the missing samples within window"
" Input: Data in window"
" Output: Full data for that window"
"========================================================================="
def roundUpRawData(X):
    # round down to 64 times
    index = 25
    X = X[X[:,index].argsort()]
    t0 = X[0,index]
    tmax = X[len(X) - 1,index]
    if t0 == tmax or tmax - t0 > 1:
        print '*************Missing too much************'
    t = t0
    XX = np.array([])
    while (t <= tmax):
        idx = np.where(X[:,index] == t)
        temp = X[idx]
        if len(temp) < 8 and len(temp) > 0:
            for i in range(8 - len(temp)):
                temp = vStackArray(temp, temp[0,:])
            XX = vStackArray(XX, temp)
        elif len(temp) == 8:
            XX =vStackArray(XX, temp)
        t += 1
    return XX

def distanceR(temp):
    return math.sqrt(temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2])


"========================================================================="
" Function: convert raw sensor data to meaningful sensor data             "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Input: raw data "
" Output:                                                                 "
"         matrix with the same size as input [x,y,z,sensortime]           "
"========================================================================="
def convertData(X):
    #X = np.asarray(X)
    X = np.int_(X)
    # convert to meaningful data sensors
    t = np.int16(X[:,startIndex:endIndex+1]) / 1000.00
    d = np.array([])
    for i in range(len(t)):
        for j in range(dRate):
            #tempIndex = range(startIndex+3*j, startIndex+3*(j+1)) + range(endIndex, 26)
            temp = t[i,startIndex+3*j : startIndex+3*(j+1)]
            temp = hStackArray(temp, distanceR(temp))
            temp = hStackArray(temp, X[i,endIndex])
            d = vStackArray(d, temp)
    return d

"========================================================================="
" Function: Read sensor from position i with the contains                 "
"           (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)  "
" Input: i is index position of sensor and sending timestamp              "
" Output: sensor data                                                     "
"========================================================================="
def readSensorData(cursor, table, i, st):
    sql="SELECT systemtime, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, \
         x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, sensortime \
         FROM %s WHERE (node_id = %s) AND (systemtime > %s) AND \
         (activity IS NULL) LIMIT 16"
    cursor.execute(sql %(table, ID[i], int(st)))
    temp = cursor.fetchall()
    return np.asarray(temp)
#    return convertData(temp)

def readSensorData2(cursor, table, i, t):
    sql="SELECT systemtime, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, \
         x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, sensortime \
         FROM %s WHERE (node_id = %s) \
         AND (sensortime = %s OR sensortime = %s) AND \
         (activity IS NULL) LIMIT 16"
    cursor.execute(sql %(table, ID[i], int(t), int(t)+1))
    temp = cursor.fetchall()
    return np.asarray(temp)

def readStartTime(cursor, table, i):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND activity IS NULL LIMIT 1"
    cursor.execute(sql % (table, ID[i]))
    temp = cursor.fetchone()
    return temp[0], temp[1]      

def readLastTime(cursor, table, i):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND activity IS NULL \
           ORDER BY systemtime DESC LIMIT 1"
    cursor.execute(sql % (table, ID[i]))
    temp = cursor.fetchone()
    return temp[0], temp[1]

"========================================================================="
" This function define the extracted features                             "
" Input: X is converted sensor data                                       "
" Output: [Min, Max, Mean, Variance, Skew, Kurtosis, 5 peaks of FFT]      "
"          within window                                                  "
"         If we don't use FFT, each dimension sensor extracts to 6 features"
"========================================================================="
def feature_unit(X):
    return np.array([np.min(X), np.max(X), np.mean(X),
                 np.var(X), scipy.stats.skew(X), scipy.stats.kurtosis(X)])

def extract_feature(sensor):
    X = feature_unit(sensor[:,0])
    Y = feature_unit(sensor[:,1])
    Z = feature_unit(sensor[:,2])
    R = feature_unit(sensor[:,3])
    return np.hstack((X,Y,Z,R))


"========================================================================="
" Function: Write the predicted value to activity field in MySQL database "
"========================================================================="
def writeActivity(table, activity, i, t):
    wSql = "UPDATE %s SET activity = %s WHERE node_id = %s AND \
            (sensortime = %s or sensortime = %s) AND activity IS NULL LIMIT 8"
    cursor.execute(wSql % (table, int(activity), ID[i], t, t+1))
    db.commit()

"========================================================================="
"========================================================================="
def main(argv):
    DATABASE = '/home/user/OpenMote/contiki/tools/collect-bdopenmote/dist/test.db'
    TABLE    = 'collect_openmote_20160904'
    SVM_STORINGMODEL = 'trainedModel/SVM_HAR_RH_PA1.pkl'
    DCT_STORINGMODEL = 'trainedModel/DCT_HAR_RH_PA1.pkl'
    print 'Prediction SBAN, version ' + VERSION
    try:
       opts, args = getopt.getopt(argv,"h:d:t:s",["database=","table=","store="])
    except getopt.GetoptError:
       print 'predictionMySqlOpenMote.py [-d <database>] [-t <table>][-s <store>]'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'predictionMySqlOpenMote.py [-d <database>] [-t <table>][-s <store>]'
          sys.exit()
       elif opt in ("-d", "--database"):
          DATABASE = arg
       elif opt in ("-t", "--table"):
          TABLE = arg
       elif opt in ("-s", "--store"):
          SVM_STORINGMODEL = arg

    print 'DATABASE is "', DATABASE
    print 'TABLE is "', TABLE
    print 'TRAINED_MODEL is "', SVM_STORINGMODEL
    #clf = joblib.load(SVM_STORINGMODEL)
    clf = joblib.load(DCT_STORINGMODEL)
    # Open database connection
    conn = sqlite3.connect(DATABASE)
    # prepare a cursor object using cursor() method
    cursor = conn.cursor()
    st, t = readLastTime(cursor, TABLE, 2)
    print 'systemtime = ', st, ' sensortime = ', t
    
    rawRH = readSensorData(cursor, TABLE, 2, st)
    print 'rawRH.shape', rawRH.shape
    tRH = rawRH[:,25]
    tRH0 = np.floor(np.mean(tRH))
    print 'starting point = ', tRH0
    missCount = 0
    while True:
        rawRH  = readSensorData2(cursor, TABLE, 2, tRH0) 
        minLen = len(rawRH)
        if minLen > 0:
            if missCount > 0:
                missCount  = 0
            # round up raw data upto 16 rows
            if len(rawRH) < 16:
                print 'need to round RH len = ', len(rawRH)
                rawRH = roundUpRawData(rawRH)
                if len(rawRH) == 0:
                    print 'skip this time, because missing too much'
                    tRH0 += 1
                    continue
            #print 'Converting raw data '
            RH = convertData(rawRH)
            RH_fe = extract_feature(RH)
            #RH_fe_s =  preprocessing.scale(RH_fe)
            RH_fe_s =  RH_fe
            RH_fe_s = RH_fe_s.reshape(1,len(RH_fe_s))
            predicted =  clf.predict(RH_fe_s)

            print ("Time = ", tRH0, "predicted = ", predicted)
            #writeActivity(TABLE, predicted, 2, t12)
            show(int(predicted) - 1)
            #time.sleep(0.01)
        else:
            missCount +=1
            if missCount < 10:
                print 'no available input sensor data'
            else: 
                pass
                #print "No input data for a while. Program is exiting!"
                #break
        time.sleep(1)
        tRH0 += 1
    #root.mainloop()
    # disconnect from server
    db.close()

if __name__ == "__main__":
   main(sys.argv[1:])

pygame.quit()
quit()
