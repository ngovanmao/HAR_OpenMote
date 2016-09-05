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

VERSION = 'PA05'
" A1 is sitting" 
" A2 is standing"
" A3 is walking"
" A4 is lying"
" A5 is Running"
" A6 is sitting and standing"
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
    t = X[0,index]
    tmax = X[len(X) - 1,index]
    XX = np.array([])
    if t == tmax or tmax - t > 1:
        print '*************Missing too much************'
        return XX
    while (t <= tmax):
        idx = np.where(X[:,index] == t)
        temp = X[idx]
        if len(temp) < 8 and len(temp) > 2:
            for i in range(8 - len(temp)):
                temp = vStackArray(temp, temp[0,:])
            XX = vStackArray(XX, temp)
        elif len(temp) == 8:
            XX =vStackArray(XX, temp)
        else:
            print 'Missing too much. Len = ', len(temp)
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

def readStartTime2(cursor, table, i, systemtime):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND systemtime > %s AND\
           activity IS NULL LIMIT 1"
    cursor.execute(sql % (table, ID[i], systemtime))
    temp = cursor.fetchone()
    return temp[0], temp[1]      

def readLastTime(cursor, table, i):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND activity IS NULL \
           ORDER BY systemtime DESC LIMIT 1"
    cursor.execute(sql % (table, ID[i]))
    temp = cursor.fetchone()
    return temp[0], temp[1]

def readLastTime2(cursor, table, i, systemtime):
    sql = "SELECT systemtime, sensortime FROM %s WHERE node_id = %s AND systemtime > %s AND\
          activity IS NULL ORDER BY systemtime DESC LIMIT 1"
    cursor.execute(sql % (table, ID[i], systemtime))
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
    TABLE    = 'collect_openmote_20160905'
    SVM_STORINGMODEL = 'trainedModel/SVM_HAR_PA1.pkl'
    DCT_STORINGMODEL = 'trainedModel/DCT_HAR_PA1.pkl'
    RF_STORINGMODEL = 'trainedModel/RF_HAR_PA1.pkl'
    print 'Prediction SBAN 5 sensors, version ' + VERSION
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
    #clf = joblib.load(DCT_STORINGMODEL)
    clf = joblib.load(RF_STORINGMODEL)
    # Open database connection
    conn = sqlite3.connect(DATABASE)
    # prepare a cursor object using cursor() method
    cursor = conn.cursor()
    st0, t0 = readLastTime(cursor, TABLE, 0)
    print 'systemtime0 = ', st0, ' sensortime0 = ', t0
    st1, t1 = readLastTime(cursor, TABLE, 1)
    print 'systemtime1 = ', st1, ' sensortime1 = ', t1
    st2, t2 = readLastTime(cursor, TABLE, 2)
    print 'systemtime2 = ', st2, ' sensortime2 = ', t2
    st3, t3 = readLastTime(cursor, TABLE, 3)
    print 'systemtime3 = ', st3, ' sensortime3 = ', t3
    st4, t4 = readLastTime(cursor, TABLE, 4)
    print 'systemtime4 = ', st4, ' sensortime4 = ', t4
    
    stmin = np.min([st0, st1, st2, st3, st4])
    stmax = np.max([st0, st1, st2, st3, st4])
    stmean = np.mean([st0, st1, st2, st3, st4])
    print 'stmin = ', stmin, ' stmax = ', stmax, 'delta = ', stmax - stmin

    #Read before 1second when all sensors are available
    st10, t10 = readLastTime2(cursor, TABLE, 0, stmean - 1000)
    st11, t11 = readLastTime2(cursor, TABLE, 1, stmean - 1000)
    st12, t12 = readLastTime2(cursor, TABLE, 2, stmean - 1000)
    st13, t13 = readLastTime2(cursor, TABLE, 3, stmean - 1000)
    st14, t14 = readLastTime2(cursor, TABLE, 4, stmean - 1000)
    print 'st0 = ', st10, ' st1 = ', st11, 'st2 = ', st12, ' st3 = ', st13, 'st4 = ', st14
    st = np.min([st10, st11, st12, st13, st14])
    rawC  = readSensorData(cursor, TABLE, 0, st)
    print 'rawC.shape', rawC.shape
    rawLH = readSensorData(cursor, TABLE, 1, st)
    print 'rawLH.shape', rawLH.shape
    rawRH = readSensorData(cursor, TABLE, 2, st)
    print 'rawRH.shape', rawRH.shape
    rawLF = readSensorData(cursor, TABLE, 3, st)
    print 'rawLF.shape', rawLF.shape
    rawRF = readSensorData(cursor, TABLE, 4, st)
    print 'rawRF.shape', rawRF.shape

    tC = rawC[:, 25]; tLH = rawLH[:,25]; tRH = rawRH[:,25]; tLF = rawLF[:,25]; tRF = rawRF[:,25]
    tC0 = np.floor(np.mean(tC))
    tLH0 = np.floor(np.mean(tLH))
    tRH0 = np.floor(np.mean(tRH))
    tLF0 = np.floor(np.mean(tLF))
    tRF0 = np.floor(np.mean(tRF))
    print 'starting point = ', tC0, tLH0, tRH0, tLF0, tRF0
    missCount = 0
    while True:
        rawC   = readSensorData2(cursor, TABLE, 0, tC0)
        rawLH  = readSensorData2(cursor, TABLE, 1, tLH0)
        rawRH  = readSensorData2(cursor, TABLE, 2, tRH0)
        rawLF  = readSensorData2(cursor, TABLE, 3, tLF0)
        rawRF  = readSensorData2(cursor, TABLE, 4, tRF0)
        minLen = np.min([len(rawC), len(rawLH), len(rawRH), len(rawLF), len(rawRF)])
        #print 'minLen = ', minLen
        if minLen > 0:
            if missCount > 0:
                missCount  = 0
            # round up raw data upto 16 rows
            #print ' round up raw data up to 16 rows'
            if len(rawC) < 16:
                print 'need to round chest len = ', len(rawC)
                rawC = roundUpRawData(rawC)
                if len(rawC) == 0:
                    print 'skip this time, because missing too much'
                    tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
                    continue
            if len(rawLH) < 16:
                print 'need to round LH len = ', len(rawLH)
                rawLH = roundUpRawData(rawLH)
                if len(rawLH) == 0:
                    print 'skip this time, because missing too much'
                    tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
                    continue
            if len(rawRH) < 16:
                print 'need to round RH len = ', len(rawRH)
                rawRH = roundUpRawData(rawRH)
                if len(rawRH) == 0:
                    print 'skip this time, because missing too much'
                    tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
                    continue
            if len(rawLF) < 16:
                print 'need to round LF len = ', len(rawLF)
                rawLF = roundUpRawData(rawLF)
                if len(rawLF) == 0:
                    print 'skip this time, because missing too much'
                    tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
                    continue
            if len(rawRF) < 16:
                print 'need to round RF len = ', len(rawRF)
                rawRF = roundUpRawData(rawRF)
                if len(rawRF) == 0:
                    print 'skip this time, because missing too much'
                    tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
                    continue

            #print 'Converting raw data '
            C = convertData(rawC)
            LH = convertData(rawLH)
            RH = convertData(rawRH)
            LF = convertData(rawRH)
            RF = convertData(rawRF)
            #RH_fe =  eFeat(RH)
            C_fe = extract_feature(C)
            LH_fe = extract_feature(LH)
            RH_fe = extract_feature(RH)
            LF_fe = extract_feature(LF)
            RF_fe = extract_feature(RF)
            #RH_fe_s =  preprocessing.scale(RH_fe)
            C_fe_s = C_fe.reshape(1,len(C_fe))
            LH_fe_s = LH_fe.reshape(1,len(LH_fe))
            RH_fe_s = RH_fe.reshape(1,len(RH_fe))
            LF_fe_s = LF_fe.reshape(1,len(LF_fe))
            RF_fe_s = RF_fe.reshape(1,len(RF_fe))

            X = np.hstack((C_fe_s, LH_fe_s, RH_fe_s, LF_fe_s, RF_fe_s))
            #X = np.hstack((C_fe_s, RH_fe_s))
            #X = RH_fe_s
            predicted =  clf.predict(X)

            print ("Time = ", tRH0, "predicted = ", predicted)
            #writeActivity(TABLE, predicted, 2, t12)
            show(int(predicted) - 1)
            #time.sleep(0.01)
        else:
            missCount +=1
            if missCount < 10:
                print 'no available input sensor data'
                #print "No input data for a while. Program is exiting!"
                #break
            else:
                pass
        time.sleep(0.8)
        tC0 += 1; tLH0 += 1; tRH0 += 1; tLF0 += 1; tRF0 += 1
    #root.mainloop()
    # disconnect from server
    db.close()

if __name__ == "__main__":
   main(sys.argv[1:])

pygame.quit()
quit()
