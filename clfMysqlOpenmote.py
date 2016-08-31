#!/usr/bin/python

import MySQLdb
import numpy as np
import scipy.stats
import time
#import pickle
import sys, getopt
import math
import multiprocessing

from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import tree 
from sklearn import svm

VERSION = 'PA03'
" First timestamp 'Thu May 19 15:45:30 2016'"
"Put the start (millisecond from 1990) time to indicate the starting time of experiment."
"gap (minute) is the interval between activites."
"and dur (minute) is duration of each activity."

" A1 is sitting" 
" A2 is standing"
" A3 is walking"
" A4 is lying"
" A5 is Running"
" A6 is sitting and standing"
noActivities = 6 
# Obtain data from database for training purpose
# the ID of sensors in the body
CID = 42337
LHID = 42486
RHID = 42509
LFID = 42440
RFID = 42545
ID = np.array([CID, LHID, RHID, LFID, RFID])
DRate = 8
SamplingRate = 64
startIndex = 1
endIndex = 25
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
" Function: Make all collected sensors data having the same label "
" Input: sensors data, e.g. five positions in the body"
" Output: uniformed sensor data "
"========================================================================="
def uniformRawData(d0, d1, d2, d3, d4):
    outd0 = np.array([])
    outd1 = np.array([])
    outd2 = np.array([])
    outd3 = np.array([])
    outd4 = np.array([])
    #activity
    label_index = 26
    Threshold = 1
    #round to all data sensors have full 64 samples
    for i in range(1,noActivities + 1):
        id0 = np.where(d0[:, label_index] == i)
        id1 = np.where(d1[:, label_index] == i)
        id2 = np.where(d2[:, label_index] == i)
        id3 = np.where(d3[:, label_index] == i)
        id4 = np.where(d4[:, label_index] == i)
        temp0 = d0[id0]; temp1 = d1[id1]; temp2 = d2[id2];
        temp3 = d3[id3]; temp4 = d4[id4]
        syst0min = temp0[0,0] 
        syst1min = temp1[0,0] 
        syst2min = temp2[0,0] 
        syst3min = temp3[0,0] 
        syst4min = temp4[0,0] 
        systmin = np.max([syst0min, syst1min, syst2min, syst3min, syst4min])
        deltaMin0 = (systmin - syst0min) / 1000
        deltaMin1 = (systmin - syst1min) / 1000
        deltaMin2 = (systmin - syst2min) / 1000
        deltaMin3 = (systmin - syst3min) / 1000
        deltaMin4 = (systmin - syst4min) / 1000
        if deltaMin0 > Threshold:
            print 'deltaMin0 = ', deltaMin0
            temp0 = temp0[deltaMin0*8:, :]
        if deltaMin1 > Threshold:
            print 'deltaMin1 = ', deltaMin1
            temp1 = temp1[deltaMin1*8:, :]
        if deltaMin2 > Threshold:
            print 'deltaMin2 = ', deltaMin2
            temp2 = temp2[deltaMin2*8:, :]
        if deltaMin3 > Threshold:
            print 'deltaMin3 = ', deltaMin3
            temp3 = temp3[deltaMin3*8:, :]
        if deltaMin4 > Threshold:
            print 'deltaMin4 = ', deltaMin4
            temp4 = temp4[deltaMin4*8:, :]
        syst0max = temp0[len(temp0)-1, 0]
        syst1max = temp1[len(temp1)-1, 0]
        syst2max = temp2[len(temp2)-1, 0]
        syst3max = temp3[len(temp3)-1, 0]
        syst4max = temp4[len(temp4)-1, 0]
        systmax = np.min([syst0max, syst1max, syst2max, syst3max, syst4max])
        deltaMax0 = (syst0max - systmax) / 1000
        deltaMax1 = (syst1max - systmax) / 1000
        deltaMax2 = (syst2max - systmax) / 1000
        deltaMax3 = (syst3max - systmax) / 1000
        deltaMax4 = (syst4max - systmax) / 1000
        if deltaMax0 > Threshold:
            print 'deltaMax0 = ', deltaMax0
            temp0 = temp0[:(len(temp0)-deltaMax0*8), :]
        if deltaMax1 > Threshold:
            print 'deltaMax1 = ', deltaMax1
            temp1 = temp1[:(len(temp1)-deltaMax1*8), :]
        if deltaMax2 > Threshold:
            print 'deltaMax2 = ', deltaMax2
            temp2 = temp2[:(len(temp2)-deltaMax2*8), :]
        if deltaMax3 > Threshold:
            print 'deltaMax3 = ', deltaMax3
            temp3 = temp3[:(len(temp3)-deltaMax3*8), :]
        if deltaMax4 > Threshold:
            print 'deltaMax4 = ', deltaMax4
            temp4 = temp4[:(len(temp4)-deltaMax4*8), :]
        minLeng = np.min([len(temp0), len(temp1), len(temp2), len(temp3), len(temp4)])
        #print 'len(temp0) = ', len(temp0)
        #print 'len(temp1) = ', len(temp1)
        #print 'len(temp2) = ', len(temp2)
        #print 'len(temp3) = ', len(temp3)
        #print 'len(temp4) = ', len(temp4)
        if len(temp0) > minLeng:
            print 'len temp0 = ', len(temp0)
            temp0 = temp0[:minLeng,:]
        if len(temp1) > minLeng:
            print 'len temp1 = ', len(temp1)
            temp1 = temp1[:minLeng,:]
        if len(temp2) > minLeng:
            print 'len temp2 = ', len(temp2)
            temp2 = temp2[:minLeng,:]
        if len(temp3) > minLeng:
            print 'len temp3 = ', len(temp3)
            temp3 = temp3[:minLeng,:]
        if len(temp4) > minLeng:
            print 'len temp4 = ', len(temp4)
            temp4 = temp4[:minLeng,:]
        outd0 = vStackArray(outd0, temp0)
        outd1 = vStackArray(outd1, temp1)
        outd2 = vStackArray(outd2, temp2)
        outd3 = vStackArray(outd3, temp3)
        outd4 = vStackArray(outd4, temp4)
    return outd0, outd1, outd2, outd3, outd4



"========================================================================="
" Function: calculate the amplitude of X, Y, Z accelerometer sensors      "
" Output: R =sqrt(X^2 + Y^2 + Z^2)                                        "
"========================================================================="
def distanceR(temp):
    return math.sqrt(temp[0]*temp[0]+temp[1]*temp[1]+temp[2]*temp[2])

"========================================================================="
" Function: convert raw sensor data to meaningful sensor data             "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Output:                                                                 "
"         matrix with the same size as input [x,y,z,sensortime, label]    "
"========================================================================="
def convertData(X):
    X = np.int_(X)
    # convert to meaningful data sensors
    t = np.int16(X[:, startIndex:endIndex+1]) / 1000.00
    d = np.array([])
    for i in range(len(t)):
        for j in range(DRate):
            temp = t[i,startIndex+3*j:startIndex+3*(j+1)]
            temp = hStackArray(temp, distanceR(temp)) 
            temp = hStackArray(temp, X[i,endIndex:27])
            d = vStackArray(d, temp)
    return d

def chopData(T):
    chopTemp = np.array([])
    t = np.int16(T[startIndex:endIndex+1]) / 1000.00
    for j in range(DRate):
        temp = t[startIndex+3*j:startIndex+3*(j+1)]
        temp = hStackArray(temp, distanceR(temp)) 
        temp = hStackArray(temp, T[endIndex:27])
        chopTemp = vStackArray(chopTemp, temp)
    return chopTemp

"========================================================================="
" Function: Parallel  convert raw sensor data to meaningful sensor data   "
" For example: acc_x = 63896; then convert to acc_x = -1.64               "
" Input : raw sensor data with multi-dimensional sensor data              "
" Output:                                                                 "
"         matrix with the same size as input [x,y,z,sensortime, label]    "
"========================================================================="
def convertDataParallel(X):
    numCPU = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(numCPU)
    X = np.int_(X)
    d = np.array([])
    tasks = []
    i=0
    # Build a list of task
    while i < len(X):
        tasks.append(X[i])
        i += 1
    # Run task in parallel
    results = [pool.apply_async( chopData, (t,)) for t in tasks]
    # Getting results
    for result in results:
        d = vStackArray(d, result.get())
    return d

"========================================================================="
" Function: Read sensor from position i with the contains                 "
"           (systemtime, acc_x, acc_y, acc_z, ..., sensortime, activity)  "
" Input: i is index position of sensor                                    "
" Output: sensor data                                                     " 
"========================================================================="
def readSensorData(cursor, table, i):
    sql="SELECT systemtime, x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4 \
         , x5, y5, z5, x6, y6, z6, x7, y7, z7, x8, y8, z8, sensortime, activity \
         FROM %s WHERE node_id = %s AND activity IS NOT NULL"
    cursor.execute(sql %(table, ID[i]))
    temp = cursor.fetchall()
    return np.asarray(temp)


"========================================================================="
"========================================================================="
def roundUpRawData(X):
    # round down to 64 times
    index = 25 
    X = X[X[:,index].argsort()]
    t0 = X[0,index]
    tmax = X[len(X) - 1,index] 
    t = t0
    XX = np.array([])
    while (t <= tmax):
        idx = np.where(X[:,index] == t)
        temp = X[idx]
        lenT = len(temp)
        if lenT >= 3 and lenT < 8:
            for i in range(8 - lenT):
                temp = vStackArray(temp, temp[0,:])
            XX = vStackArray(XX, temp)
        elif lenT == 8:
            XX =vStackArray(XX, temp)
        else: 
            pass
            # There are lots of missing data?
            # print 't = ', t, 'lenT = ', lenT
        t += 1
    return XX
     


    
"========================================================================="
' Feature extraction' 
' For example: size = 5s, data ' 
" Return indecies of sequence                                             "
"========================================================================="
def slidingWindow(sequence, winSize, step):
    try: it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if step > winSize:
        raise Exception("**ERROR** type(winSize) and type(step) must be in int")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize)/step) + 1
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]


"========================================================================="
" Autocorrelation function                                                "
"========================================================================="
def autocorr(x):
    result = numpy.correlate(x, x, mode='full')    
    return result[result.size/2:]

def feature_unit(X):
    try:
        return np.array([np.min(X), np.max(X), np.mean(X),
                 np.var(X), scipy.stats.skew(X), scipy.stats.kurtosis(X)])
    except ValueError:
        print X

def extract_feature(sensor):
    X = feature_unit(sensor[:,0])
    Y = feature_unit(sensor[:,1])
    Z = feature_unit(sensor[:,2])
    R = feature_unit(sensor[:,3])
    return np.hstack((X,Y,Z,R)) 


def get_feature(S):
    S_fe = np.array([])
    lS_fe =np.array([])
    for i in range(1, noActivities + 1):
        # get all data having the same label
        ic = np.where(S[:,5] == i)
        Si = S[ic]
        Si = Si[:,0:5]
        sTime = Si[0,4]
        sTimeMax = Si[len(Si)-1,4]
        # array contain extracted feature data of each label data
        S_fe_unit = np.array([])
        while sTime < sTimeMax:
            # indecies of data within 2 seconds (window) and having the same label
            ict = np.append(np.where(Si[:,4] == sTime), np.where(Si[:,4] == sTime+1))
            Si_temp = Si[ict]
            if (len(Si_temp) == 0):
                #print sTime
                pass
            else:
                S_fe_temp = extract_feature(Si_temp)
                S_fe_unit = vStackArray(S_fe_unit, S_fe_temp)
            sTime += 1
        #create label according the extracted feature
        lS_fe_temp = np.ones(len(S_fe_unit), dtype=int) * i
        S_fe = vStackArray(S_fe, S_fe_unit)
        lS_fe = np.append(lS_fe, lS_fe_temp)
    return S_fe, lS_fe 




"========================================================================="
" This function define the extracted features"
" Input: X is converted sensor data                                       "
" Output: [Min, Max, Mean, Variance, Skew, Kurtosis, 5 peaks of FFT]   "
"          within window                                                  "
"========================================================================="
def fE(X, winSize = 128, step = 64):
    X_fe = np.array([])
    for wx in slidingWindow(X, winSize, step):
        dft = np.fft.fft(wx)
        if len(X_fe) == 0:
            X_fe = np.array([np.min(wx), np.max(wx), np.mean(wx),
                      np.var(wx), scipy.stats.skew(wx), scipy.stats.kurtosis(wx)])
            # sort and get only 5 peak FFT value in the window, and then sort them in descending
            X_fe = np.append(X_fe, dft.argsort()[-5:][::-1])
        else:
            temp = np.array([np.min(wx), np.max(wx), np.mean(wx),
                      np.var(wx), scipy.stats.skew(wx), scipy.stats.kurtosis(wx)])
            temp = np.append(temp, dft.argsort()[-5:][::-1])
            X_fe = np.vstack((X_fe, temp))
    return X_fe


        
"========================================================================="
"========================================================================="
def main(argv):
    USERNAME = 'vanmao_ngo'
    PASSWORD = 'ngovanmao'
    DATABASE = 'testdb'
    #TABLE    = 'training_data_openmote' 
    TABLE    = 'collect_openmote_20160706_label'
    SVM_STORINGMODEL = 'trainedModel/SVM_HAR_PA1.pkl'
    DCT_STORINGMODEL = 'trainedModel/DCT_HAR_PA1.pkl'
    RF_STORINGMODEL = 'trainedModel/RF_HAR_PA1.pkl'
    print 'Classification SBAN for OpenMote platform, version ' + VERSION
    try:
       opts, args = getopt.getopt(argv,"hu:p:d:t",["username=","password=","database=","table="])
    except getopt.GetoptError:
       print 'classifyMySqlOpenMote.py [-u <username>] [-p <password>] [-d <database>] [-t <table>] [-s <store>]'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'classifyMySqlOpenMote.py [-u <username>][-p <password>][-d <database>][-t <table>][-s <store>]'
          sys.exit()
       elif opt in ("-u", "--username"):
          USERNAME = arg
       elif opt in ("-p", "--password"):
          PASSWORD = arg
       elif opt in ("-d", "--database"):
          DATABASE = arg
       elif opt in ("-t", "--table"):
          TABLE = arg
       elif opt in ("-s", "--store"):
          SVM_STORINGMODEL = arg
    print 'USERNAME is "', USERNAME
    print 'PASSWORD is "', PASSWORD
    print 'DATABASE is "', DATABASE
    print 'TABLE is "', TABLE
    print 'STORING MODEL "', SVM_STORINGMODEL
    # Open database connection
    db = MySQLdb.connect("localhost", USERNAME, PASSWORD, DATABASE)
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    print 'Read sensor from chest'
    rawC  = readSensorData(cursor, TABLE, 0)
    print 'Read sensor from LH'
    rawLH = readSensorData(cursor, TABLE, 1)
    print 'Read sensor from RH'
    rawRH = readSensorData(cursor, TABLE, 2)
    print 'Read sensor from LF'
    rawLF = readSensorData(cursor, TABLE, 3)
    print 'Read sensor from RF'
    rawRF = readSensorData(cursor, TABLE, 4)
    print np.shape(rawRH)
    # Round up raw data 
    print 'Round up raw data for chest sensor'
    rrawC = roundUpRawData(rawC)
    print 'Round up raw data for LH sensor'
    rrawLH = roundUpRawData(rawLH)
    print 'Round up raw data for RH sensor'
    rrawRH = roundUpRawData(rawRH)
    print 'Round up raw data for LF sensor'
    rrawLF = roundUpRawData(rawLF)
    print 'Round up raw data for RF sensor'
    rrawRF = roundUpRawData(rawRF)
    
    # C, LH,.. have format [X, Y, Z, sensortime, label]
    print 'Uniforming raw data'
    (uRawC, uRawLH, uRawRH, uRawLF, uRawRF) = uniformRawData(rrawC, rrawLH, rrawRH, rrawLF, rrawRF)
 
    # convert raw data to refined data with format:
    print 'Convert data to meaningful accelerometers for chest'
    C = convertDataParallel(uRawC)
    print 'Convert data to meaningful accelerometers for LH'
    LH = convertDataParallel(uRawLH)
    print 'Convert data to meaningful accelerometers for RH'
    RH = convertDataParallel(uRawRH)
    print 'Convert data to meaningful accelerometers for LF'
    LF = convertDataParallel(uRawLF)
    print 'Convert data to meaningful accelerometers for RF'
    RF = convertDataParallel(uRawRF)

    print 'RH.shape = ', RH.shape
    ''' Now RH = [X, Y, Z, R, sensortime, label] '''
    print 'Extracting feature vector'
    C_fe, lC_fe = get_feature(C) 
    LH_fe, lLH_fe = get_feature(LH) 
    RH_fe, lRH_fe = get_feature(RH) 
    LF_fe, lLF_fe = get_feature(LF) 
    RF_fe, lRF_fe = get_feature(RF) 

    # TODO: this is a trick to overcome the missing data. Just a funny work around.
    lenCFE = len(C_fe)
    lenLHFE = len(LH_fe)
    lenRHFE = len(RH_fe)
    lenLFFE = len(LF_fe)
    lenRFFE = len(RF_fe)
    maxLenFE = np.max((lenCFE, lenLHFE, lenRHFE, lenLFFE, lenRFFE))
    if lenCFE < maxLenFE:
        for i in range(maxLenFE - lenCFE):
            C_fe = vStackArray(C_fe, C_fe[lenCFE-1,:])
    else:
        lFE = lC_fe
    if lenLHFE < maxLenFE:
        for i in range(maxLenFE - lenLHFE):
            LH_fe = vStackArray(LH_fe, LH_fe[lenLHFE-1,:])
    else:
        lFE = lLH_fe
    if lenRHFE < maxLenFE:
        for i in range(maxLenFE - lenRHFE):
            RH_fe = vStackArray(RH_fe, RH_fe[lenRHFE-1,:])
    else:
        lFE = lRH_fe
    if lenLFFE < maxLenFE:
        for i in range(maxLenFE - lenLFFE):
            LF_fe = vStackArray(LF_fe, LF_fe[lenLFFE-1,:])
    else:
        lFE = lLF_fe
    if lenRFFE < maxLenFE:
        for i in range(maxLenFE - lenRFFE):
            RF_fe = vStackArray(RF_fe, RF_fe[lenRFFE-1,:])
    else:
        lFE = lRF_fe


    " Combine these extracted features of 5 position in X vector "
    " 33 features x 5 sensor = 165 features for each activity ??? Too many features"
    print 'Prepare data for training'
    X = np.hstack((C_fe, LH_fe, RH_fe, LF_fe, RF_fe))
    #X = np.hstack((C_fe, RH_fe))
    #X = RH_fe

    " lable vector is the same for all features. Choose one of them. "
    y = lFE
    
    # Divide training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) 
    #scale data
    #X_train_s = preprocessing.scale(X_train)
    #X_test_s = preprocessing.scale(X_test)
    
    # TODO: Note that, if using scaling data before handling. The result might be high
    # However, when running realtime, the result is terrible.

    #Using SVM:
    clf_svm = svm.SVC(decision_function_shape='ovr')
    clf_svm.fit(X_train, y_train)
    predicted_svm = clf_svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, predicted_svm)
    print("SVM accuracy = ", accuracy_svm)
    joblib.dump(clf_svm, SVM_STORINGMODEL)

    #Using decision tree:
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print("Decision Tree accuracy = ", accuracy)
    joblib.dump(clf, DCT_STORINGMODEL)

    # Using random forest model:
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(n_estimators=10)
    clf_rf.fit(X_train, y_train)
    predicted_rf = clf_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predicted_rf)
    print("Random forest accuracy = ", accuracy_rf)

    #"Store the trained model "
    #s = pickle.dumps(clf)
    joblib.dump(clf_rf, RF_STORINGMODEL)
    # disconnect from server
    db.close()

if __name__ == "__main__":
   main(sys.argv[1:])

quit()
