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

VERSION = 'PA02'
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
        if len(temp) < 8 and len(temp) > 0:
            for i in range(8 - len(temp)):
                temp = vStackArray(temp, temp[0,:])
            XX = vStackArray(XX, temp)
        elif len(temp) == 8:
            XX =vStackArray(XX, temp)
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
    return np.array([np.min(X), np.max(X), np.mean(X),
                 np.var(X), scipy.stats.skew(X), scipy.stats.kurtosis(X)])

def extract_feature(sensor):
    X = feature_unit(sensor[:,0])
    Y = feature_unit(sensor[:,1])
    Z = feature_unit(sensor[:,2])
    R = feature_unit(sensor[:,3])
    return np.hstack((X,Y,Z,R)) 



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
"Function: Extract feature from matrix [X,Y,Z,label] sensor to features   "
"          for further processing                                         "
"Input:                                                                   "
"          Sensor data [X, Y, Z, sensortime, label]                       "
"          Sensor data [systemtime, X, Y, Z, sensortime, label]           "
"Output:                                                                  "
"          [X_fe, Y_fe, Z_fe, l_fe]                                       "
"          [time, X_fe, Y_fe, Z_fe, l_fe]                                 "
"========================================================================="
def eFeat(sensor):
    X = sensor[:,0]
    Y = sensor[:,1]
    Z = sensor[:,2]
    #R = sqrt(X^2+Y^2+Z^2)
    time = sensor[:,3] 
    label = sensor[:,4]
    X_fe = np.array([])
    Y_fe = np.array([])
    Z_fe = np.array([])
    l_fe = np.array([])
    temp_no_fe = np.array([])
    for i in range(1, noActivities + 1):
        ic = np.where(label == i)
        iX_fe = fE(X[ic])
        iY_fe = fE(Y[ic])
        iZ_fe = fE(Z[ic])
        il_fe = np.ones(len(iX_fe), dtype=int) * i
        X_fe = vStackArray(X_fe, iX_fe)
        Y_fe = vStackArray(Y_fe, iY_fe)
        Z_fe = vStackArray(Z_fe, iZ_fe)
        l_fe = np.append(l_fe, il_fe)
        temp_no_fe = np.append(temp_no_fe, len(il_fe))
    return np.hstack((X_fe, Y_fe, Z_fe)), l_fe, temp_no_fe

"========================================================================="
"========================================================================="
def main(argv):
    USERNAME = 'vanmao_ngo'
    PASSWORD = 'ngovanmao'
    DATABASE = 'testdb'
    #TABLE    = 'training_data_openmote' 
    TABLE    = 'collect_openmote_20160706_label'
    SVM_STORINGMODEL = 'SVM_HAR_PA1.pkl'
    DCT_STORINGMODEL = 'DCT_HAR_PA1.pkl'
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
    print 'Read sensor from RH'
    rawRH = readSensorData(cursor, TABLE, 2)
    print np.shape(rawRH)
    # Round up raw data 
    print 'Round up raw data for RH sensor'
    rawRH = roundUpRawData(rawRH)
    
    # convert raw data to refined data with format:
    print 'Convert data to meaningful accelerometers for RH'
    # TODO: This takes long time for processing. It needs to be improved.
    # by using convertData(), it takes 103seconds. 
    # Parallel multiprocessing speedup the complex computation function
    tStart = time.time()
    RH = convertDataParallel(rawRH)
    tExecute = time.time() - tStart
    print("Conversion execute time = ", tExecute)
    print 'RH.shape = ', RH.shape
    ''' Now RH = [X, Y, Z, R, sensortime, label] '''
    
    RH_fe = np.array([])
    lRH_fe =np.array([])
    for i in range(1, noActivities + 1):
        # get all data having the same label
        ic = np.where(RH[:,5] == i)
        RHi = RH[ic]
        RHi = RHi[:,0:5]
        sTime = RHi[0,4]
        sTimeMax = RHi[len(RHi)-1,4]
        # array contain extracted feature data of each label data
        RH_fe_unit = np.array([])
        while sTime < sTimeMax:
            # indecies of data within 2 seconds (window) and having the same label
            ict = np.append(np.where(RHi[:,4] == sTime), np.where(RHi[:,4] == sTime+1))
            RHi_temp = RHi[ict]
            RH_fe_temp = extract_feature(RHi_temp)
            sTime += 1 
            RH_fe_unit = vStackArray(RH_fe_unit, RH_fe_temp)
        #create label according the extracted feature
        lRH_fe_temp = np.ones(len(RH_fe_unit), dtype=int) * i
        RH_fe = vStackArray(RH_fe, RH_fe_unit)
        lRH_fe = np.append(lRH_fe, lRH_fe_temp)

    print 'Extracting feature vector'
    #RH_fe, lRH_fe, nRH_fe = eFeat(RH)
    " Combine these extracted features of 5 position in X vector "
    " 33 features x 5 sensor = 165 features for each activity ??? Too many features"
    print 'Prepare data for training'
    X = RH_fe
    " lable vector is the same for all features. Choose one of them. "
    y = lRH_fe
    
    # Divide training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0) 
    #scale data
    X_train_s = preprocessing.scale(X_train)
    X_test_s = preprocessing.scale(X_test)
    #X_train_s = X_train
    #X_test_s = X_test
    
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

    # Using random forest model:
    from sklearn.ensemble import RandomForestClassifier
    clf_rf = RandomForestClassifier(n_estimators=10)
    clf_rf.fit(X_train, y_train)
    predicted_rf = clf_rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, predicted_rf)
    print("Random forest accuracy = ", accuracy_rf)


    #"Store the trained model "
    #s = pickle.dumps(clf)
    # For now, we use SVM because DCT is overfit to the data.
    joblib.dump(clf_svm, DCT_STORINGMODEL)
    # disconnect from server
    db.close()

if __name__ == "__main__":
   main(sys.argv[1:])

quit()
