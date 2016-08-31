#!/usr/bin/python
import MySQLdb
import time
import sys, getopt

" A1 is sitting" 
" A2 is standing"
" A3 is walking"
" A4 is lying"
" A5 is Running"
" A6 is sitting and standing"


def writeLabel(db, cursor, table, label, start, end):
    print table
    # Prepare SQL query to UPDATE label activities to training datasets 
    sql = "UPDATE %s SET activity = %s WHERE (systemtime > %s) AND (systemtime < %s)" %\
            (table, int(label), int(start), int(end))
    try:
        cursor.execute(sql)
        db.commit()
    except:
        db.rollback()
        print "Error: unable to fetch data"

def atoi(str):
    INT_MAX, INT_MIN = 2147483647, -2147483648
    index = 0
    result = 0
    sign = 1        # Default we are processing a non-negative number
    if len(str) == 0:   return 0    # The result for empty string is 0.
    while str[index].isspace():     index += 1  # Discard whitespace
    if str[index] == "-":           sign = -1
    if str[index] in "-+":          index += 1  # Discard sign char
    while index < len(str) and str[index].isdigit():
            result = result * 10 + (ord(str[index]) - ord("0")) * sign
            index += 1

            # Handle overflow. Because Python could handle the large
            # integer, this code works well. If with C/C++, we should use
            # another method. For example, with non-negative number, the
            # test condition for overflow would be:
            # if( INT_MAX / 10 < result || INT_MAX - result * 10 <
            #    (ord(str[index]) - ord("0")) )
            if sign == 1 and result >= INT_MAX:     return INT_MAX
            elif sign == -1 and result <= INT_MIN:  return INT_MIN

    return result

def atol(string):
   # INT_MAX, INT_MIN = 2147483647, -2147483648
    index = 0
    result = 0
    sign = 1        # Default we are processing a non-negative number
    if len(string) == 0:   return 0    # The result for empty string is 0.
    while string[index].isspace():     index += 1  # Discard whitespace
    if string[index] == "-":           sign = -1
    if string[index] in "-+":          index += 1  # Discard sign char
    while index < len(string) and string[index].isdigit():
            result = result * 10 + (ord(string[index]) - ord("0")) * sign
            index += 1
            # Handle overflow. Because Python could handle the large
            # integer, this code works well. If with C/C++, we should use
            # another method. For example, with non-negative number, the
            # test condition for overflow would be:
            # if( INT_MAX / 10 < result || INT_MAX - result * 10 <
            #    (ord(str[index]) - ord("0")) )
            #if sign == 1 and result >= INT_MAX:     return INT_MAX
            #elif sign == -1 and result <= INT_MIN:  return INT_MIN
    return result

def main(argv):
    USERNAME = 'vanmao_ngo'
    PASSWORD = 'ngovanmao'
    DATABASE = 'testdb'
    TABLE    = 'training_data_openmote' 
    " First timestamp 'Thu May 19 15:45:30 2016'"
    "Put the start (millisecond from 1990) time to indicate the starting time of experiment."
    "gap (minute) is the interval between activites."
    "and dur (minute) is duration of each activity."
    #start = 1453964035808
    start = 1467751230691
    gap = 1 
    dur = 5 
    print 'Set label program, version PA01'
    try:
       opts, args = getopt.getopt(argv,"hu:pd:t:s",["username=","password=","database=","table=","starttime"])
    except getopt.GetoptError:
       print 'labelDataMySqlOpenmote.py [-u <username>] [-p <password>] [-d <database>] [-t <table>] [-s <starttime>]'
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print 'labelDataMySqlOpenMote.py [-u <username>] [-p <password>] [-d <database>] [-t <table>] [-s <starttime>]'
          print 'Start time is measured in millisecond counted from 1990'
          sys.exit()
       elif opt in ("-u", "--username"):
          USERNAME = arg
       elif opt in ("-p", "--password"):
          PASSWORD = arg
       elif opt in ("-d", "--database"):
          DATABASE = arg
       elif opt in ("-t", "--table"):
          TABLE = arg
       elif opt in ("-s", "--starttime"):
          tempStart = arg
          start = atol(tempStart) 
    print 'USERNAME is "', USERNAME
    print 'PASSWORD is "', PASSWORD
    print 'DATABASE is "', DATABASE
    print 'TABLE is "', TABLE
    print 'start time is ', start
    # Open database connection
    db = MySQLdb.connect("localhost", USERNAME, PASSWORD, DATABASE)
    # prepare a cursor object using cursor() method
    cursor = db.cursor()
    # execute SQL query using execute() method.
    cursor.execute("SELECT VERSION()")
    # Fetch a single row using fetchone() method.
    data = cursor.fetchone()
    print "Database version : %s " % data
    
    print("Start time is                      ", time.ctime(start/1000))
    start_A1 = start + 0.1 * 60 *1000
    end_A1 = start_A1 + dur * 60 * 1000 
    print("Start time for sitting activity    ", time.ctime(start_A1/1000)) 
    print("End time for sitting activity      ", time.ctime(end_A1/1000))
    
    start_A2 = end_A1 + 1.5 * 60 * 1000
    end_A2 = start_A2 + dur * 60 * 1000
    print("Start time for standing activity   ", time.ctime(start_A2/1000))
    print("End time for standing activity     ", time.ctime(end_A2/1000))
    
    start_A3 = end_A2 + gap * 60 * 1000
    end_A3 = start_A3 + dur * 60 * 1000
    print("Start time for walking activity    ", time.ctime(start_A3/1000))
    print("End time for walking activity      ", time.ctime(end_A3/1000))
    
    start_A4 = end_A3 + gap * 60 * 1000
    end_A4 = start_A4 + dur * 60 * 1000
    print("Start time for lying back activity ", time.ctime(start_A4/1000))
    print("End time for lying back activity   ", time.ctime(end_A4/1000))
    
    start_A5 = end_A4 + gap * 60 * 1000
    end_A5 = start_A5 + dur * 60 * 1000
    print("Start for standing-sitting activity", time.ctime(start_A5/1000))
    print("End for standing-sitting activity  ", time.ctime(end_A5/1000))
    
    start_A6 = end_A5 + gap * 60 * 1000
    end_A6 = start_A6 + dur * 60 * 1000
    print("Start time for running activity    ", time.ctime(start_A6/1000))
    print("End time for running activity      ", time.ctime(end_A6/1000))
    writeLabel(db, cursor, TABLE, 1, start_A1, end_A1)
    writeLabel(db, cursor, TABLE, 2, start_A2, end_A2)
    writeLabel(db, cursor, TABLE, 3, start_A3, end_A3)
    writeLabel(db, cursor, TABLE, 4, start_A4, end_A4)
    writeLabel(db, cursor, TABLE, 5, start_A5, end_A5)
    writeLabel(db, cursor, TABLE, 6, start_A6, end_A6)
    # disconnect from server
    db.close()

if __name__ == "__main__":
   main(sys.argv[1:])
