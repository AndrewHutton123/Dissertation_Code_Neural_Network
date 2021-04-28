import numpy as np
import json
import csv
import sys
import os
from sgp4.api import Satrec
from sgp4.api import jday
from sgp4 import omm
from sgp4 import exporter
import pandas as pd

#A set of routines to take a JSON file from space-track.org and manipulate it to produce
#a set of labelled data which can be used to model how to track debris and satellites
#through time.
#
#The features produced are the original position, velocity, eccentricity etc of the
#object plus the time in seconds to the next measurement, and the label is then the
#resulting position and velocity.
#
#The aim is to then generate a model which can be used to establish where debris and
#satellite are after a specific time based on some initial data about where they were.
#
#The model should be trained differently for satellites and debris to see how they differ
#since satellites are relatively large and controlled, and so should be predictable, while
#space debris isn't controlled and something as light as space debris can be severely
#perturbed by a lot of things, in a pretty short time, depending on the height of the
#orbit, and thus drag coefficient
#
#Hence we can get two models, one for satellites and one for debris and then test against
#each other


#Opens the satellite data JSON file and returns the list of dicts with all the data
def openJSONFile(jsonFileName):

    print("Reading the JSON File:", jsonFileName)

    #Open a JSON file and returns a list of dict
    try:
        read_file = open(jsonFileName, "r")
    except IOError:
        print("Could not open/read file:", jsonFileName)
        sys.exit()
    
    data = json.load(read_file)
    read_file.close()

    return(data)

#end openJSONFile

#Write a list of dicts into a CSVFile just to easily see what is in there
def writeRaWCSVFile(jsonData):

    csvFileName = "Raw" + jsonData[0]["NORAD_CAT_ID"] + ".csv"

    #Extract individual keys for the dicts, and then create a list with all of the keys
    keys = jsonData[0].keys()
    csv_columns = []
    for key in keys:
        csv_columns.append(key)

    #Open the csvFile. Note that the newline='' stops the writer from inserting extra lines in the csv file
    try:
        csvFile = open(csvFileName, "w+", newline='')
    except IOError:
        print("Could not write file:", csvFileName)
        sys.exit()
    
    writer = csv.DictWriter(csvFile, fieldnames=csv_columns)

    #Write the header information, and then for each dict in the list, write out the data which is formated by the csv_columns information
    writer.writeheader()

    for item in jsonData:

        writer.writerow(item)
            
    csvFile.close()
    print("Written the CSV File:", csvFileName)

#end writeRaWCSVFile

#Since we have the data, then we can do some early wrangling and processing of it
def wrangleData(jsonData):

    defaultRCS_SIZE = "SMALL"

    #sometimes the RCS_SIZE data is empty, so fill it with the existing values
    for item in jsonData:

        RCS_SIZE = item["RCS_SIZE"]

        if (RCS_SIZE == "LARGE") or (RCS_SIZE == "MEDIUM") or (RCS_SIZE == "SMALL"):
            defaultRCS_SIZE = item["RCS_SIZE"]
        else:
            item["RCS_SIZE"] = defaultRCS_SIZE
    
    return(jsonData)

#end def wrangleData


#Setup a format for the data CSV file just so it's all in one place
#type = "f" means the format, type = "d" means the data
def formatCSVFile(oldData,satellite,item,type):

    if type == "f":

        csv_columns = ["OldJD","OldJDFR","OldXPOS","OldYPOS","OldZPOS","OldXVEL","OldYVEL","OldZVEL"]
        csv_columns = csv_columns + ["BSTAR","ECCENTRICITY","INCLINATION","MEAN_ANOMALY","MEAN_MOTION","MEAN_MOTION_DDOT","MEAN_MOTION_DOT"]
        csv_columns = csv_columns + ["RCS_SIZE","RA_OF_ASC_NODE","ARG_OF_PERICENTER","SEMIMAJOR_AXIS","PERIOD","APOAPSIS","PERIAPSIS"]
        csv_columns = csv_columns + ["DeltaTime"]
        csv_columns = csv_columns + ["NewXPOS","NewYPOS","NewZPOS"]
        return(csv_columns)

    elif type == "d":

        #Change the text describing the size into a number so that everything in the CSV file is numeric
        if item["RCS_SIZE"] == "LARGE":
            size = 10
        elif item["RCS_SIZE"] == "MEDIUM":
            size = 1
        elif item["RCS_SIZE"] == "SMALL":
            size = 0.1
        else:
            print("Cannot find an RCS_SIZE")
            size = 0

        oldData = oldData + [satellite.bstar,satellite.ecco,satellite.inclo,item["MEAN_ANOMALY"],item["MEAN_MOTION"],item["MEAN_MOTION_DDOT"],item["MEAN_MOTION_DOT"]]
        oldData = oldData + [size,item["RA_OF_ASC_NODE"],item["ARG_OF_PERICENTER"],item["SEMIMAJOR_AXIS"],item["PERIOD"],item["APOAPSIS"],item["PERIAPSIS"]]
        return(oldData)

    else:
        print("Error in formatting of CSVFile")
        sys.exit()

#end def formatCSVFile

#Use the SGP4 processing library and extract a load of data to a CSV file. The CSV file should be a set of labelled data
#containing the previous time in Julian format, the current time, the previous position and velocity and then a time
#to the next period, and that new position and velocity
def writeLabelledCSVFile(jsonData, Extend):

    firstTimeThrough = True
    filterLevel = 500
    firstEPOCH = jsonData[0]["EPOCH"]
    year, month, time = firstEPOCH.split("-")
    day, time = time.split("T")
    firstPeriod = year + month + day
    lastEPOCH = jsonData[len(jsonData)-1]["EPOCH"]
    year, month, time = lastEPOCH.split("-")
    day, time = time.split("T")
    lastPeriod = year + month + day
    
    newCSVFileName = "Labelled" + jsonData[0]["NORAD_CAT_ID"] + "_" + firstPeriod + "_" + lastPeriod + ".csv"
    
    #Open the csvFile. Note that the newline='' stops the writer from inserting extra lines in the csv file
    try:
        csvFile = open(newCSVFileName, "w+", newline='')
    except IOError:
        print("Could not write file:", newCSVFileName)
        sys.exit()
    
    #Create a list with all of the column headers for the CSV file
    csv_columns = formatCSVFile(0,0,0,"f")
    csv_writer = csv.writer(csvFile)
    csv_writer.writerow(csv_columns) # write header

    for item in jsonData:

        TLE1 = item["TLE_LINE1"]
        TLE2 = item["TLE_LINE2"]
        epoch = item["EPOCH"]
        year, month, time = epoch.split("-")
        day, time = time.split("T")
        hour, minute, sec = time.split(":")
        jdBase, frBase = jday(float(year),float(month),float(day),float(hour),float(minute),float(sec))

        #The call to Satrec.twoline2rv uses the two TLE data elements and extracts a ton of information
        satellite = Satrec.twoline2rv(TLE1, TLE2)

        #Extract the Julian Date and the fractional value, which makes all the calculations above somewhat pointless!
        jd = satellite.jdsatepoch
        jdFr = satellite.jdsatepochF
        
        #error will be a non-zero error code if the satellite position could not be computed for the given date
        #position - the satellite position in kilometers from the center of the earth in True Equator Mean Equinox coordinate frame
        #velocity -  is the rate at which the position is changing, expressed in kilometers per second.
        error, position, velocity = satellite.sgp4(jd, jdFr)

        if error != 0:
            print("Error extracting position and velocity")
            sys.exit()
            
        xpos, ypos, zpos = position
        xvel, yvel, zvel = velocity

        if firstTimeThrough:
            
            firstTimeThrough = False

        else:
            
            DeltaTime = 3600*(jd+jdFr-oldData[0]-oldData[1])
            jumpInX = max(abs(oldData[2] / xpos),abs(xpos / oldData[2]))
            jumpInY = max(abs(oldData[3] / ypos),abs(ypos / oldData[3]))
            jumpInZ = max(abs(oldData[4] / zpos),abs(zpos / oldData[4]))

            # Data cleanup by filtering out any tiny changes in time due to rounding errors,
            # and any massive jumps in position data which will cause outliers
            if (DeltaTime > 0.1) and (jumpInX<=filterLevel) and (jumpInY<=filterLevel) and (jumpInZ<=filterLevel):

                newData = [DeltaTime,xpos, ypos, zpos]
                row = oldData + newData
                csv_writer.writerow(row)

        oldData = [jd, jdFr, xpos, ypos, zpos, xvel, yvel, zvel]
        oldData = formatCSVFile(oldData,satellite,item,"d")
            
    csvFile.close()
    print("Written the labelled CSV File:", newCSVFileName)

    if Extend:
        extendLabelledCSVFile(newCSVFileName)

#end def writeLabelledCSVFile:

#labelledCSVFile = "D:\OneDrive\Desktop\Training\Labelled24946.csv"

#Take an existing LabelledCSVFile, and extend it by replicating data and change the Delta time
def extendLabelledCSVFile(labelledCSVFile):

    csvFileName = "Ext_" + labelledCSVFile

    df = pd.read_csv(labelledCSVFile)

    columnNames = list(df)

    numberOfValues = len(df)
    
    print(numberOfValues)

    for x in range(0,numberOfValues):

        if x % 50 == 0:
            print(x)

        old_item = df.iloc[x].copy()

        deltaTime = old_item["DeltaTime"]

        newList = []

        for y in range(x+1,numberOfValues-1):
            
            new_item = df.iloc[y]

            deltaTime = deltaTime + new_item["DeltaTime"]

            old_item["DeltaTime"] = deltaTime
            old_item["NewXPOS"] = new_item["NewXPOS"]
            old_item["NewYPOS"] = new_item["NewYPOS"]
            old_item["NewZPOS"] = new_item["NewZPOS"]

            zipped = zip(columnNames, old_item.values)
            a_dictionary = dict(zipped)
            newList.append(a_dictionary)

        df = df.append(newList, True)

    #print(len(df),"an expansion of",len(df)/numberOfValues,"times")

    df.to_csv(csvFileName, index = False)

    print("Written the Extended CSV File:", csvFileName)

#end def extendLabelledCSVFile

#Main. Opens a directory, scans for all *.json files, prints each one to a named Raw
#CSVFile, wrangles the data, and prints a Labelled CSV File. Names of the CSV file are
#from the NORAD_CAT_ID in the JSON file
#
#We dump a set of training data to one directory and a set of test data to another
#and then convert them both
        
#Directory Variables
directoriesToProcess = ["D:\OneDrive\Desktop\Training","D:\OneDrive\Desktop\Test"]

for directory in directoriesToProcess:
    print("Processing directory:",directory)
    os.chdir(directory)
    with os.scandir() as entries:
        for entry in entries:
            if entry.name.split(".")[1] == "json":
                jsonData = openJSONFile(entry.name)
                writeRaWCSVFile(jsonData)
                jsonData = wrangleData(jsonData)
                writeLabelledCSVFile(jsonData, True)


