import numpy as np
import tensorflow as tf
# for reproducibility set a seed
np.random.seed(1234) 
import pandas as pd
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from keras.models import load_model
import csv
import sys
import os
import math
import joblib
from datetime import datetime

#Creates the models based on the Training Data, and then saves the X/Y/Z models and the Scaler used to scale things for later use
def createTheModels(TrainingDataCSVFile):

    training_data_df = pd.read_csv(TrainingDataCSVFile)

    #Define the Scaler that will be used for all data
    TrainingScaler = MinMaxScaler(feature_range =(0,1))
    TrainingScaler.fit(training_data_df)
    ScaledTraining = TrainingScaler.transform(training_data_df)

    #Create the csv file just so we can debug things later
    scaled_training_df = pd.DataFrame(ScaledTraining, columns=training_data_df.columns.values)
    scaled_training_df.to_csv("D:\OneDrive\Desktop\Scaled_Training_Data.csv", index = False)

    X = scaled_training_df.drop(['NewXPOS', 'NewYPOS', 'NewZPOS'], axis=1).values #<---- Drop them all 
    inputLength = len(X[0])
    XPOSY = scaled_training_df[['NewXPOS']].values
    YPOSY = scaled_training_df[['NewYPOS']].values
    ZPOSY = scaled_training_df[['NewZPOS']].values

    modelActivation = 'relu'
    modelLevels = [50,100,50,1]
    modelOptimizer = "adam"
    modelLoss = "mean_squared_error"

    model = Sequential()
    model.add(Dense(modelLevels[0], input_dim=inputLength, activation=modelActivation)) 
    model.add(Dense(modelLevels[1], activation=modelActivation))
    model.add(Dense(modelLevels[2], activation=modelActivation))
    model.add(Dense(modelLevels[3], activation=modelActivation))
    
    model.compile(loss=modelLoss, optimizer=modelOptimizer)

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    print("Fitting X ...")
    model.fit(X,XPOSY,epochs=50,shuffle=True,verbose=2,callbacks=es)
    model.save("Xtrained_model.h5")
  
    print("Fitting Y ...")
    model.fit(X,YPOSY,epochs=50,shuffle=True,verbose=2,callbacks=es)
    model.save("Ytrained_model.h5")

    print("Fitting Z ...")
    model.fit(X,ZPOSY,epochs=50,shuffle=True,verbose=2,callbacks=es)
    model.save("Ztrained_model.h5")

    #Save the Scaler for later use
    joblib.dump(TrainingScaler, "scaler.save")

#end def createTheModels

#Test how good the models are on a set of testing data
def testTheModels(TestDataCSVFile, simplePrediction):

    # Bring back all three models
    Xmodel = load_model("Xtrained_model.h5")
    Ymodel = load_model("Ytrained_model.h5")
    Zmodel = load_model("Ztrained_model.h5")

    # Load the scaler
    Scaler = joblib.load("scaler.save")

    testing_data_df = pd.read_csv(TestDataCSVFile)

    ScaledTesting = Scaler.transform(testing_data_df)

    scaled_testing_df = pd.DataFrame(ScaledTesting, columns=testing_data_df.columns.values)

    #Create the csv file just so we can debug things later
    scaled_testing_df.to_csv("D:\OneDrive\Desktop\Scaled_Testing_Data.csv", index = False)

    X_test = scaled_testing_df.drop(['NewXPOS', 'NewYPOS', 'NewZPOS'], axis=1).values #<---- Drop them all
    Y_testX = scaled_testing_df[['NewXPOS']].values
    Y_testY = scaled_testing_df[['NewYPOS']].values
    Y_testZ = scaled_testing_df[['NewZPOS']].values

    test_error_rate = Xmodel.evaluate(X_test, Y_testX, verbose=0)
    print("X Mean squared error is: {}".format(test_error_rate))

    test_error_rate = Ymodel.evaluate(X_test, Y_testY, verbose=0)
    print("Y Mean squared error is: {}".format(test_error_rate))

    test_error_rate = Zmodel.evaluate(X_test, Y_testZ, verbose=0)
    print("Z Mean squared error is: {}".format(test_error_rate))

    if simplePrediction:

        #Load in the simple scaled prediction data
        pred = pd.read_csv("D:\OneDrive\Desktop\Scaled_Prediction_Data.csv").values

    else:
        
        pred = scaled_testing_df.drop(['NewXPOS', 'NewYPOS', 'NewZPOS'], axis=1).values #<---- Drop them all 

    predictions = []

    CompleteXPredictions = Xmodel.predict(pred)
    CompleteYPredictions = Ymodel.predict(pred)
    CompleteZPredictions = Zmodel.predict(pred)
    
    for x in range(0,len(pred)):

        Xprediction = CompleteXPredictions[x][0]
        Yprediction = CompleteYPredictions[x][0]
        Zprediction = CompleteZPredictions[x][0]
        
        Xpred = (Xprediction * (Scaler.data_max_[23] - Scaler.data_min_[23])) +  Scaler.data_min_[23]
        Ypred = (Yprediction * (Scaler.data_max_[24] - Scaler.data_min_[24])) +  Scaler.data_min_[24]
        Zpred = (Zprediction * (Scaler.data_max_[25] - Scaler.data_min_[25])) +  Scaler.data_min_[25]

        if simplePrediction:

            print("Prediction of X_POS is {}".format(Xprediction))
            print("Prediction of Y_POS is {}".format(Yprediction))
            print("Prediction of Z_POS is {}".format(Zprediction))
            print("Scaled prediction of X_POS is {}".format(Xpred))
            print("Scaled prediction of Y_POS is {}".format(Ypred))
            print("Scaled prediction of Z_POS is {}".format(Zpred))

        predictions.append([Xpred,Ypred,Zpred]) 

    return(predictions)

#end def testTheModels

#Write a CSV file which compares what the actual new X/Y/Z positions are vs. the predicted ones, just so we can see how good or bad things are
def compareActualVsPrediction(TestDataCSVFile, logFile):

    # Load the scaler
    Scaler = joblib.load("scaler.save")

    comparisonCSVFile = "Comparison.csv"
    
    #Open the csvFile. Note that the newline='' stops the writer from inserting extra lines in the csv file
    try:
        csvFile = open(comparisonCSVFile, "w+", newline='')
    except IOError:
        print("Could not write file:", comparisonCSVFile)
        sys.exit()

    testing_data_df = pd.read_csv(TestDataCSVFile).values
        
    #Create a list with all of the column headers for the CSV file
    csv_columns = ["OldJD","OldJDFR","NewXPOS","NewYPOS","NewZPOS","PredXPOS","PredYPOS","PredZPOS","XErrorKM","YErrorKM","ZErrorKM"]
    csv_writer = csv.writer(csvFile)
    csv_writer.writerow(csv_columns) # write header

    predictions = testTheModels(TestDataCSVFile, False)

    count = 0
    LargestXError = 0
    LargestYError = 0
    LargestZError = 0

    for item in testing_data_df:

        OldJD = item[0]
        OldJDFR = item[1]
        NewXPOS = item[23]
        NewYPOS = item[24]
        NewZPOS = item[25]
        PredXPOS = predictions[count][0]
        PredYPOS = predictions[count][1]
        PredZPOS = predictions[count][2]

        XError = math.sqrt((NewXPOS - PredXPOS)**2)
        YError = math.sqrt((NewYPOS - PredYPOS)**2)
        ZError = math.sqrt((NewZPOS - PredZPOS)**2)

        if XError > LargestXError:
            LargestXError = XError
        if YError > LargestYError:
            LargestYError = YError
        if ZError > LargestZError:
            LargestZError = ZError         
        
        row = [OldJD,OldJDFR,NewXPOS,NewYPOS,NewZPOS,PredXPOS,PredYPOS,PredZPOS,XError,YError,ZError]
        csv_writer.writerow(row)
        count = count + 1
      
    csvFile.close()
    print("Written the CSV File:", comparisonCSVFile)
    largestError = max(LargestXError, LargestYError, LargestZError)
    print("Largest Absolute Error:", round(largestError,2), "XError:YError:ZError", round(LargestXError,2), round(LargestYError,2), round(LargestZError,2), file=logFile, flush=True)

#end def testTheModel

#Main Processing loop which runs all the experiments
#Define what to run, and how long to run it for
#Normally run it twice including training just to get round any restarting inconsistencies
#Or simply set the range to zero to stop running them

Run24946onTest = 0
Run24907onTest = 0
RunMergedSatellite = 0
Run34367Debris = 0
Run35479Debris = 0
Run37558Debris = 0
RunMergedDebris = 0    

logFile = open("D:\OneDrive\Desktop\Results.txt", 'w')

print("Running the experiments ...", file=logFile, flush=True)

dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_Labelled24946_20200101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled24946_20190101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled24946_20180101_20201230.csv"]

print("Running the first part of the experiments ...", file=logFile, flush=True)


for x in range (Run24946onTest):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for ID 24946 (1997-051C) on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on ID 24946 (1997-051C) data on ID 24946 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210113.csv"]

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        #See if the 24946 data can be used for the other Satellite 1997-043E – Id: 24907
        print("Testing the model trained on ID 24946 (1997-051C) data on ID 24907 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210113.csv"]

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

print("Running the second part of the experiments ...", file=logFile, flush=True)           
#Repeat the test for the 24907 data
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_Labelled24907_20200101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled24907_20190101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled24907_20180101_20201230.csv"]
            
for x in range (Run24907onTest):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for ID 24907 (1997-043E) on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on ID 24907 (1997-043E) data on ID 24946 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210113.csv"]
        

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        #See if the 24946 data can be used for the other Satellite 1997-043E – Id: 24907
        print("Testing the model trained on ID 24907 (1997-043E) data on ID 24907 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210113.csv"]

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

print("Running the third part of the experiments ...", file=logFile, flush=True)
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_LabelledM24946_24907_20200101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledM24946_24907_20190101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledM24946_24907_20180101_20201230.csv"]

for x in range (RunMergedSatellite):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for Merged Dataset of 24946 and 24907 on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on Merged Dataset of 24946 and 24907 on ID 24946 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24946_20210101_20210113.csv"]

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        #See if the 24946 data can be used for the other Satellite 1997-043E – Id: 24907
        print("Testing the model trained on Merged Dataset of 24946 and 24907 on ID 24907 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210101.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled24907_20210101_20210113.csv"]

        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

print("Running the fourth part of the experiments ...", file=logFile, flush=True)           
#Now repeat for one of the debris models
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_Labelled34367_20201201_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20201001_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20200701_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20200401_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20200101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20190101_20201230.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled34367_20180101_20201230.csv"]

for x in range (Run34367Debris):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for ID 34367 (1997-051FK) on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on ID 34367 (1997-051FK) data on ID 34367 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210113.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210120.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210127.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)


print("Running the fifth part of the experiments ...", file=logFile, flush=True)           
#Now repeat for another of the debris models
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_Labelled35479_20201201_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20201001_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20200702_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20200401_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20200101_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20190101_20201225.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled35479_20180101_20201225.csv"]

for x in range (Run35479Debris):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for ID 35479 (1997-051PZ) on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on ID 35479 (1997-051PZ) data on ID 35479 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210110.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210117.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210124.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)


print("Running the sixth part of the experiments ...", file=logFile, flush=True)           
#Now repeat for another of the debris models
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_Labelled37558_20201201_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20201002_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20200702_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20200401_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20200101_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20190101_20201228.csv",
                 "D:/OneDrive/Desktop/Training/Ext_Labelled37558_20180101_20201228.csv"]

for x in range (Run37558Debris):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for ID 37558 (1997-051XT) on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on ID 37558 (1997-051XT) data on ID 37558 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210105.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210112.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210119.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210126.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

print("Running the seventh and final part of the experiments with merged debris data ...", file=logFile, flush=True)
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_LabelledMergedDec2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMergedOctDec2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMergedJulyDec2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMergedAprDec2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMerged2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMerged2019_2020.csv",
                 "D:/OneDrive/Desktop/Training/Ext_LabelledMerged2018_2020.csv"]

for x in range (RunMergedDebris):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for Merged Debris Dataset on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on the merged Debris dataset on ID 34367 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210113.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210120.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210127.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        print("Testing the model trained on the merged Debris dataset on ID 35479 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210110.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210117.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210124.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        print("Testing the model trained on the merged Debris dataset on ID 37558 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210105.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210112.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210119.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210126.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)


#Top up tests
#Now repeat for another of the debris models
print("Running the seventh and final part of the experiments with merged debris data ...", file=logFile, flush=True)
dataToTrainOn = ["D:/OneDrive/Desktop/Training/Ext_LabelledMergedOctDec2020.csv"]

for x in range (5):

    for trainingData in dataToTrainOn:

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Training for Merged Debris Dataset on: ", trainingData, "at:", current_time, file=logFile, flush=True)

        createTheModels(trainingData)

        print("Testing the model trained on the merged Debris dataset on ID 34367 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210103.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210106.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210113.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210120.csv",
                        "D:/OneDrive/Desktop/Test/Labelled34367_20210101_20210127.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        print("Testing the model trained on the merged Debris dataset on ID 35479 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210110.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210117.csv",
                        "D:/OneDrive/Desktop/Test/Labelled35479_20210102_20210124.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)

        print("Testing the model trained on the merged Debris dataset on ID 37558 test data.", file=logFile, flush=True)
        dataToTestOn = ["D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210105.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210112.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210119.csv",
                        "D:/OneDrive/Desktop/Test/Labelled37558_20210103_20210126.csv"]
        
        for testData in dataToTestOn:

            print("Accuracy for : ", testData, file=logFile, flush=True)
            testTheModels(testData,True)
            compareActualVsPrediction(testData,logFile)




logFile.close()
