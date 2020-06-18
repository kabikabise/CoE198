# Code originally from Jin-Man Park and Jong-hwan Kim
# "Online recurrent extreme learning machine and its application to
# time-series prediction." Neural Networks (IJCNN), 2017 International Joint Conference on. IEEE, 2017.
# ------------------------------------------------------
# Modified for use in PV prediction

import csv
from optparse import OptionParser
from matplotlib import pyplot as plt
import numpy as np
from scipy import random
import pandas as pd
from errorMetrics import *
from algorithms.OR_ELM import ORELM
from algorithms.FOS_ELM import FOSELM
from algorithms.NAOS_ELM import NAOSELM

def _getArgs():
  parser = OptionParser(usage="%prog [options]"
                              "\n\nOnline Recurrent Extreme Learning Machine (OR-ELM)"
                              "and its application to time-series prediction,"
                              "with Solcast Radiation dataset.")
  parser.add_option("-d",
                    "--dataSet",
                    type=str,
                    default='solcast_data_old',
                    dest="dataSet",
                    help="DataSet Name, choose from sine or solcast_data_old")
  parser.add_option("-l",
                    "--numLags",
                    type=int,
                    default='100',
                    help="the length of time window, this is used as the input dimension of the network")
  parser.add_option("-p",
                    "--predStep",
                    type=int,
                    default='1',
                    help="the prediction step of the output")
  parser.add_option("-a",
                    "--algorithm",
                    type=str,
                    default='ORELM',
                    help="Algorithm name, choose from FOSELM, NFOSELM, NAOSELM, ORELM")
  (options, remainder) = parser.parse_args()
  return options, remainder


def initializeNet(nDimInput, nDimOutput, numNeurons=100, algorithm='ORELM',
                  LN=True, InWeightFF=0.999, OutWeightFF=0.999, HiddenWeightFF=0.999,
                  ORTH=True, AE=True, PRINTING=True):

  assert algorithm =='FOSELM' or algorithm == 'NFOSELM' or algorithm == 'NAOSELM' or algorithm == 'ORELM'
  if algorithm=='FOSELM':
      '''
      Fully Online Sequential ELM (FOSELM). It's just like the basic OSELM, except its initialization.
      Wong, Pak Kin, et al. "Adaptive control using fully online sequential-extreme learning machine
      and a case study on engine air-fuel ratio regulation." Mathematical Problems in Engineering 2014 (2014).
      '''
      net = FOSELM(nDimInput, nDimOutput,
                  numHiddenNeurons=numNeurons,
                  activationFunction='sig',
                  forgettingFactor=OutWeightFF,
                  LN=False,
                  ORTH=ORTH)
  if algorithm=='NFOSELM':
      '''
      FOSELM + layer Normalization. + forgetting factor
      '''
      net = FOSELM(nDimInput, nDimOutput,
                  numHiddenNeurons=numNeurons,
                  activationFunction='sig',
                  forgettingFactor=OutWeightFF,
                  LN=True,
                  ORTH=ORTH)
  elif algorithm=='NAOSELM':
      '''
      FOSELM + layer Normalization + forgetting factor + input layer weight Auto-encoding.
      '''
      net = NAOSELM(nDimInput, nDimOutput,
                    numHiddenNeurons=numNeurons,
                    activationFunction='sig',
                    LN=LN,
                    inputWeightForgettingFactor=InWeightFF,
                    outputWeightForgettingFactor=OutWeightFF,
                    ORTH=ORTH,
                    AE=AE)
  elif algorithm=='ORELM':
      '''
      Online Recurrent Extreme Learning Machine (OR-ELM).
      FOSELM + layer normalization + forgetting factor + input layer weight auto-encoding + hidden layer weight auto-encoding.
      '''
      net = ORELM(nDimInput, nDimOutput,
                  numHiddenNeurons=numNeurons,
                  activationFunction='sig',
                  LN=LN,
                  inputWeightForgettingFactor=InWeightFF,
                  outputWeightForgettingFactor=OutWeightFF,
                  hiddenWeightForgettingFactor=HiddenWeightFF,
                  ORTH=ORTH,
                  AE=AE)


  if PRINTING:
    print('----------Network Configuration-------------------')
    print('Algotirhm = '+algorithm)
    print('#input neuron = '+str(nDimInput))
    print('#output neuron = '+str(nDimOutput))
    print('#hidden neuron = '+str(numNeurons))
    print('Layer normalization = ' + str(net.LN))
    print('Orthogonalization = '+str(ORTH))
    print('Auto-encoding = '+str(AE))
    print('input weight forgetting factor = '+str(InWeightFF))
    print('output weight forgetting factor = ' + str(OutWeightFF))
    print('hidden weight forgetting factor = ' + str(HiddenWeightFF))
    print('---------------------------------------------------')

  return net

def readDataSet(dataSet):
  filePath = 'data/'+dataSet+'.csv'
  if dataSet=='solcast_data_old':
    df = pd.read_csv(filePath, header=0, skiprows=[1,2],
                     names=['time', 'dhi', 'temp', 'dayofweek'])
    dhi = df['dhi']
    dayofweek = df['dayofweek']
    temp = df['temp']
    seq = pd.DataFrame(np.array(pd.concat([dhi, temp, dayofweek], axis=1)),
                        columns=['dhi', 'temp', 'dayofweek'])
  elif dataSet=='sine':
    df = pd.read_csv(filePath, header=0, skiprows=[1, 2], names=['time', 'data'])
    sequence = df['data']
    seq = pd.DataFrame(np.array(sequence), columns=['data'])
  else:
    raise(' unrecognized dataset type ')

  return seq



def getTimeEmbeddedMatrix1(sequence, numLags=100, predictionStep=1):
  print ("generate time embedded matrix ")
  inDim = numLags
  X = np.zeros(shape=(len(sequence), inDim))
  T = np.zeros(shape=(len(sequence), 1))
  for i in range(numLags-1, len(sequence)-predictionStep):
    X[i, :] = np.array(sequence['dhi'][(i-numLags+1):(i+1)])
    T[i, :] = sequence['dhi'][i+predictionStep]
  return (X, T)

def getTimeEmbeddedMatrix2(sequence, numLags=100, predictionStep=1):
  print ("generate time embedded matrix ")
  inDim = numLags
  X = np.zeros(shape=(len(sequence), inDim))
  T = np.zeros(shape=(len(sequence), 1))
  for i in range(numLags-1, len(sequence)-predictionStep):
    X[i, :] = np.array(sequence['temp'][(i-numLags+1):(i+1)])
    T[i, :] = sequence['temp'][i+predictionStep]
  return (X, T)

def saveResultToFile(dataSet, predictedInput, algorithmName,predictionStep,dataname):
  inputFileName = 'data/' + dataSet + '.csv'
  inputFile = open(inputFileName, "r")
  csvReader = csv.reader(inputFile)
  # skip header rows
  next(csvReader)
  next(csvReader)
  next(csvReader)
  outputFileName = './prediction/' + dataSet + '_' + algorithmName +'_'+dataname+ '_pred.csv'
  outputFile = open(outputFileName, "w")
  csvWriter = csv.writer(outputFile)
  csvWriter.writerow(
    ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
  csvWriter.writerow(['datetime', 'float', 'float'])
  csvWriter.writerow(['', '', ''])

  lastseq = range(len(sequence))[-1]
  for i in range(len(sequence)):
    if i < lastseq:
        row = next(csvReader)
    csvWriter.writerow([row[0], row[1], predictedInput[i]])

  inputFile.close()
  outputFile.close()
  print ('Prediction result is saved to ' + outputFileName)


if __name__ == "__main__":

  (_options, _args) = _getArgs()
  algorithm = _options.algorithm
  dataSet = _options.dataSet
  numLags = _options.numLags
  predictionStep = _options.predStep
  print ("run ", algorithm, " on ", dataSet)
  # prepare dataset
  sequence = readDataSet(dataSet)
  # standardize data by subtracting mean and dividing by std
  meanSeq = np.mean(sequence['dhi'])
  stdSeq = np.std(sequence['dhi'])
  sequence['dhi'] = (sequence['dhi'] - meanSeq)/stdSeq

  (X, T) = getTimeEmbeddedMatrix1(sequence, numLags, predictionStep)

  random.seed(6)

  net = initializeNet(nDimInput=X.shape[1],
                      nDimOutput=1,
                      numNeurons=23,
                      algorithm=algorithm,
                      LN=True,
                      InWeightFF=1,
                      OutWeightFF=0.915,
                      HiddenWeightFF=1,
                      AE=True,
                      ORTH=False)
  net.initializePhase(lamb = 0.0001)

  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))

  for i in range(numLags, len(sequence)-predictionStep-1):
    net.train(X[[i], :], T[[i], :])
    Y = net.predict(X[[i+1], :])

    predictedInput[i+1] = Y[-1]
    targetInput[i+1] = sequence['dhi'][i+1+predictionStep]
    trueData[i+1] = sequence['dhi'][i+1]
    print ("{:5}th timeStep -  target: {:8.4f}   |    prediction: {:8.4f} ".format(i, targetInput[i+1], predictedInput[i+1]))
    if Y[-1] > 100000:
      print ("Output has diverged, terminate the process")
      predictedInput[(i + 1):] = 100000
      break

  '''
  Calculate total Normalized Root Mean Square Error (NRMSE)
  '''
  # Reconstruct original value
  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  trueData = (trueData * stdSeq) + meanSeq
  # Calculate NRMSE from stpTrain to the end
  skipTrain = numLags
  from plot import computeSquareDeviation

  squareDeviation = computeSquareDeviation(predictedInput, targetInput)
  squareDeviation[:skipTrain] = None
  nrmse1 = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
  # Save prediction result as csv file
  saveResultToFile(dataSet, predictedInput, 'FF' + str(net.forgettingFactor) + algorithm + str(net.numHiddenNeurons),
                     predictionStep, 'dhi')

  '''
  Plot predictions and target values
  '''
  plt.figure(figsize=(15,6))
  targetPlot,=plt.plot(targetInput,label='target',color='red',marker='.',linestyle='-')
  predictedPlot,=plt.plot(predictedInput,label='predicted',color='blue',marker='.',linestyle=':')
  plt.xlim([8000,8200])
  plt.ylim([0, 1000])
  plt.ylabel('dhi',fontsize=15)
  plt.xlabel('time',fontsize=15)
  plt.ion()
  plt.grid()
  plt.legend(handles=[targetPlot, predictedPlot])
  #plt.title('Time-series Prediction of '+algorithm+' on '+'dhi',fontsize=20,fontweight=40)
  plot_path = './fig/predictionPlot_dhi.png'
  plt.savefig(plot_path, bbox_inches='tight')
  plt.draw()
  plt.show(block=False)
  plt.pause(1)
  print ('Prediction plot is saved to'+plot_path)


  #temp

  # prepare dataset
  sequence = readDataSet(dataSet)
  # standardize data by subtracting mean and dividing by std
  meanSeq = np.mean(sequence['temp'])
  stdSeq = np.std(sequence['temp'])
  sequence['temp'] = (sequence['temp'] - meanSeq)/stdSeq

  (X, T) = getTimeEmbeddedMatrix2(sequence, numLags, predictionStep)

  random.seed(6)

  net = initializeNet(nDimInput=X.shape[1],
                      nDimOutput=1,
                      numNeurons=23,
                      algorithm=algorithm,
                      LN=True,
                      InWeightFF=1,
                      OutWeightFF=0.915,
                      HiddenWeightFF=1,
                      AE=True,
                      ORTH=False)
  net.initializePhase(lamb = 0.0001)

  predictedInput = np.zeros((len(sequence),))
  targetInput = np.zeros((len(sequence),))
  trueData = np.zeros((len(sequence),))

  for i in range(numLags, len(sequence)-predictionStep-1):
    net.train(X[[i], :], T[[i], :])
    Y = net.predict(X[[i+1], :])

    predictedInput[i+1] = Y[-1]
    targetInput[i+1] = sequence['temp'][i+1+predictionStep]
    trueData[i+1] = sequence['temp'][i+1]
    print ("{:5}th timeStep -  target: {:8.4f}   |    prediction: {:8.4f} ".format(i, targetInput[i+1], predictedInput[i+1]))
    if Y[-1] > 100000:
      print ("Output has diverged, terminate the process")
      predictedInput[(i + 1):] = 100000
      break

  '''
  Calculate total Normalized Root Mean Square Error (NRMSE)
  '''
  # Reconstruct original value
  predictedInput = (predictedInput * stdSeq) + meanSeq
  targetInput = (targetInput * stdSeq) + meanSeq
  trueData = (trueData * stdSeq) + meanSeq
  # Calculate NRMSE from stpTrain to the end
  skipTrain = numLags
  from plot import computeSquareDeviation

  squareDeviation = computeSquareDeviation(predictedInput, targetInput)
  squareDeviation[:skipTrain] = None
  nrmse2 = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
  print ("NRMSE(dhi) {}".format(nrmse1))
  print ("NRMSE(temp) {}".format(nrmse2))
  # Save prediction result as csv file
  saveResultToFile(dataSet, predictedInput, 'FF' + str(net.forgettingFactor) + algorithm + str(net.numHiddenNeurons),
                     predictionStep,'temp')

  '''
  Plot predictions and target values
  '''
  plt.figure(figsize=(15,6))
  targetPlot,=plt.plot(targetInput,label='target',color='red',marker='.',linestyle='-')
  predictedPlot,=plt.plot(predictedInput,label='predicted',color='blue',marker='.',linestyle=':')
  plt.xlim([8000,8200])
  plt.ylim([0, 40])
  plt.ylabel('temp',fontsize=15)
  plt.xlabel('time',fontsize=15)
  plt.ion()
  plt.grid()
  plt.legend(handles=[targetPlot, predictedPlot])
  #plt.title('Time-series Prediction of '+algorithm+' on '+'temp',fontsize=20,fontweight=40)
  plot_path = './fig/predictionPlot_temp.png'
  plt.savefig(plot_path, bbox_inches='tight')
  plt.draw()
  plt.show(block=False)
  plt.pause(1)
  print ('Prediction plot is saved to'+plot_path)
