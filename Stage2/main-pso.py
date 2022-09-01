from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import time

batchSize = 32
imgSize = 224

test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/images/test',
                                          target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary', shuffle = False)

labels = test_set.classes

modelNames = os.listdir('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/')
modelPredictions = list()
modelAccuracies = list()

for name in modelNames:
    model = load_model(('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/' + name))
    modelPrediction = model.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)
    modelPredictions.append(modelPrediction)
    
    score = 0
    for i in range(test_set.n):
        if round(float(modelPrediction[i])) == labels[i]:
            score += 1
    score /= test_set.n
    modelAccuracies.append(score)
  
bestAcc = max(modelAccuracies)
averageAcc = sum(modelAccuracies) / len(modelAccuracies)

test_set2=test_datagen.flow_from_directory('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/images/test2',
                                          target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary', shuffle = False)

labels2 = test_set2.classes

modelPredictions2 = list()
modelAccuracies2 = list()

for name in modelNames:
    model = load_model(('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/' + name))
    modelPrediction2 = model.predict_generator(test_set2, steps=test_set2.samples/test_set2.batch_size,verbose=1)
    modelPredictions2.append(modelPrediction2)
    
    score = 0
    for i in range(test_set2.n):
        if round(float(modelPrediction2[i])) == labels2[i]:
            score += 1
    score /= test_set2.n
    modelAccuracies2.append(score)
  
bestAcc2 = max(modelAccuracies2)
averageAcc2 = sum(modelAccuracies2) / len(modelAccuracies2)

def generatePopulation(populationNumber):
    population = list()
    for i in range(populationNumber):  
        weights = list()
        for j in range(len(modelNames)):
            weights.append(random.random())

        sumOfWeights = sum(weights)
        
        for i in range(len(weights)):
            weights[i] /= sumOfWeights
        population.append(list(weights))
    return population

def calculatePopulationScore(population):
    results = list()
    for person in population:
        newPredictions = person[0] * modelPredictions[0]
        for i in range(1, len(modelNames)):
            newPredictions += person[i] * modelPredictions[i]
        newPredictions = [round(float(item)) for item in newPredictions]
        score = 0
        for i in range(test_set.n):
            if newPredictions[i] == labels[i]:
                score += 1
        score /= test_set.n
        results.append(score)
    return results

def run():
    velocityWeight = 0.8
    c1 = 2
    c2 = 2
    lowerLimit = 0.001
    upperLimit = 0.999
    velocityLimit = (upperLimit - lowerLimit) / 100
    
    population = generatePopulation(100)
    results = calculatePopulationScore(population)
    
    personalBestValues = population
    personalBestResults = results
    
    globalBestResult = max(results)
    idx = results.index(globalBestResult)
    globalBestValue = population[idx]
    
    plotValues = list()
    plotValues.append(globalBestResult)
    
    velocity = np.zeros([len(population),len(modelNames)])
    
    for i in range(100):
        for j in range(len(population)):
            for k in range(len(modelNames)):
                x = (velocityWeight * velocity[j][k]) + (c1 * random.random() * (personalBestValues[j][k]) - population[j][k]) + (c2 * random.random() * (globalBestValue[k] - population[j][k]))
                if x > velocityLimit:
                    x = velocityLimit
                elif x < -velocityLimit:
                    x = -velocityLimit
                velocity[j][k] = x
        
        population += velocity
        
        for j in range(len(population)):
            for k in range(len(modelNames)):
                if population[j][k] > upperLimit:
                    population[j][k] = upperLimit
                elif population[j][k] < lowerLimit:
                    population[j][k] = lowerLimit
        
        for j in range(len(population)):
            sumOfWeights = sum(population[j])
            for k in range(len(modelNames)):
                population[j][k] /= sumOfWeights
                  
        results = calculatePopulationScore(population)
        
        for j in range(len(population)):
            if results[j] > personalBestResults[j]:
                personalBestResults[j] = results[j]
                personalBestValues[j] = population[j]
    
        if max(results) > globalBestResult:
            globalBestResult = max(results)
            idx = results.index(globalBestResult)
            globalBestValue = population[idx]
        plotValues.append(globalBestResult)    
    
    return globalBestResult, globalBestValue, plotValues
        
def main():
    start = time.time()
    counter = 1
    globalPlotValues = list()
    globalBestResult = 0
    globalBestValue = list()  
    plotValues = list()
    while True:
        globalBestResult, globalBestValue, plotValues = run()
        globalPlotValues.append(globalBestResult)
        plt.plot(plotValues)
        plt.xlabel("Tur: " + str(counter))
        plt.show() 
        print("Tur: " + str(counter) + " En iyi değer: " + "{:.3f}".format(globalBestResult))
        counter += 1
        if globalBestResult >= 0.638:
            break
    stop = time.time()   
    print("En iyi sonuca ulaşma süresi(s): ", int(stop - start))    
    print(globalBestValue)    
            
    plt.plot(globalPlotValues)
    plt.xlabel("Dış Döngü")
    plt.show() 
    
    print("En iyi Modelin Accuracy Değeri: " + "{:.3f}".format(bestAcc) + " Tüm Modellerin Ortalama Accuracy Değeri: " + "{:.3f}".format(averageAcc))
    print("En iyi modele göre kazanılan accuracy artışı: " + "{:.3f}".format(globalBestResult - bestAcc) + " Yüzdelik olarak: % " + "{:.2f}".format(((100 * globalBestResult) / bestAcc) - 100))
    print("Modellerin ortalamasına göre kazanılan accuracy artışı: " + "{:.3f}".format(globalBestResult - averageAcc) + " Yüzdelik olarak: % " + "{:.2f}".format(((100 * globalBestResult) / averageAcc) - 100))
    
    newPredictions = globalBestValue[0] * modelPredictions2[0]
    
    for i in range(1, len(modelNames)):
        newPredictions += globalBestValue[i] * modelPredictions2[i]
        
    newPredictions = [round(float(item)) for item in newPredictions]
    score = 0
    for i in range(test_set2.n):
        if newPredictions[i] == labels2[i]:
            score += 1
    score /= test_set2.n
    
    print("Skor: " + str(score))
    print("En iyi Modelin Accuracy Değeri: " + "{:.3f}".format(bestAcc2) + " Tüm Modellerin Ortalama Accuracy Değeri: " + "{:.3f}".format(averageAcc2))
    print("En iyi modele göre kazanılan accuracy artışı: " + "{:.3f}".format(score - bestAcc2) + " Yüzdelik olarak: % " + "{:.2f}".format(((100 * score) / bestAcc2) - 100))
    print("Modellerin ortalamasına göre kazanılan accuracy artışı: " + "{:.3f}".format(score - averageAcc2) + " Yüzdelik olarak: % " + "{:.2f}".format(((100 * score) / averageAcc2) - 100))
    return ((100 * score) / bestAcc2) - 100
    
while True:
    x = main()
    if x > 10:
        break