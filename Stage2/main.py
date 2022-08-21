from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import random
import os

batchSize = 32
imgSize = 224

test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/images/test',
                                          target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary', shuffle = False)

labels = test_set.classes

modelNames = os.listdir('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/')
modelPredictions = list()
#bestAcc = 0
#bestLoss = 0
#averageAcc = 0
for name in modelNames:
    model = load_model(('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/' + name))
    modelPrediction = model.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)
    modelPredictions.append(modelPrediction)
    
    """
    model = load_model(('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/' + name))
    loss, acc = model.evaluate(test_set)
    averageAcc += acc
    if bestAcc < acc:
        bestAcc = acc
        bestLoss = loss
    
    """
#averageAcc /= len(modelNames)
#print("En iyi Modelin: Loss: " + "{:.2f}".format(bestLoss) + " Accuracy: " + "{:.3f}".format(bestAcc) + " Average Accuracy: " + "{:.3f}".format(averageAcc))

def generateFirstGeneration(populationNumber):
    firstGeneration = list()
    for i in range(populationNumber):  
        weights = list()
        for j in range(len(modelNames)):
            weights.append(random.random())

        sumOfWeights = sum(weights)
        
        for i in range(len(weights)):
            weights[i] /= sumOfWeights
        firstGeneration.append(list((weights, 0)))
    return firstGeneration
    
def calculateGenerationScore(generation):
    for chromosome in generation:
        newPredictions = chromosome[0][0] * modelPredictions[0]
        for i in range(1, len(modelNames)):
            newPredictions += chromosome[0][i] * modelPredictions[i]
        
        newPredictions = [round(float(item)) for item in newPredictions]
        score = 0
        for i in range(test_set.n):
            if newPredictions[i] == labels[i]:
                score += 1
        score /= test_set.n
        chromosome[1] = score
    return generation
 
def getBestOfXPercentage(X, generation):
    sortedGeneration = sorted(generation, key=lambda kv: kv[1], reverse=True)
    chromosomeAmount = int(len(sortedGeneration) * X)
    return sortedGeneration[:chromosomeAmount]

def generateNewGeneration(population, generation):
    newGeneration = list()
    for i in range(population):
        x = random.randint(0, len(generation)-1)
        y = random.randint(0, len(generation)-1)
        while x == y:
            y = random.randint(0, len(generation)-1)            
        num = random.randint(1, len(modelNames) - 1)
        newChromosome = list()
        for i in range(num):
            newChromosome.append(generation[x][0][i])
        for i in range(num, len(modelNames)):
            newChromosome.append(generation[y][0][i])  
        sumOfWeights = sum(newChromosome)
        for i in range(len(newChromosome)):
            newChromosome[i] /= sumOfWeights   
        newGeneration.append(list((newChromosome, 0)))
    return newGeneration 

def run():
    firstGeneration = generateFirstGeneration(1000)
    firstGenerationScores = calculateGenerationScore(firstGeneration)
    best20OfGeneration = getBestOfXPercentage(0.2, firstGenerationScores)
    bestScore = best20OfGeneration[0][1]
    bestChromosome = best20OfGeneration[0][0]
    bestGeneration = 0
    
    for i in range(100):
        newGeneration = generateNewGeneration(1000, best20OfGeneration)
        newGenerationScores = calculateGenerationScore(newGeneration)
        best20OfGeneration = getBestOfXPercentage(0.2, newGenerationScores)
        if bestScore < best20OfGeneration[0][1]:
            bestScore = best20OfGeneration[0][1]
            bestChromosome = best20OfGeneration[0][0]
            bestGeneration = i + 1
            
    return bestScore, bestGeneration, bestChromosome
    

def main():
    bestScore = 0.1
    bestGen = -1
    bestChromosome = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bestRun = 0
    runCounter = 0
    
    while bestScore < 0.85:
        runCounter += 1
        score, gen, chromosome = run()
        if score > bestScore:
            bestScore = score
            bestGen = gen
            bestChromosome = chromosome
            bestRun = runCounter
            from datetime import datetime
            now = datetime.now()  
            print ("%s.%s.%s - 0%s:%s" % (now.day,now.month,now.year,now.hour,now.minute)) 
            print("En iyi nesil: ", str(bestGen), "\nEn iyi skor: ", str(bestScore), "\nEn iyi kromozom: ", str(bestChromosome), "\nEn iyi sonucun üretildiği çalışma: ", str(bestRun), "\n")
        
    from datetime import datetime
    now = datetime.now()  
    print ("%s.%s.%s - 0%s:%s" % (now.day,now.month,now.year,now.hour,now.minute)) 
    print("En iyi nesil: ", str(bestGen), "\nEn iyi skor: ", str(bestScore), "\nEn iyi kromozom: ", str(bestChromosome), "\nEn iyi sonucun üretildiği çalışma: ", str(bestRun), "\n")

main()