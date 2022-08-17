from keras.models import load_model
import numpy as np
import math
import random

model = load_model('E:/Github/CNN-Challenge/Challenge/model2')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batchSize = 32
imgSize = 224
train_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory('E:/Github/CNN-Challenge/Challenge/train',
                                          target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical', shuffle = False)

fnames = train_set.filenames
ground_truth = train_set.classes
label2index = train_set.class_indices
idx2label = dict((v,k) for k,v in label2index.items())
predictions = model.predict_generator(train_set, steps=train_set.samples/train_set.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)
errors = np.where(predicted_classes != ground_truth)[0]
totalError = len(errors) / train_set.n
significance = abs(0.5 * math.log2((1 - totalError) / totalError))
array = np.empty((0,6))

for i in range(train_set.n):
    path = fnames[i].split('\\')[1]
    label = fnames[i].split('\\')[0]
    pred_label = idx2label[np.argmax(predictions[i])]

    weight = 1 / train_set.n
    if label == pred_label:
        weight *= pow(math.e, -significance)
    else:
        weight *= pow(math.e, significance)
        
    row = [path, label, pred_label, weight, 0, 0]
    array = np.vstack([array, row])

sumOfWeights = sum(list(map(float, array[: , 3])))
thresholdValue = 0
for row in array:
    scaledWeight = float(row[3]) / sumOfWeights
    thresholdValue += scaledWeight
    row[4] = scaledWeight
    row[5] = thresholdValue
    
"""
for i in range(2998,3000):
    a = 'Path : {}, Original Label : {}, Prediction : {}, Weight : {:.6f}, Scaled Weight : {:.6f}, Threshold : {:.6f}'.format(
        array[i][0],
        array[i][1],
        array[i][2],
        float(array[i][3]),
        float(array[i][4]),
        float(array[i][5]))
    print(a)
"""  

newDataset = np.empty((0,6))
for i in range(train_set.n):
    randomNum = random.random()
    
    mid = int(math.ceil(len(array) / 2)) - 1
    bot = 0
    top = len(array)
    
    while True:
        if randomNum < float(array[mid][5]):
            top = mid
            last = mid
        elif randomNum > float(array[mid][5]):
            bot = mid
        else:
            newDataset = np.vstack([newDataset, array[mid]])
            break
        
        lastMid = mid
        mid = int((top + bot) / 2)
        
        if lastMid == mid:
            newDataset = np.vstack([newDataset, array[last]])
            break
       
"""
for i in range(10):
    a = 'Path : {}, Original Label : {}, Prediction : {}, Weight : {:.6f}, Scaled Weight : {:.6f}, Threshold : {:.6f}'.format(
        newDataset[i][0],
        newDataset[i][1],
        newDataset[i][2],
        float(newDataset[i][3]),
        float(newDataset[i][4]),
        float(newDataset[i][5]))
    print(a)
""" 
        
        
        
        