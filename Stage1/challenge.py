from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

batchSize = 32
imgSize = 224

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('E:/Github/Bitirme-Projesi/Akciger-Hastaliklari-Tespit-Sistemi/train', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')
validation_set=train_datagen.flow_from_directory('E:/Github/Bitirme-Projesi/Akciger-Hastaliklari-Tespit-Sistemi/validation', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')
test_set=test_datagen.flow_from_directory('E:/Github/Bitirme-Projesi/Akciger-Hastaliklari-Tespit-Sistemi/test', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical')

def createModel(classSize):
    totalLayerAmount = random.randint(5,26)
        
    kernelSize = random.randint(2, 4)
    filterAmount = random.randint(10, 64)
    model=Sequential()
    model.add(Conv2D(input_shape=(imgSize,imgSize,3), filters=filterAmount, kernel_size=(kernelSize,kernelSize), padding="same", activation="relu"))
    
    poolSize = random.randint(2, 4)
    strideSize = random.randint(2, 4)
    model.add(MaxPooling2D(pool_size=(poolSize,poolSize), strides=(strideSize,strideSize)))
    
    if totalLayerAmount < 12:
        denseLayerAmount = 2
    elif totalLayerAmount < 18:
        denseLayerAmount = 3
    else:
        denseLayerAmount = 4
    CNNLayerAmount = totalLayerAmount - denseLayerAmount
    
    while(CNNLayerAmount > 0):
        for i in range(random.randint(1, 3)):
            kernelSize = random.randint(2, 5)
            filterAmount = random.randint(10, 64)
            model.add(Conv2D(filters=filterAmount, kernel_size=(kernelSize,kernelSize), padding="same", activation="relu"))
            CNNLayerAmount -= 1
        
        if random.random() < 0.65:
            model.add(BatchNormalization())
            CNNLayerAmount -= 1
        
        poolSize = random.randint(2, 4)
        strideSize = random.randint(2, 4)
        model.add(MaxPooling2D(pool_size=(poolSize,poolSize), strides=(strideSize,strideSize)))
        
        CNNLayerAmount -= 1

    model.add(Flatten())
    
    for i in range(denseLayerAmount - 1):
        unitAmount = random.randint(256, 4096)
        model.add(Dense(units=unitAmount,activation="relu"))
        
        if random.random() < 0.66:
            dropoutRate = random.uniform(0.2, 0.75)
            model.add(Dropout(dropoutRate))
    
    model.add(Dense(units=classSize, activation="softmax"))
    return model

def runAndSaveModel(classSize, modelAmount):
    model = createModel(classSize)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    epochAmount = random.randint(15, 35)
    history = model.fit(training_set, steps_per_epoch=training_set.n//batchSize, epochs=epochAmount, validation_data=validation_set, validation_steps=validation_set.n//batchSize)
    model.save("model" + str(modelAmount))
    
for i in range(5):
    runAndSaveModel(4, i+1)
































