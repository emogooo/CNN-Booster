from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

batchSize = 32
imgSize = 224

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/images/train', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary')
validation_set=train_datagen.flow_from_directory('E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/images/validation', target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='binary')

def createModel():
    kernelSize = random.randint(2, 4)
    filterAmount = random.randint(10, 64)
    model=Sequential()
    model.add(Conv2D(input_shape=(imgSize,imgSize,3), filters=filterAmount, kernel_size=(kernelSize,kernelSize), padding="same", activation="relu"))
    
    poolSize = random.randint(2, 6)
    strideSize = random.randint(2, 6)
    model.add(MaxPooling2D(pool_size=(poolSize,poolSize), strides=(strideSize,strideSize), padding="same"))
    
    CNNLayerAmount = random.randint(7,13)
    
    for i in range(CNNLayerAmount):
        kernelSize = random.randint(2, 5)
        filterAmount = random.randint(10, 64)
        model.add(Conv2D(filters=filterAmount, kernel_size=(kernelSize,kernelSize), padding="same", activation="relu"))
        
        if random.random() < 0.4:
            model.add(BatchNormalization())
        
        poolSize = random.randint(2, 6)
        strideSize = random.randint(2, 6)
        model.add(MaxPooling2D(pool_size=(poolSize,poolSize), strides=(strideSize,strideSize), padding="same"))

    model.add(Flatten())
    
    denseLayerAmount = random.randint(1,3)
   
    for i in range(denseLayerAmount):
        unitAmount = random.randint(128, 1024)
        model.add(Dense(units=unitAmount,activation="relu"))
        
        if random.random() < 0.66:
            dropoutRate = random.uniform(0.2, 0.75)
            model.add(Dropout(dropoutRate))
    
    model.add(Dense(units=1, activation="sigmoid"))
    return model

def runAndSaveModel(modelAmount):
    model = createModel()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    epochAmount = random.randint(8, 17)
    model.fit(training_set, steps_per_epoch=training_set.n//batchSize, epochs=epochAmount, validation_data=validation_set, validation_steps=validation_set.n//batchSize)
    model.save("E:/Github/CNN-Challenge-Workshop/Stage2-Workshop/models/model" + str(modelAmount))
    
for i in range(10):
    runAndSaveModel(i+1)
































