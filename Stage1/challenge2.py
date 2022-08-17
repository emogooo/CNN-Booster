from keras.models import load_model

model1 = load_model(('model1'))
model2 = load_model(('model2'))
model3 = load_model(('model3'))
model4 = load_model(('model4'))
model5 = load_model(('model5'))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
batchSize = 32
imgSize = 224
train_datagen=ImageDataGenerator(rescale=1./255)
train_set=train_datagen.flow_from_directory('E:/Github/Bitirme-Projesi/Akciger-Hastaliklari-Tespit-Sistemi/Challenge/train',
                                          target_size=(imgSize,imgSize), batch_size=batchSize, class_mode='categorical', shuffle = False)

loss1, acc1 = model1.evaluate(train_set)
loss2, acc2 = model2.evaluate(train_set)
loss3, acc3 = model3.evaluate(train_set)
loss4, acc4 = model4.evaluate(train_set)
loss5, acc5 = model5.evaluate(train_set)

print("1. Model: Loss: " + "{:.2f}".format(loss1) + " Accuracy: " + "{:.3f}".format(acc1))
print("2. Model: Loss: " + "{:.2f}".format(loss2) + " Accuracy: " + "{:.3f}".format(acc2))
print("3. Model: Loss: " + "{:.2f}".format(loss3) + " Accuracy: " + "{:.3f}".format(acc3))
print("4. Model: Loss: " + "{:.2f}".format(loss4) + " Accuracy: " + "{:.3f}".format(acc4))
print("5. Model: Loss: " + "{:.2f}".format(loss5) + " Accuracy: " + "{:.3f}".format(acc5))

#En iyi model: model2 Loss: 1.14 Accuracy: 0.486
