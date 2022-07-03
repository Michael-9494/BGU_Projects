import scipy.io
import numpy as np
import os
from keras.models import Sequential
from keras.layers import  Dense, LSTM, Dropout, BatchNormalization
import keras
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
import sklearn.metrics as sk
from keras import backend as K 
import seaborn as sns 
#matplotlib qt

#change directory to your path 
path1 = r"C:/BGU/Semester 7/LastLabSignalProcessing/ECG"
os.chdir(path1)

#load training data
Data = scipy.io.loadmat('ECGarrayTrain.mat')
Data = Data['ECGarrayTrain']

Labels = scipy.io.loadmat('LabelsTrain.mat')
Labels = Labels['LabelsTrain']

Data = Data[:, :, np.newaxis]


#define neural network parameters
lstm1 = 256
do = 0.2
rdo = 0.0
lstm2 = 128
reg = 0.01
decay_rate = 0.95
batch_size = 64
learning_rate =0.0001

model = Sequential()
model.add((LSTM(lstm1, input_shape=(Data.shape[1], Data.shape[2]), 
                kernel_initializer= RandomNormal(mean=0.0, stddev=(2/lstm1)**0.5), kernel_regularizer=l2(reg), 
                recurrent_dropout = rdo, return_sequences=True)))
model.add(Dropout(do))
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model.add(LSTM(lstm2, kernel_initializer= RandomNormal(mean=0.0, stddev=(2/lstm2)**0.5), 
               recurrent_dropout = rdo, kernel_regularizer=l2(reg)))
model.add(Dropout(do))
model.add(Dense(1, activation='sigmoid'))

model.summary()

def lr_scheduler(epoch, lr):
                  return lr * decay_rate
              
callbacks= [LearningRateScheduler(lr_scheduler, verbose=1)]
                
opt = keras.optimizers.Adam(lr = learning_rate)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
                
history = model.fit(Data, Labels, batch_size = batch_size, epochs = 100, verbose = 1, callbacks = callbacks)

# run till here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#plotting learning curve of training data
history = history.history

acc = history['acc']

loss = history['loss']
  
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc)
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
  
plt.subplot(2, 1, 2)
plt.plot(loss)
plt.ylabel('Cross Entropy')
plt.title('Training Loss')
plt.xlabel('epoch')                   


TestData = scipy.io.loadmat('ECGarrayTest.mat')
TestData = TestData['ECGarrayTest']

TestLabels = scipy.io.loadmat('LabelsTest.mat')
TestLabels = TestLabels['LabelsTest']


TestData = TestData[:, :, np.newaxis]

outs_test = model.predict(TestData)
preds_test = np.zeros_like(outs_test)
for i in range(outs_test.shape[0]):
  if outs_test[i]>0.5:
    preds_test[i] = 1
Cmat_test=sk.confusion_matrix(TestLabels, preds_test)                  
print(Cmat_test)

scores = model.evaluate(TestData, TestLabels, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))#the first value is the loss, we want the accuracy so take the second

plt.figure(figsize=(8, 8))
sns.heatmap(Cmat_test, annot= True)

''' With Class Weight '''


ClassWeight =  np.floor(1*(Labels.shape[0]/np.sum(Labels))) - 1
class_weight = {0:1 , 1:ClassWeight}

K.clear_session()

model2 = Sequential()
model2.add((LSTM(lstm1, input_shape=(Data.shape[1], Data.shape[2]), 
                kernel_initializer= RandomNormal(mean=0.0, stddev=(2/lstm1)**0.5), kernel_regularizer=l2(reg), 
                recurrent_dropout = rdo, return_sequences=True)))
model2.add(Dropout(do))
model2.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
model2.add(LSTM(lstm2, kernel_initializer= RandomNormal(mean=0.0, stddev=(2/lstm2)**0.5), 
               recurrent_dropout = rdo, kernel_regularizer=l2(reg)))
model2.add(Dropout(do))
model2.add(Dense(1, activation='sigmoid'))

model2.summary()

def lr_scheduler(epoch, lr):
                  return lr * decay_rate
              
callbacks= [LearningRateScheduler(lr_scheduler, verbose=1)]
                
opt = keras.optimizers.Adam(lr = learning_rate)
model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
                
history2 = model2.fit(Data, Labels, batch_size = batch_size, epochs = 100, verbose = 1, 
                      callbacks = callbacks, class_weight = class_weight)

                
history2 = history2.history

acc = history2['acc']

loss = history2['loss']
  
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc)
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
  
plt.subplot(2, 1, 2)
plt.plot(loss)
plt.ylabel('Cross Entropy')
plt.title('Training Loss')
plt.xlabel('epoch')                   


TestData = scipy.io.loadmat('ECGarrayTest.mat')
TestData = TestData['ECGarrayTest']

TestLabels = scipy.io.loadmat('LabelsTest.mat')
TestLabels = TestLabels['LabelsTest']


TestData = TestData[:, :, np.newaxis]

outs_test = model2.predict(TestData)
preds_test = np.zeros_like(outs_test)
for i in range(outs_test.shape[0]):
  if outs_test[i]>0.5:
    preds_test[i] = 1
Cmat_test=sk.confusion_matrix(TestLabels, preds_test)                  
print(Cmat_test)

scores = model2.evaluate(TestData, TestLabels, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))#the first value is the loss, we want the accuracy so take the second
plt.figure(figsize=(8, 8))
sns.heatmap(Cmat_test, annot= True)