# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.core import Dense,Activation,Dropout,Flatten
from tensorflow.keras import datasets, layers, models, optimizer
from keras.datasets import cifar100

(train_set,train_label),(test_set,test_label) = cifar100.load_data()

train_set = train_set.astype('float32')/255.0
test_set = test_set.astype('float32')/255.0

from keras.utils import np_utils

train_label = np_utils.to_categorical(train_label,100)
test_label = np_utils.to_categorical(test_label,100)

classify = Sequential()

classify.add(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)))

"""

classify.add(Conv2D(32,(3,3),activation='relu',padding='same'))

classify.add(MaxPool2D(pool_size=(2,2)))

classify.add(Dropout(0.25))

classify.add(Conv2D(64,(3,3),activation='relu',padding='same'))

"""

classify.add(MaxPool2D(pool_size=(2,2)))

classify.add(Dropout(0.25))

classify.add(Flatten())

classify.add(Dense(32,activation='relu'))

classify.add(Dropout(0.5))

classify.add(Dense(100,activation='softmax'))

classify.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
 
classify.summary()

train_nn = classify.fit(train_set,train_label,batch_size=128, epochs =50,verbose=1)

score = classify.evaluate(test_set,test_label,verbose=0)
print('deviation',score[0])
print('accuracy:',score[1])

prediction = classify.predict_classes(test_set)

test=np.random.randint(0,10000)

plt.imshow(test_set[test])
plt.title('prediction:{0},index:{1}'.format(prediction[test],test_label[test]))
plt.show()

