import numpy as np # for image array operations
import cv2 # for image pre processing
import os  # for file navigation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

path = "myData"

allfile = os.listdir(path)

print(allfile)

images = [] # contains alll the image
classes = [] # contains repective class
print("Fetching All the classes "+str(allfile)+" ")
for x in allfile:
    print("Fetching class "+ x,end=" ")
    classpic = os.listdir(path+"/"+x)
    # print(classpic)
    # reading each image via Open CV
    for y in classpic:
        currentImg = cv2.resize(cv2.imread(path+"/"+x+"/"+y),(32,32))
        images.append(currentImg)
        classes.append(x)
    print(" Image in this class ", len(classpic))

X_train, X_test, y_train, y_test = train_test_split(images, classes, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img




# preprocess the data

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))


# Reshape the image From (**,32,32,3) -> (**,32,32,1)



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)


# DataGenator for increasing input data set

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=20)


dataGen.fit(X_train)


y_train = to_categorical(y_train,len(classes))
y_test = to_categorical(y_test,len(classes))
y_validation = to_categorical(y_validation,len(classes))

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2,2)
    noOfNodes= 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(32,
                      32,1),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(classes), activation='softmax'))

    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model


model = myModel()

print(model.summary())

history = model.fit(dataGen.flow(X_train,y_train,
                                 batch_size=50),
                                 steps_per_epoch=100,
                                 epochs=10,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1,
                                 verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("accuracy.png")
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")
plt.show()

pickle_out= open("model_trained_saved_pickle.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()

model.save('my_model_by_tf')
