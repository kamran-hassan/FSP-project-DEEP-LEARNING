import numpy as np
import cv2
import pickle

import tensorflow as tf


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

model = tf.keras.models.load_model('my_model_by_tf')
vid = cv2.VideoCapture(1)
width = 640
height = 480

vid.set(3, width)
vid.set(4, height)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    img = preProcessing(frame)
    img = cv2.resize(img, (32, 32))
    img = np.asarray(img)
    img = img.reshape(1, 32, 32, 1)

    predict_x = model.predict(img)
    classes_x = np.argmax(predict_x, axis=1)
    # print(classes_x[0])
    st = "predicted value "+str(classes_x[0])
    cv2.putText(frame,st,
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

exit()

"""
while True:
    
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    classIndex = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(classIndex,probVal)

    if probVal> threshold:
        cv2.putText(imgOriginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    success, imgOriginal = cap.read()

    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image", imgOriginal)
    """







img = cv2.imread('myData/7/img008-00001.png')





img = preProcessing(img)

print(img)


cv2.imshow('win', img)
cv2.waitKey(0)

img = cv2.resize(img, (32, 32))

img = np.asarray(img)
img = img.reshape(1, 32, 32, 1)

#print(img.shape)


model = tf.keras.models.load_model('my_model_by_tf')


predict_x = model.predict(img)
classes_x = np.argmax(predict_x,axis=1)
print(predict_x)
print(classes_x)