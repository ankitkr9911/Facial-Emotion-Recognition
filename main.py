import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, Sequential

with open('network_emotions.json','r') as json_file:
  json_saved_model = json_file.read()
json_saved_model


network_loaded = model_from_json(json_saved_model, custom_objects={'Sequential': Sequential})
network_loaded.load_weights('weights_emotions.hdf5')
network_loaded.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0 :
    connected,frame = video.read()
    if not connected:
        break

    faces = face_detector.detectMultiScale(frame)
    for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            roi = frame[y:y+h,x:x+w]
            roi = cv2.resize(roi,(48,48))
            roi = roi / 255
            roi = np.expand_dims(roi,axis=0)

            prediction = network_loaded.predict(roi)
            if prediction is not None:
                cv2.putText(frame,emotions[np.argmax(prediction)],(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)

            if cv2.waitKey(1) and 0XFF == ord('q'):
                break

    cv2.imshow("Emotions",frame)

cv2.destroyAllWindows()



