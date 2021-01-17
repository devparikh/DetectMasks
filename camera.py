import cv2
from keras.models import load_model
import cv2
import numpy as np

model = load_model('model-009.model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}


# defining face detector
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
       #extracting frames
        ret, frame = self.video.read()


        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
   
        faces=face_clsfr.detectMultiScale(gray,1.3,5)  

        for person_id, (x,y,w,h) in enumerate(faces):
            
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=model.predict(reshaped)

            label=np.argmax(result,axis=1)[0]

            text = labels_dict[label] + " :Person " +  str(person_id)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
            cv2.putText(frame, text, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()