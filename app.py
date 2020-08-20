import cv2
import tools
import consts
import numpy as np
from keras.models import load_model

model = load_model('ASL96.h5')

cap = cv2.VideoCapture(0)

past = [None] * 10
sentence = []

while(True):

    if len(past) > 20:
        del past[0]
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.rectangle(frame,(100,100),(400,400),(255,0,0),3)
    image = frame[100:400,100:400]
    image = cv2.resize(image,consts.shape)
    input = np.reshape(image,consts.shape_for_nsamples(1))/255.0
    prediction = tools.getPrediction(model.predict(input)[0])
    past.append(prediction)

    if tools.check(past):
       if not prediction == None  and not prediction == 'nothing':
          if prediction == 'space':
            sentence.append(' ')
          elif prediction == 'del':
            del sentence[len(sentence) - 1]
          else:
            sentence.append(prediction)
          past = [None] * 20

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,prediction,(600,100),font, 4,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(frame,''.join(sentence),(100,600),font, 4,(0,0,255),2,cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
