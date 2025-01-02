# -*- coding: utf-8 -*-


# %pip install mediapipe
# %pip install cv2
import cv2
import mediapipe as mp

from google.protobuf.json_format import MessageToDict

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, model_complexity=1,
                      min_detection_confidence=0.75, min_tracking_confidence=0.75, max_num_hands=2)

capture = cv2.VideoCapture(0)
if capture.read():
    while True:
        success, img = capture.read()
        flipedImage = cv2.flip(img, 1)
        RGBImage = cv2.cvtColor(flipedImage, cv2.COLOR_BGR2RGB)
        results = hands.process(RGBImage)
        if cv2.waitKey(1) == ord('q'):
            break

        if results.multi_hand_landmarks:

            if len(results.multi_handedness) == 2:
                cv2.putText(img, 'Both Hands', (250, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            else:
                for i in results.multi_handedness:
                    label = MessageToDict(i)['classification'][0]['label']
                    cv2.putText(img, 'Left' if label == 'Left' else 'Right', (250, 50) if label == 'Left' else (450, 50),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            cv2.destroyAllWindows()
            break
