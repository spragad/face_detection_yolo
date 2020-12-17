
import os
print(os.getcwd())
os.chdir(".../faced-master/")

import cv2
from faced import FaceDetector
from faced.utils import annotate_image
from time import process_time

#___________________________________________________For Image______________________________________________________
face_detector = FaceDetector()

img = cv2.imread("face_det.jpg")

rgb_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

# Receives RGB numpy image (HxWxC) and
# returns (x_center, y_center, width, height, prob) tuples. 
bboxes = face_detector.predict(rgb_img, 0.7)

# Use this utils function to annotate the image.
ann_img = annotate_image(img, bboxes)

#save img
cv2.imwrite('face_detd.jpg', ann_img)

# Show the image
cv2.imshow('Result',ann_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#____________________________________________________For Video_______________________________________________________
video='Vid.mp4'
cap = cv2.VideoCapture(video)
face_detector = FaceDetector()

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height)

result = cv2.VideoWriter('Face_det_out.mp4',cv2.VideoWriter_fourcc(*'XVID'), 15, size)
pro_time=[]
while(True):
    t1_start = process_time() 
    ret, frame = cap.read()
    if ret== True:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Receives RGB numpy image (HxWxC) and
        # returns (x_center, y_center, width, height, prob) tuples. 
        bboxes = face_detector.predict(rgb_img, 0.7)

        # Use this utils function to annotate the image.
        ann_img = annotate_image(frame, bboxes)

        # Save video
        result.write(ann_img)
        
        # Show the image
        cv2.imshow('Result',ann_img)
        # quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    t1_stop = process_time() 
    pro_time.append(t1_stop-t1_start)
    
    
cap.release()
result.release()
cv2.destroyAllWindows()
print("Average Procesing time per frame: ",sum(pro_time)/len(pro_time))
