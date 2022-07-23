import cv2
import numpy as np
import os
from utils import *

#Main function

# Get path to the current working directory
CWD_PATH = os.getcwd()
video = cv2.VideoCapture('./input_videos/challenge.mp4')
outvideo = cv2.VideoWriter('./Results/curved_lanes_detection.avi', cv2.VideoWriter_fourcc(*'MJPG'),25, (1920,1080))
while True:

    _, frame = video.read()
    if frame is None:
        break
    ip = frame.copy()
    ip_resized = cv2.resize(ip,(640,360))
    birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)
    birdView_resized= cv2.resize(birdView,(640,360))
    img, hls, grayscale, thresh, blur, canny = processImage(birdView)
    imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
    imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)
    
    hist, leftBase, rightBase = histogram(thresh)
    ploty, left_fit, right_fit, left_fitx, right_fitx ,slide_img= slide_window_search(thresh, hist)
    slide_img_resized=cv2.resize(slide_img,(640,360))
    draw_info = fill_area(thresh, left_fit, right_fit)
    curveRad, curveDir = lane_curve(ploty, left_fitx, right_fitx)
    # Filling the area of detected lanes with green
    meanPts, result = color_lanes(frame, thresh, minverse, draw_info)
   
    # Adding text to our final image
    finalImg = addText(result, curveRad, curveDir)
    output_image = np.zeros((1080,1920,3),np.uint8)
    output_image[0:720,0:1280]=finalImg
    output_image[0:360,1280:1920]= birdView_resized
    output_image[360:720,1280:1920]=slide_img_resized
    output_image[720:1080,1280:1920]=ip_resized
    output_image=cv2.putText(output_image,'1. Final ouput ',(30,800),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1,cv2.LINE_AA)
    output_image=cv2.putText(output_image,'2. Warped Image ',(30,850),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1,cv2.LINE_AA)
    output_image=cv2.putText(output_image,'3. Slide window detected lanes ',(30,900),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1,cv2.LINE_AA)
    output_image=cv2.putText(output_image,'4. Input Image ',(30,950),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 0),1,cv2.LINE_AA)

    outvideo.write(output_image)
    # Displaying final image
    cv2.imshow("Final", output_image)
    
    # Wait for the ENTER key to be pressed to stop playback
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == 27:
       break

# Cleanup
video.release()
cv2.destroyAllWindows()