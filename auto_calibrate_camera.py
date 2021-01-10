import numpy as np
import cv2
import time
import pickle

cap = cv2.VideoCapture(0)

# Define the dimensions of checkerboard 
CHECKERBOARD = (6, 9) 
  
  
# stop the iteration when specified 
# accuracy, epsilon, is reached or 
# specified number of iterations are completed. 
criteria = (cv2.TERM_CRITERIA_EPS + 
            cv2.TERM_CRITERIA_MAX_ITER, 60, 0.001) 
  
  
# Vector for 3D points 
points_3D = [] 
  
# Vector for 2D points 
points_2D = [] 
  
  
#  3D points real world coordinates 
objectp3d = np.zeros((1, CHECKERBOARD[0]  
                      * CHECKERBOARD[1],  
                      3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 
                               0:CHECKERBOARD[1]].T.reshape(-1, 2) 
prev_img_shape = None
  
cv2.namedWindow('Calibration Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Calibration Image', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
for i in range(1,26,2
               ):
    img = cv2.imread(f'assets/calibration/patterns/pattern_{i}.png',0)
    cv2.imshow('Calibration Image',img)
    cv2.waitKey(500)
    ret, frame = cap.read()
    cv2.waitKey(500)
    og_frame = frame.copy()
    cv2.imwrite(f'assets/calibration/captures/capture_{i}.jpg',frame)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
  
    # Find the chess board corners 
    # If desired number of corners are 
    # found in the image then ret = true 
    ret, corners = cv2.findChessboardCorners( 
                    gray_frame, CHECKERBOARD,  
                    cv2.CALIB_CB_ADAPTIVE_THRESH  
                    + cv2.CALIB_CB_FAST_CHECK + 
                    cv2.CALIB_CB_NORMALIZE_IMAGE) 
  
    # If desired number of corners can be detected then, 
    # refine the pixel coordinates and display 
    # them on the images of checker board 
    if ret == True: 
        points_3D.append(objectp3d) 
  
        # Refining pixel coordinates 
        # for given 2d points. 
        corners2 = cv2.cornerSubPix( 
            gray_frame, corners, (11, 11), (-1, -1), criteria) 
  
        points_2D.append(corners2) 
  
        # Draw and display the corners 
        image = cv2.drawChessboardCorners(og_frame,  
                                          CHECKERBOARD,  
                                          corners2, ret)
        cv2.imwrite(f'/home/pi/Documents/root/assets/calibration/drawchessboard_results/chessboard_result_{i}.jpg', image)

cv2.destroyAllWindows()

ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
    points_3D, points_2D, gray_frame.shape[::-1], None, None)

camera_params = {'matrix': matrix, 'distortion': distortion}
file = open('camera_params.txt', 'wb')
pickle.dump(camera_params, file)
file.close()

# Test calibration matrix
for i in range(1,10):
    print(i)
    img = cv2.imread(f'/home/pi/Documents/root/assets/calibration/distortion_test_images/test{i}.jpg')
    
    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 0, (w,h))

    # undistort
    dst = cv2.undistort(img, matrix, distortion, None, newcameramtx)

    # crop the image
    cv2.imwrite(f'assets/calibration/distortion_test_results/result{i}.jpg', dst)
    

