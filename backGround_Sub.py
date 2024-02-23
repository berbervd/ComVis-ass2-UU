import cv2
import numpy as np
import os

"""
Step 1: create background model for the frames. 
First: 'simple way'>> averaging the frames. 
Later try GMM approach?

The background images are now saved and stored in their respective folders. These are the ones by the 'averaging method'
"""

# so this function creates for each camera a background image using the average of the frames per camera 
# The output is 4 images with the background per umage. 
def background_model(base_path='data', save_image=True): # if true then save the output
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']  # Camera folders
    # for all cams separately 
    for cam_dir in camera_dirs:
        video_path = os.path.join(base_path, cam_dir, 'background.avi')
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video in {cam_dir}.")
            continue
        
        # loop over all frames
        acc = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            float_frame = frame.astype(np.float32)
            if acc is None:
                acc = float_frame
            else:
                acc += float_frame
            frame_count += 1
        #print(frame_count)
            
        # average all frames per camera 
        avg_frame = acc / frame_count
        background_model = np.uint8(avg_frame)
        cap.release()

        if save_image:
            save_path = os.path.join(base_path, cam_dir, f'background_{cam_dir}.jpg')
            cv2.imwrite(save_path, background_model)
            print(f"Background image saved for {cam_dir}.")

# Om even te testen of het werkt kan je zo de funtie aanroepen. Maar hoef ipc maar 1x tenzij we de methode veranderen naar bijv die GMM
#background_model('data', True) # of true then save the output 
 
