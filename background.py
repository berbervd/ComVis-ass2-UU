
import cv2
import numpy as np
import os



"""
This script does the background subtraction. 
First we created the background image bu averaging the frames per video,
however we decided to do the background subtraction via the gmm_background_image which uses openCV's createBackgroundSubtractorMOG2 function (for the choice task on automatic thresholding)
"""



# so this function creates for each camera a background image using the average of the frames per camera 
# The output is 4 images with the background per umage. 
def background_image(base_path='data', save_image=True): # if true then save the output
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

            float_frame = frame.astype(np.float32) # aantal frames miss nog aanpassen
            if acc is None:
                acc = float_frame
            else:
                acc += float_frame
            frame_count += 1
        #print(frame_count)
        #print(float_frame)
            
        # average all frames per camera 
        avg_frame = acc / frame_count
        background_model = np.uint8(avg_frame)
        cap.release()

        if save_image:
            save_path = os.path.join(base_path, cam_dir, f'background_images/background_{cam_dir}.jpg')
            cv2.imwrite(save_path, background_model)
            
        
    return background_model
# Om even te testen of het werkt kan je zo de funtie aanroepen. Maar hoef ipc maar 1x tenzij we de methode veranderen naar bijv die GMM
#background_image('data', False) # of true then save the output 
 
def gmm_background_image(base_path='data', save_image=True):
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']  # Camera folders
    for cam_dir in camera_dirs:
        video_path = os.path.join(base_path, cam_dir, 'background.avi')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: video not visible for {cam_dir}.")
            continue

        # GMM way for background sub: 
        # for dynamic backgrounds better a lower value 100-300? for history : 500 deault
        # var threshold: (16 default): lower more sensitive (detecting more shadows), higher: less sensitive
        backSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)


        # training the background model
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Do background sub for each frame
            fg_mask = backSub.apply(frame)

        # Release the video capture object
        cap.release()

        background_model = backSub.getBackgroundImage() if hasattr(backSub, 'getBackgroundImage') else frame

        if save_image:
            save_path = os.path.join(base_path, cam_dir, f'background_images/backgroundGMM_{cam_dir}.jpg')
            cv2.imwrite(save_path, background_model)
            

gmm_background_image('data', save_image=True)

"""
Background subtraction for the 4 cams

Feedback voor dit:
Regarding the background subtraction results: these are decent but could be a bit better in terms of the holes inside the silhouette, and shadows around the feet. 
The noise in the ceiling, along the edges of the carpet and walls (mostly cam 2), and because of the chessboard that should not have been put there (cam 3) is fine.

"""
 


def background_subtraction(base_path='data'):
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']  # Camera folders

    for cam_dir in camera_dirs:
        background_path = os.path.join(base_path, cam_dir, f'background_images/backgroundGMM_{cam_dir}.jpg')
        video_path = os.path.join(base_path, cam_dir, 'video.avi')

        # Load the background image
        background_img = cv2.imread(background_path)
        if background_img is None:
            print(f"Failed for {cam_dir}")
            continue

        """
        Niet zeker of blur ook bij background immage moet worden toegepast?
        """
        # Gaussian blur >> the background image
        gaus_blur_background_img = cv2.GaussianBlur(background_img, (5, 5), 0)
        # IMG to HSV
        background_hsv = cv2.cvtColor(gaus_blur_background_img, cv2.COLOR_BGR2HSV)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video for {cam_dir}")
            continue
        
        # Go through each frame 
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Gaussian blur >> reduces noise (applied to the whole frame. could try applying after the frames are splitted?)
            gaus_blur = cv2.GaussianBlur(frame, (5, 5), 0)

            # Convert blurred current frame to HSV
            frame_hsv = cv2.cvtColor(gaus_blur, cv2.COLOR_BGR2HSV)

            # Split frames
            frame_h, frame_s, frame_v = cv2.split(frame_hsv)
            background_h, background_s, background_v = cv2.split(background_hsv)

            # abs difference between current frame & background per channel
            diff_h = cv2.absdiff(frame_h, background_h)
            diff_s = cv2.absdiff(frame_s, background_s)
            diff_v = cv2.absdiff(frame_v, background_v)

            # Threshold > per channel 
            # Apply thresholding for each channel : the otsu finds the optimal threshold value 
            # kijken of threshold werklt of misschien adaptive ? threshold VS adaptiveThreshold
            _, fg_mask_h = cv2.threshold(diff_h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # + cv.THRESH_OTSU >> lijkt niet beter
            _, fg_mask_s = cv2.threshold(diff_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, fg_mask_v = cv2.threshold(diff_v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Combine channels: bitwise_and en bitwise_or
            fg_mask_combined = cv2.bitwise_or(fg_mask_h, fg_mask_s)
            fg_mask_combined = cv2.bitwise_or(fg_mask_combined, fg_mask_v)
            #print(fg_mask_h)
            print(fg_mask_combined)


            ### POST DETECTION 
            # Dit doet meer cleanen van de achterground ruis dan van het persoontje?
            # Optionally >> apply morphological operations to "clean" foreground mask
            # moet nog ff gefinetuned worden ?!?1
            kernel = np.ones((3,3), np.uint8)
            # op gecombineerde
            fg_mask_combined = cv2.erode(fg_mask_combined, kernel, iterations=1)
            fg_mask_combined = cv2.dilate(fg_mask_combined, kernel, iterations=1)
 
            # Als je deze in de imshow stopt ipv fg_mask_combined zie je de echte foto (kunnen we beter de gaten zien)
            frame_real = cv2.bitwise_and(frame, frame, mask=fg_mask_combined)

            # foreground mask
            # fg_mask_combined voor zwart wit, frame_real voor de echte foto foreground
            cv2.imshow(f'Foreground for {cam_dir}', fg_mask_combined) 
            save_foreground = os.path.join(base_path, cam_dir, 'foreground_image.jpg')
            cv2.imwrite(save_foreground, fg_mask_combined)
            if cv2.waitKey(0) & 0xFF == ord('q'):  # q klikken om door te gaan
                break

        cap.release()
    cv2.destroyAllWindows()

background_subtraction('data')


