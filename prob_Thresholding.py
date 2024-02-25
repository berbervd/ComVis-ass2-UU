import cv2
import numpy as np
import os

def automatic_threshold(diff_channel):
    # Calculate histogram of the difference image
    hist = cv2.calcHist([diff_channel], [0], None, [256], [0, 256])
    # Normalize histogram
    hist_norm = hist.ravel() / hist.max()
    Q = hist_norm.cumsum()

    bins = np.arange(256)

    fn_min = np.inf
    thresh = -1

    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # Return the optimal threshold value found
    print('Best threshold vals:{i} ', thresh, '\n volgende: \n')
    return thresh


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

            # Thresholding automatically: 
            # abs difference and then automatic thresholding
            ### H
            diff_h = cv2.absdiff(frame_h, background_h)
            thresh_h = automatic_threshold(diff_h)
            _, fg_mask_h = cv2.threshold(diff_h, thresh_h, 255, cv2.THRESH_BINARY)

            ### S
            diff_s = cv2.absdiff(frame_s, background_s)
            thresh_s = automatic_threshold(diff_s)
            _, fg_mask_s = cv2.threshold(diff_s, thresh_s, 255, cv2.THRESH_BINARY)

            ### V
            diff_v = cv2.absdiff(frame_v, background_v)
            thresh_v = automatic_threshold(diff_v)
            _, fg_mask_v = cv2.threshold(diff_v, thresh_v, 255, cv2.THRESH_BINARY)

            # Combine channels and refine mask using morphology
            fg_mask_combined = cv2.bitwise_or(fg_mask_h, fg_mask_s)
            fg_mask_combined = cv2.bitwise_or(fg_mask_combined, fg_mask_v)
    

            """
            # abs difference between current frame & background per channel
            diff_h = cv2.absdiff(frame_h, background_h)
            diff_s = cv2.absdiff(frame_s, background_s)
            diff_v = cv2.absdiff(frame_v, background_v)

            # Threshold > per channel 
            #thresh = 30  # SINGLE FRAME
            # Apply thresholding for each channel
            thresh_h = 40  
            thresh_s = 40  
            thresh_v = 50 

            # Thresholding for difference for foreground mask >> per HSV
            _, fg_mask_h = cv2.threshold(diff_h, thresh_h, 255, cv2.THRESH_BINARY)
            _, fg_mask_s = cv2.threshold(diff_s, thresh_s, 255, cv2.THRESH_BINARY)
            _, fg_mask_v = cv2.threshold(diff_v, thresh_v, 255, cv2.THRESH_BINARY)
  
            # Combine channels: bitwise_and en bitwise_or
            fg_mask_combined = cv2.bitwise_or(fg_mask_h, fg_mask_s)
            fg_mask_combined = cv2.bitwise_or(fg_mask_combined, fg_mask_v)

            """
            
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
            if cv2.waitKey(0) & 0xFF == ord('q'):  # q klikken om door te gaan
                break

        cap.release()
    cv2.destroyAllWindows()

background_subtraction('data')
