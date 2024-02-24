"""
TO DO / CHECKEN:
* background in color or gray?

"""
import cv2
import numpy as np
import os

"""
Tips:
* Background subtraction
1. OpenCV has some methods that can "automatically" learn the background, and can provide the background without having to set thresholds. 
Check out the available functions.
2. for the quality of the background subtraction, 
I will mainly be looking at the presence of holes in the foreground (apart from the shape in the t-shirt that we might ignore, 
there these should not be there) and whether the legs of the chair are not "glued" together with the legs of the person.
3. it's better to have a bit more foreground pixels than to have foreground pixels that are missing. 
Everytime a foreground pixel is missing, it typically leads to a how line of voxels missing. 
See also Voxel reconstruction point 2 below.
"""


"""
Step 1: create background model for the frames. 
First: 'simple way'>> averaging the frames. 
Later try GMM approach?

The background images are now saved and stored in their respective folders. These are the ones by the 'averaging method'
"""
def gmm_background_image(base_path='data', save_image=True):
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']  # Camera folders
    for cam_dir in camera_dirs:
        video_path = os.path.join(base_path, cam_dir, 'background.avi')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video in {cam_dir}.")
            continue

        # Create a background subtractor object with history to cover the entire video
        backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update the background model
            fg_mask = backSub.apply(frame)

        # Release the video capture object
        cap.release()

        # The background model should have learned the background by now
        # Let's get the background image from the model
        # For some OpenCV versions, you might directly use backSub.getBackgroundImage()
        # But in some versions, you need to grab the background from the last frame
        background_model = backSub.getBackgroundImage() if hasattr(backSub, 'getBackgroundImage') else frame

        if save_image:
            save_path = os.path.join(base_path, cam_dir, f'backgroundGMM_{cam_dir}.jpg')
            cv2.imwrite(save_path, background_model)
            print(f"Background image saved for {cam_dir}.")

#gmm_background_image('data', save_image=True)


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
            save_path = os.path.join(base_path, cam_dir, f'background_{cam_dir}.jpg')
            cv2.imwrite(save_path, background_model)
            print(f"Background image saved for {cam_dir}.")
        
    return background_model
        

# Om even te testen of het werkt kan je zo de funtie aanroepen. Maar hoef ipc maar 1x tenzij we de methode veranderen naar bijv die GMM
#background_image('data', False) # of true then save the output 
 





"""
Background subtraction for the 4 cams

Feedback voor dit:
Regarding the background subtraction results: these are decent but could be a bit better in terms of the holes inside the silhouette, and shadows around the feet. 
The noise in the ceiling, along the edges of the carpet and walls (mostly cam 2), and because of the chessboard that should not have been put there (cam 3) is fine.

"""

def background_subtraction(base_path='data'): # (backgroun_model, base_path?)
    camera_dirs = ['cam1', 'cam2', 'cam3', 'cam4']  # Camera folders

    for cam_dir in camera_dirs:
        # backgroundGMM_ of background_ (afhankelijk of we de GMM of average background image gebruikine. Heb nog weinig vershcil gezien)
        background_path = os.path.join(base_path, cam_dir, f'backgroundGMM_{cam_dir}.jpg') # eerst background_image voor de background images per camera.
        video_path = os.path.join(base_path, cam_dir, 'video.avi')

        # Load the background image
        background_img = cv2.imread(background_path)
        if background_img is None:
            print(f"Failed to load background image for {cam_dir}")
            continue

        # Convert  background im to HSV
        background_hsv = cv2.cvtColor(background_img, cv2.COLOR_BGR2HSV)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video for {cam_dir}")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            #  current frame to HSV
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # abs difference between current frame & background
            diff = cv2.absdiff(frame_hsv, background_hsv)

            """
            threshold aanpassen voor meer accurate background subt
            >> methode voor?
            """

            # Apply thresholding to identify foreground
            thresh = 30  # 

            _, fg_mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

            # Optionally >> apply morphological operations to clean up the foreground mask
            # moet nog ff gefinetuned worden 
            kernel = np.ones((3,3), np.uint8)
            fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

            # Display the foreground mask
            cv2.imshow(f'Foreground for {cam_dir}', fg_mask)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
    cv2.destroyAllWindows()

background_subtraction('data')



"""
To improve your background subtraction based on the provided images and the description of your task, you may consider the following suggestions:

Gaussian Mixture Model (GMM):

Replace the averaging method with a GMM to handle variations in the background more robustly. OpenCV provides the BackgroundSubtractorMOG2 class, which implements such a model.
Dynamic Thresholding:

Instead of using a static threshold, consider using methods like Otsu's thresholding or adaptive thresholding which can automatically adjust the threshold based on the image characteristics.
Channel-wise Thresholding:

Perform thresholding on each channel (Hue, Saturation, and Value) separately and then combine the results to determine foreground pixels. This can help in handling the color variations in the images more effectively.
Morphological Operations:

Use morphological operations like erosion and dilation more strategically to close holes and separate connected objects. The choice of kernel size and the number of iterations can significantly affect the results.
Noise Reduction:

Apply filters to reduce noise before performing subtraction. A median or Gaussian blur can help in reducing the effects of sensor noise or compression artifacts.
Segmentation Techniques:

Consider using advanced segmentation techniques like Graph Cuts or Watershed to refine the edges of the foreground objects and separate them from the background more cleanly.
Post-processing:

After applying background subtraction and thresholding, post-processing steps such as contour detection can help identify and fill holes within the foreground objects.
Parameter Optimization:

Implement a function that can automatically optimize the parameters by comparing the algorithm’s output to manual segmentation or by minimizing the noise in the binary mask.
"""