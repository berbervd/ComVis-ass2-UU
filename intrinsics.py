import cv2 as cv
import os 
import glob
import numpy as np 

from ComVis.ass1.offline import interpolate_corners, draw_chessboard_corners, click_coords

# TODO: cv.setmouseclicks werkt nog niet, komt door de functie click_coords die wordt aangeroepen daarin. 
# Deze functie gebruikt een variable buiten de functie en ik weet even niet hoe we dat moeten oplossen zonder alle code te kopieren 

def show_vid_and_save_frames(camera_number, save_frames=False, max_frames=5): 
    """
    displays the video (.avi file) from the four cameras and allows for saving of frames 
    specify the number of the camera, whether the frames can be saved and the max ammount of frames 
    -> delete the old frames first if you want new frames 
    pressing 's' while the video plays saves the current frame 
    pressing 'q' quits the video 
    """
    
    save_dir = f'data/cam{camera_number}/saved_frames'
    os.makedirs(save_dir, exist_ok=True)

    cap_cam1 = cv.VideoCapture(f'data/cam{camera_number}/intrinsics.avi')
    
    while True:
        ret, frame = cap_cam1.read()

        # quit if video is done 
        if not ret:
            print("Reached the end of the video or failed to read. Exiting...")
            break

        # show frame 
        cv.imshow('Video Playback', frame)

        key = cv.waitKey(1) & 0xFF
        
        # to quit, press q 
        if key == ord('q'):
            break
        
        # check how many frames already saved
        frames_taken = len(os.listdir(save_dir))
        
        # if allowed, save a frame by pressing s 
        if save_frames and key == ord('s') and frames_taken <= max_frames-1:
            frame_number = int(cap_cam1.get(cv.CAP_PROP_POS_FRAMES))
            filename = os.path.join(save_dir, f"camera{camera_number}_frame_{frame_number}.png")
            cv.imwrite(filename, frame)
            print(f"Frame {frame_number} from camera {camera_number} saved as {filename}.")
            
    cap_cam1.release()
    cv.destroyAllWindows()

# UNCOMMENT TO DISPLAY VIDEO AND SAVE FRAMES 
# if __name__ == '__main__':
#     show_vid_and_save_frames(1, save_frames=True)
    

# ==== getting camera intrinsics, code copied and slightly altered from ComVis assignment 1 ====

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# define size of chessboard (from checkerboard.xml)
checkerboard_width = 8
checkerboard_height = 6
checkerboard_square_size = 115 

# object points 
objp = np.zeros((checkerboard_width*checkerboard_height,3), np.float32)
objp[:,:2] = np.mgrid[0:checkerboard_width,0:checkerboard_height].T.reshape(-1,2)

objpoints = []  
imgpoints = []   

images = glob.glob('data/cam1/saved_frames/*.png')
print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    click_coordinates = [] 

    # find chessboard corners automatically 
    ret, corners = cv.findChessboardCorners(gray, (checkerboard_width,checkerboard_height), None)
    
    if ret:
        
        # if found, add image and object points and draw image
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        img = cv.drawChessboardCorners(img, (checkerboard_width,checkerboard_height), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
        
    else: 
        
        # if not found automatically, manually select corners
        print("Automatic detection failed. Please select corners manually.")
        cv.namedWindow('Image')
        cv.setMouseCallback('Image', click_coords)
        cv.imshow('Image', img)
        cv.waitKey(0)

        if len(click_coordinates) == 4:
            
            # interpolate all corners from the four outer corners 
            interpolated_corners = interpolate_corners(click_coordinates, checkerboard_width, checkerboard_height)
            objpoints.append(objp)
            imgpoints.append(interpolated_corners)
            # laat zien na het klikken van de 4 corners of het is gelukt >> aNOG NIET!! DUS NOG FF LATEN
            draw_chessboard_corners(img, interpolated_corners, checkerboard_width, checkerboard_height)
        else:
            print("Not enough points selected.")
            
cv.destroyAllWindows()



