import cv2 as cv
import glob
import numpy as np
import pickle 
import os 

# HOW TO USE: running this file will show you the checkerboard images with the found corners. These are used to calculate the intrinsics (camera matrix and dist coeffs)
# The last image that pops up is the image to calculate the extrinsics. You need to manually annotate the corners here. To do that, drag te front window a bit to the side to show the magnification window (for better accuracy)
# TODO: the final axes of the world-origin are not correct. From my testing, it appears that there is a lot of variance based on how i annotate the corners to calculate the extrinsics, so the problem should lie there 
# the manual annotation is either not accurate enough or there is something wrong in the code 

class ChessboardCalibration:
    def __init__(self, corn_vert=8, corn_horiz=6):
        self.corn_vert = corn_vert
        self.corn_horiz = corn_horiz
        self.objp = np.zeros((corn_vert*corn_horiz, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:corn_vert, 0:corn_horiz].T.reshape(-1, 2)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        self.images = None 
        self.training_points = {"objpoints": [], "imgpoints": []} 
        self.image_shape = None
        self.gray = None 
        self.intrinsics = None 
        self.images_intrinsics = None 
        self.images_extrinsics = None 
        self.image_extrinsics_corners = None 

    def magnify_area(self, img, point, magnification=6, size=100):
        """
        shows area around cursor  in a different window with a certain magnification 
        """
        x, y = point
        height, width = img.shape[:2]
        
        # calculate the Region of Interest
        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(width, x + size // 2), min(height, y + size // 2)
        
        # crop and magnify the ROI
        roi = img[y1:y2, x1:x2]
        magnified_roi = cv.resize(roi, None, fx=magnification, fy=magnification, interpolation=cv.INTER_LINEAR)
        
        # calculate the center of the magnified area and draw circle 
        center_of_magnified = (magnified_roi.shape[1] // 2, magnified_roi.shape[0] // 2)
        cv.circle(magnified_roi, center_of_magnified, 4, (0, 0, 255), -1)
        
        cv.imshow('Magnified', magnified_roi)


    def click_coords(self, event, x, y, flags, param):
        """
        registers clikcing of coordinates and whether magnification is needed, click once to magnify 
        """
        if event == cv.EVENT_LBUTTONDOWN and len(self.click_coordinates) < 4:
            self.click_coordinates.append((x, y))
            self.draw_click_point(self.img, (x, y))
        elif event == cv.EVENT_MOUSEMOVE:
            # When the mouse is moved, show the magnified area around the cursor 
            self.magnify_area(self.img, (x, y))
            


    def draw_click_point(self, img, point):
        
        """
        draws crosshairs on clicked points 
        """
        
        #  draw crosshairs for precision
        cv.circle(img, point, 2, (0, 255, 0), -1)
        cv.line(img, (point[0] - 20, point[1]), (point[0] + 20, point[1]), (0, 255, 0), 2)
        cv.line(img, (point[0], point[1] - 20), (point[0], point[1] + 20), (0, 255, 0), 2)
        cv.imshow('Image', img)
        cv.waitKey(1)  


    def draw_chessboard_corners(self, img, corner_points, fname):
        cv.drawChessboardCorners(img, (self.corn_vert, self.corn_horiz), corner_points, True)
        cv.imshow(fname, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

            
    def interpolate_corners(self, corner_points):
        topleft, topright, bottomright, bottomleft = self.click_coordinates
        all_points = []
        for i in range(self.corn_horiz):
            left = np.add(np.multiply(np.subtract(bottomleft, topleft), i / (self.corn_horiz - 1)), topleft)
            right = np.add(np.multiply(np.subtract(bottomright, topright), i / (self.corn_horiz - 1)), topright)
            for j in range(self.corn_vert):
                point = np.add(np.multiply(np.subtract(right, left), j / (self.corn_vert - 1)), left)
                all_points.append(point)
        return np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

    def process_images(self, save_corners=False, intrinsics=True):
        """ 
        Reads images (intrinsics images or extrinsics images) and proceeds to calculate the image points 
        """
        
        print('processing images...')
        
        # load images depending on what the activity is (intrinsics or extrinsics )
        if intrinsics: 
            images = self.images_intrinsics
        else: 
            images = self.images_extrinsics
        
        if not images: 
            print('No images found')
            return 
        
        for fname in images:
            self.img = cv.imread(fname)
            self.gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
            self.image_shape = self.gray.shape[::-1]
            self.click_coordinates = []

            ret, corners = cv.findChessboardCorners(self.gray, (self.corn_vert, self.corn_horiz), None)

            if ret:
                corners2 = cv.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.criteria)
                self.training_points['objpoints'].append(self.objp)
                self.training_points['imgpoints'].append(corners2)
                self.draw_chessboard_corners(self.img, corners2, fname)

                if save_corners: 
                    self.image_extrinsics_corners = corners2
                
            # set corners manually 
            else:
                print("Automatic detection failed. Please select corners manually.")
                
                cv.namedWindow('Image')
                cv.setMouseCallback('Image', self.click_coords)
                
                cv.imshow('Image',self.img)
                cv.waitKey(0)

                # only save the image and object points for the intrinsics phase 
                if intrinsics: 
                    if len(self.click_coordinates) == 4:
                        interpolated_corners = self.interpolate_corners(self.click_coordinates)
                        self.training_points['objpoints'].append(self.objp)
                        self.training_points["imgpoints"].append(interpolated_corners)
                        self.draw_chessboard_corners(self.img, interpolated_corners)
                    else:
                        print("Not enough points selected.")
                        
                if not intrinsics: 
                    if len(self.click_coordinates) == 4:
                        interpolated_corners = self.interpolate_corners(self.click_coordinates)
                        self.image_extrinsics_corners = interpolated_corners
                        self.draw_chessboard_corners(self.img, interpolated_corners)
                    else:
                        print("Not enough points selected.")
                    
            

    def obtain_intrinsics(self, intrinsics_image_directory):
        """
        obtains the intrinsics, given a folder with training images 
        saves the intrinsics internally
        """
        
        self.images_intrinsics = glob.glob(f'{intrinsics_image_directory}/*.png')
        
        self.process_images(self)
    
        object_points = self.training_points['objpoints']
        image_points = self.training_points['imgpoints']
        if not object_points or not image_points: 
            ('Either image points or object points empty')

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
                object_points, image_points, self.image_shape, None, None)
        
        self.intrinsics = camera_matrix, dist_coeffs
        
        print('camera matrix: ', camera_matrix)
        print('')
        print('dist coeff: ', dist_coeffs)
        print('')
    
    def obtain_extrinsics(self, extrinsics_image_directory): 
        # Make sure we've already obtained the intrinsic parameters
        if self.intrinsics is None:
            raise ValueError("Intrinsic parameters not available. Run obtain_intrinsics first.")
        
        # Load the test image
        self.img = cv.imread(extrinsics_image_directory)
        if self.img is None:
            raise FileNotFoundError("Test image not found.")
        
        # Convert to grayscale
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
        # Find the chessboard corners in the test image
        ret, corners = cv.findChessboardCorners(gray, (self.corn_vert, self.corn_horiz), None)
        
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
            
        if not ret:
            cv.namedWindow('Image')
            cv.setMouseCallback('Image', self.click_coords)
            
            cv.imshow('Image', self.img)
            cv.waitKey(0)
            
            if len(self.click_coordinates) == 4:
                corners = self.interpolate_corners(self.click_coordinates)
                self.draw_chessboard_corners(self.img, corners, 'extrinsics im')
            else:
                print("Not enough points selected.")
        
        
        self.image_extrinsics_corners = corners

        # Use the intrinsic parameters
        camera_matrix, dist_coeffs = self.intrinsics

        # Solve for rotation and translation vectors
        _, rvecs, tvecs, _ = cv.solvePnPRansac(self.objp, corners, camera_matrix, dist_coeffs)

        # Save the extrinsic parameters
        self.extrinsics = rvecs, tvecs

        return rvecs, tvecs
        
        
        
        
    def draw_axes(self, img, corners, rotation_vector, translation_vector, camera_matrix, dist_coeffs, axis_length=3):
        # Define the 3D points for the axes (X, Y, Z)
        axis = np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]).reshape(-1, 3)

        # Project the 3D points to the image plane
        axis_points, _ = cv.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        # Ensure the points are in the correct format (integer tuples)
        axis_points = np.int32(axis_points).reshape(-1, 2)

        # Draw the axes lines on the image. The first corner of the chessboard is used as the origin.
        origin = tuple(corners[0].ravel().astype(int))
        img = cv.line(img, origin, tuple(axis_points[0]), (0, 255, 0), 2)  # X-Axis in red
        img = cv.line(img, origin, tuple(axis_points[1]), (255, 0, 0), 2)  # Y-Axis in green
        img = cv.line(img, origin, tuple(axis_points[2]), (0, 0, 255), 2)  # Z-Axis in blue
        return img

    
    def show_image_with_axes(self, image_file, save_to=None):
        # Ensure that the intrinsics and extrinsics have been computed
        if self.intrinsics is None or self.extrinsics is None:
            raise ValueError("Must obtain intrinsics and extrinsics before drawing axes.")

        # Load the image
        img = cv.imread(image_file)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_file}")

        # Convert to grayscale and find chessboard corners
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        corners = self.image_extrinsics_corners
        
        # Draw the axes on the image
        camera_matrix, dist_coeffs = self.intrinsics
        rotation_vector, translation_vector = self.extrinsics
        img_with_axes = self.draw_axes(img, corners, rotation_vector, translation_vector, camera_matrix, dist_coeffs, axis_length=3)

        # Show the image
        cv.imshow('Image with Axes', img_with_axes)
        cv.waitKey(0)
        cv.destroyAllWindows()




# ---------------- for reading and saving frames -----------------

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
        
        

        if save_frames and key == ord('s'): 
            # Convert to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
            # if allowed, save a frame by pressing s
            ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
            
            if ret: 
                frame_number = int(cap_cam1.get(cv.CAP_PROP_POS_FRAMES))
                filename = os.path.join(save_dir, f"camera{camera_number}_frame_{frame_number}.png")
                cv.imwrite(filename, frame)
                print(f"Frame {frame_number} from camera {camera_number} saved as {filename}.")
            
    cap_cam1.release()
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    
    # UNCOMMENT THIS TO OBTAIN NEW FRAMES FOR CAMERA 1 
    # show_vid_and_save_frames(1, save_frames=True)

    # calibration for camera 1
    cc = ChessboardCalibration(corn_vert=8, corn_horiz=6)
    cc.obtain_intrinsics('data/cam1/saved_frames')
    
    extr_im = 'data/cam1/saved_frames_extrinsics/camera1_frame_33.png'
    cc.obtain_extrinsics(extr_im)
    cc.show_image_with_axes(extr_im)