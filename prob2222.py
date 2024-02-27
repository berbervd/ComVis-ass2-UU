import pickle
import numpy as np
import cv2
from tqdm import trange
 
 
def init_cam_params(cam_no):
    # Load the calibration data from a pickle file
    #with open(f'data/cam{cam_no}/calibration_data_camera{cam_no}.pkl', 'rb') as f:
    with open(f'data/cam{cam_no}/calibration_data_camera{cam_no}.pkl', 'rb') as f:
        calibration_data = pickle.load(f)

    # Extract the intrinsic and extrinsic parameters
    mtx, dist = calibration_data['intrinsics']
    rvec, tvec = calibration_data['extrinsics']

    # Load the foreground image
    #fg = cv2.imread(f'data/cam{cam_no}/foreground_image.jpg')
    return rvec, tvec, mtx, dist, #fg

def look_up_table():
    """
     lookup table for mapping 3D voxel positions to 2D pixel positions
    on images from different cameras.
    """
    width, height, depth = 60, 60, 60  # Define the size of the 3D grid

    # Iterate through each of the 4 cameras to generate lookup tables
    for cam in range(1, 5):
        table = {}  # Initialize the lookup table for the current camera
        # Load camera parameters
        rvec, tvec, mtx, dist = init_cam_params(cam)

        # Iterate through each voxel within the 3D grid
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    # Define the voxel's 3D position, adjusting for visualization
                    voxel_pos = ((x - width / 2) * 40, (y - height / 2) * 40, -z * 40)

                    # Project the 3D voxel position to 2D pixel position
                    pixel_pts, _ = cv2.projectPoints(np.array([voxel_pos]), rvec, tvec, mtx, dist)
                    pixel_pts = pixel_pts.reshape(2)  # Reshape to get the x,y coordinates
                    
                    # Invert x and y to match image coordinates if needed
                    pixel_pts = pixel_pts[::-1]

                    # Store the mapping in the lookup table
                    table[voxel_pos] = pixel_pts

        # Save the lookup table to a file
        save_path = f'data/cam{cam}/look_up_table.pickle'
        with open(save_path, "wb") as f:
            pickle.dump(table, f)

        print(f"Lookup table for Camera {cam} saved to {save_path}.")


 