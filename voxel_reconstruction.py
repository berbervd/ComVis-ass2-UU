import pickle
import numpy as np
import cv2 as cv
from pprint import pprint 


def init_cam_params(cam_no):
    with open(f'data/cam{cam_no}/calibration_data_camera{cam_no}.pkl', 'rb') as f:
        calibration_data = pickle.load(f)
    mtx, dist = calibration_data['intrinsics']
    rvec, tvec = calibration_data['extrinsics']
    return rvec, tvec, mtx, dist

# def look_up_table(width_voxelvol, height_voxelvol, depth_voxelvol, block_size=1):
#     width, height, depth = width_voxelvol, height_voxelvol, depth_voxelvol
#     table = {cam: [] for cam in range(1, 5)}
#     voxel_positions = np.mgrid[0:width, 0:height, 0:depth].T.reshape(-1,3)
#     voxel_positions = voxel_positions * block_size - np.array([width*block_size/4, 0, depth*block_size/4])
    
#     for cam in range(1, 5):
#         rvec, tvec, mtx, dist = init_cam_params(cam)
#         pixel_positions, _ = cv.projectPoints(voxel_positions, rvec, tvec, mtx, dist)
#         pixel_positions = pixel_positions.squeeze().astype(int)
#         table[cam] = pixel_positions, voxel_positions
    
#     save_path = f'data/lookuptable.pkl'
#     with open(save_path, "wb") as f:
#         pickle.dump(table, f)
    
#     print('Table done')
#     return table 

def look_up_table(width_voxelvol, height_voxelvol, depth_voxelvol, block_size=1):
    """
     lookup table for mapping 3D voxel positions to 2D pixel positions
    on images from different cameras.
    """
    width, height, depth = width_voxelvol, height_voxelvol, depth_voxelvol 

    # nested dict for each camera 
    table = {
        1: {},
        2: {},
        3: {},
        4: {},
    }

    # iterate through each voxel in the voxel volume 
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                
                # give voxel a label 
                voxel = (x,y,z)
                
                # iterate through each of the 4 views / cameras  
                for cam in range(1, 5):
                    
                    # load camera parameters
                    rvec, tvec, mtx, dist = init_cam_params(cam)
        
                    # define the voxel's 3D position
                    voxel_pos = (x*block_size - width/2, y*block_size, z*block_size - depth/2) 

                    # project the 3D voxel position to 2D pixel position
                    pixel_pos, _ = cv.projectPoints(np.array([voxel_pos]), rvec, tvec, mtx, dist)
                    
                    pixel_pos = np.round(pixel_pos).astype(int)
                    
                    # pixel_pos = pixel_pos.reshape(2) 
                    
                    # # Invert x and y to match image coordinates if needed
                    # pixel_pos = pixel_pos[::-1]
                    
                    # postion of pixel, if not work maybe try line above 
                    x_im, y_im = pixel_pos[0][0]
                    
                    pixel = (x_im, y_im)

                    # store info in the table 
                    table[cam][pixel] = voxel

    # Save the lookup table to a file
    save_path = f'data/lookuptable(lowres).pkl'
    with open(save_path, "wb") as f:
        pickle.dump(table, f)
    
    print('Table done')
    return table 

def load_lookup_table(pickle_file_path):

    with open(pickle_file_path, 'rb') as file:
        lookuptable = pickle.load(file)
    print('Lookup table loaded successfully.')
    return lookuptable


def reconstruct_voxel_model(table, background_masks):
    table_visible = {cam: set() for cam in range(1, 5)}
    
    for cam in range(1, 5):
        mask = background_masks[cam-1]
        
        # Placeholder for storing voxels corresponding to white pixels
        voxels = []

        # Iterate through each pixel in the image
        for y in range(mask.shape[0]):  # image height
            for x in range(mask.shape[1]):  # image width
                if mask[y, x] == 255:  # Check if the pixel is white
                    # print(table[cam])
                    
                    # Check if this pixel has a corresponding voxel
                    if (x, y) in table[cam]:
                        # Add the corresponding voxel to the list
                        voxels.append(table[cam][(x, y)])
        
        table_visible[cam] = voxels
    
    # intersectoin to find shared voxels 
    shared_voxels = set(table_visible[1])
    for cam in range(2, 5):
        shared_voxels &= set(table_visible[cam])
    
    shared_voxels = list(shared_voxels)
    print('Voxels visible in all views:', shared_voxels)
    
    return shared_voxels

background_masks = []
for cam in range(1, 5):
    mask = cv.imread(f'data/cam{cam}/foreground_image.jpg', cv.IMREAD_GRAYSCALE)
    background_masks.append(mask)

lookuptable_lowres = look_up_table(128, 64, 128, block_size=1)

# lookuptable = load_lookup_table('data/lookuptable.pkl')
# pprint(lookuptable_lowres)

voxel_model = reconstruct_voxel_model(lookuptable_lowres, background_masks)

save_path = f'data/voxel_model.pkl'
with open(save_path, "wb") as f:
    pickle.dump(voxel_model, f)