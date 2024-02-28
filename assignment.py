import glm
import random
import numpy as np
import pickle
import cv2
from tqdm import trange
from tqdm import tqdm

block_size = 1.0  

def get_calib_data(filename):
    with open(filename, 'rb') as f:
        calib_data = pickle.load(f)
    return calib_data

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):

    with open('data/voxel_model.pkl', 'rb') as file:
        data = pickle.load(file)
        colors = [i for i in range(len(data))]
                
    return data, colors

"""
Retrieving the camera positions
from teams hint 
>> TOEGEVOEGD: *3 want anders te veel op de grid. Nu lijkt het ok?
"""
def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    cam_positions = []
    for i_camera in range(1, 5):   
        calib_data = get_calib_data(f'data/cam{i_camera}/calibration_data_camera{i_camera}.pkl')
        tvec = calib_data['extrinsics'][1]  # transl matrix
        rvec = calib_data['extrinsics'][0] # rotation matrix

        rotation_matrix = cv2.Rodrigues(rvec)[0]
        position_vector = -np.matrix(rotation_matrix).T * np.matrix(tvec)
        cam_positions.append([position_vector[0]*3, -position_vector[2]*3, position_vector[1]*3]) # vermenigvuldigen zodat ze buiten de grid zijn > anders op de grid
        # misschien moeten we dat nog aanpassen? Weet niet echt of dit kan/mag
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():

    cam_rotations = []  
    for i_camera in range(1, 5): 
        calib_data = get_calib_data(f'data/cam{i_camera}/calibration_data_camera{i_camera}.pkl')
        rvec = calib_data['extrinsics'][0]  # rotation 
        rotation_matrix_cv, _ = cv2.Rodrigues(rvec)

        # Adjust the order of axes to match the visualization's coordinate system, if needed.
        # Here, we're swapping the y and z axes based on the specific visualization requirements.
        rotation_matrix_adjusted = rotation_matrix_cv[:, [0, 2, 1]]

        # Initialize a 4x4 identity matrix for the final rotation matrix.
        rot_matrix = np.eye(4)
        
        # Apply additional rotation to align the camera's view with the visualization's coordinate system.
        additional_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        rot_matrix[:3, :3] = np.dot(rotation_matrix_adjusted, additional_rotation)
        
        # Convert the numpy 4x4 matrix to a GLM matrix for use in graphics applications.
        cam_rotation_glm = glm.mat4(*rot_matrix.flatten())
        # Append the GLM rotation matrix to the list of camera rotations.
        cam_rotations.append(cam_rotation_glm)

    return cam_rotations
