import glm
import random
import numpy as np
import pickle
import cv2

block_size = 1.0

def get_calib_data(filename):
    with open(filename, 'rb') as f:
        calib_data = pickle.load(f)
    return calib_data

#xx = get_calib_data('data/cam1/calibration_data_camera1.pkl')
#print(xx)

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
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
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


"""
Ik snap niet of dit goed is 
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
"""
def get_cam_rotation_matrices():
    cam_rotations = []
    for i_camera in range(1, 5):   
        calib_data = get_calib_data(f'data/cam{i_camera}/calibration_data_camera{i_camera}.pkl')
        rvec = calib_data['extrinsics'][0]  # Rotation 
        
        # Convert rotation vector to rotation matrix
        rotation_matrix_cv, _ = cv2.Rodrigues(rvec)
        
        # Convert the numpy 3x3 rotation matrix to a format suitable for your application, e.g., glm for OpenGL
        rotation_matrix_glm = glm.mat4(1)  # Initialize a 4x4 identity matrix
        for i in range(3):  # Copy the 3x3 rotation matrix to the 4x4 matrix
            for j in range(3):
                rotation_matrix_glm[i][j] = rotation_matrix_cv[i, j]
        
        cam_rotations.append(rotation_matrix_glm)
        
    return cam_rotations

"""
Hieronder ff de 'oude' functies om te kunnen kijken naar de default 

"""

def get_cam_rotation_matrices_DUMMY():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations


def get_cam_positions_DUMMY():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]] 
