import pandas as pd
import numpy as np
import glob
import tifffile

csv_path = "0.21_0.26_projection.csv"
Tiffs_folder_path = "anisotropy_images/*.tiff"
csv_df = pd.read_csv(csv_path, sep = ";")

anisotropy_threshold = 0.2

# Load Position and Vectors from csv file

positions = csv_df[["PosX", "PosY", "PosZ"]].to_numpy(float)
vectors = csv_df[["VectX", "VectY", "VectZ"]].to_numpy(float)


tiff_files = glob.glob("anisotropy_images/*.tiff")
tiff_files_stack = np.stack([tifffile.imread(f) for f in files], axis = -1)
stack_y, stack_x, stack_z =  tiff_files_stack.shape

def mm_to_pixel(mm_input, total_pixels):
    pixel_size = (mm_input.max() -mm_input.min()) / (total_pixels - 1)
    return np.round((mm - mm.min()) /pixel_size).astype(int)

position_x_to_pixel = mm_to_pixel(positions[:, 0], stack_x)

position_y_to_pixel = mm_to_pixel(positions[:, 1], stack_y)

position_z_to_pixel = mm_to_pixel(positions[:, 2], stack_z)


anisotropy_values = stack[position_y_to_pixel, position_x_to_pixel, position_z_to_pixel]

points_to_keep = anisotropy_values >= anisotropy_threshold

vectors = vectors[points_to_keep]



lengths = np.linalg.norm(vectors, axis = 1, keepdims = True)
vectors = vectors[lengths[:, 0] > 0]
lengths = lengths[lengths[:, 0] > 0]
vectors = vectors / lengths

print("vectors kept: ", len(vectors))

resulting_matrix = vectors.T @ vectors

eigen_values, eigen_vectors = np.linalg.eigh(resulting_matrix)

principal_direction = eigen_vectors[:, -1]

angle_from_x_axis = np.degrees(np.arccos(abs(principal_direction[0])))

angle_from_y_axis = np.degrees(np.arccos(abs(principal_direction[1])))

angle_from_z_axis = np.degrees(np.arccos(abs(principal_direction[2])))


print("Dominant direction:", np.round(principal_direction, 3))
print("Angle from Z-axis:", round(angle_from_z_axis, 1), "deg")
print("Angle from X-axis:", round(angle_from_x_axis, 1), "deg")
print("Angle from Y-axis:", round(angle_from_y_axis, 1), "deg")