import pandas as pd
import numpy as np

csv_path = "0.21_0.26_projection.csv"
Tiffs_folder_path = "anisotropy_images/*.tiff"
csv_df = pd.read_csv(csv_path, sep = ";")


# Load Position and Vectors from csv file

positions = csv_df[["PosX", "PosY", "PosZ"]].to_numpy(float)
vectors = csv_df[["VectX", "VectY", "VectZ"]].to_numpy(float)


lengths = np.linalg.norm(vectors, axis = 1, keepdims = True)
vectors = vectors[lengths[:, 0] > 0]
lengths = lengths[lengths[:, 0] > 0]
vectors = vectors / lengths

print("vectors kept: ", len(vectors))

resulting_matrix = vectors.T @ vectors

eigen_values, eigen_vectors = np.linalg.eigh(resulting_matrix)

principal_direction = eigen_vectors[:, -1]

angle_from_x_axis = np.degrees(np.arccos(abs(d[0])))

angle_from_y_axis = np.degrees(np.arccos(abs(d[1])))

angle_from_z_axis = np.degrees(np.arccos(abs(d[2])))


print("Dominant direction:", np.round(d, 3))
print("Angle from Z-axis:", round(angle_from_z_axis, 1), "deg")
print("Angle from X-axis:", round(angle_from_x_axis, 1), "deg")
print("Angle from Y-axis:", round(angle_from_y_axis, 1), "deg")