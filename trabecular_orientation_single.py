"""
========================================================================
Trabecular bone: dominant orientation of the struts
========================================================================
Physical idea
-------------
Inside a femoral head the trabeculae (tiny bony struts) line up along the
main lines of mechanical stress, forming the diagonal / arch-like pattern
you see in a CT slice. Each row of the CSV gives the LOCAL orientation of a
strut at one sampled point. This script collapses that whole field of local
vectors into ONE dominant direction, then measures how that direction sits
relative to the X / Y / Z axes.

Inputs
------
1. CSV  : one row per point, with a position (PosX/Y/Z, in micrometres) and
          a local orientation vector (VectX/Y/Z).
2. TIFFs: a stack of images forming a 3D volume. Each voxel's brightness is
          the local "anisotropy" -- how strongly directional the structure
          is there (~0 = blobby / unreliable, higher = strongly aligned).
          Used only to discard unreliable points before averaging.

Output
------
The dominant direction vector plus its angles to the axes and planes.

Requires:  numpy, pandas, tifffile   (matplotlib only for the optional plot)
    pip install numpy pandas tifffile matplotlib
========================================================================
"""

import glob
import numpy as np
import pandas as pd
import tifffile

# ----------------------------------------------------------------------
# CONFIG -- point these at your data
# ----------------------------------------------------------------------
CSV_PATH        = "0.21_0.26_projection.csv"  # the per-point vectors
TIFF_GLOB       = "*.tiff"                     # one TIFF per z-slice (or one multi-page tiff)
ANISO_THRESHOLD = 0.2                          # keep points with anisotropy >= this
SHOW_PLOT       = False                        # True -> draw the 3D arrow + point cloud


# ----------------------------------------------------------------------
# 1. Load the per-point orientation vectors
# ----------------------------------------------------------------------
# pos = physical location of each sample point (micrometres).
# vec = the raw local orientation vector at that point (not yet unit length).
df  = pd.read_csv(CSV_PATH)
pos = df[["PosX", "PosY", "PosZ"]].to_numpy(dtype=float)
vec = df[["VectX", "VectY", "VectZ"]].to_numpy(dtype=float)


# ----------------------------------------------------------------------
# 2. Load the anisotropy volume from the TIFF stack
# ----------------------------------------------------------------------
# Each TIFF is one Z-slice. Stacking them along the 3rd axis builds a volume
# of shape (Ny, Nx, Nz) -- i.e. (rows=Y, cols=X, slices=Z). Keeping this exact
# axis order matters, because we index into it with (Y, X, Z) further down.
files = sorted(glob.glob(TIFF_GLOB))
if not files:
    raise FileNotFoundError(f"No TIFFs matched {TIFF_GLOB!r}")

if len(files) == 1:
    # A single file might already be a multi-page (3D) stack.
    stack = tifffile.imread(files[0]).astype(float)
    if stack.ndim == 3:                       # tifffile gives (Nz, Ny, Nx) ...
        stack = np.moveaxis(stack, 0, -1)     # ... so move Z to the end -> (Ny, Nx, Nz)
    else:                                     # a single 2D image
        stack = stack[:, :, None]
else:
    stack = np.stack([tifffile.imread(f).astype(float) for f in files], axis=-1)

Ny, Nx, Nz = stack.shape


# ----------------------------------------------------------------------
# 3. Convert physical (um) coordinates -> integer voxel indices
# ----------------------------------------------------------------------
# We need a voxel index for each point so we can look up its anisotropy.
# pixel_size is derived from the X span, ASSUMING the voxels are isotropic
# (equal spacing in X, Y, Z). If your X/Y/Z spans differ, this is the line
# to revisit. Note: 0-indexed here (Python), unlike the +1 in MATLAB.
pixel_size = (pos[:, 0].max() - pos[:, 0].min()) / (Nx - 1)
xi = np.round((pos[:, 0] - pos[:, 0].min()) / pixel_size).astype(int)
yi = np.round((pos[:, 1] - pos[:, 1].min()) / pixel_size).astype(int)
zi = np.round((pos[:, 2] - pos[:, 2].min()) / pixel_size).astype(int)

# Clamp any rounding that fell outside the image bounds.
xi = np.clip(xi, 0, Nx - 1)
yi = np.clip(yi, 0, Ny - 1)
zi = np.clip(zi, 0, Nz - 1)


# ----------------------------------------------------------------------
# 4. Keep only the well-aligned (high-anisotropy) points
# ----------------------------------------------------------------------
# Look up each point's anisotropy in the volume. Indexing order is (Y, X, Z)
# to match how the stack was built. Points below the threshold sit in
# blobby / poorly-defined regions where the orientation is unreliable, so we
# drop them before computing the average direction.
aniso = stack[yi, xi, zi]
keep  = aniso >= ANISO_THRESHOLD
vec   = vec[keep]
print(f"Kept {keep.sum()} / {keep.size} points (anisotropy >= {ANISO_THRESHOLD})")


# ----------------------------------------------------------------------
# 5. Normalize every vector to unit length
# ----------------------------------------------------------------------
# We care about DIRECTION, not magnitude, so make each vector unit length.
# The `where=norms>0` guard avoids dividing by zero for any all-zero rows
# (those simply become the zero vector instead of NaN).
norms = np.linalg.norm(vec, axis=1, keepdims=True)
v = np.divide(vec, norms, out=np.zeros_like(vec), where=norms > 0)


# ----------------------------------------------------------------------
# 6. Find the single dominant direction (orientation tensor)
# ----------------------------------------------------------------------
# Trabecular vectors are AXIAL: +v and -v mean the same physical direction,
# so their signs are essentially random. A plain averaging or centered PCA
# can cancel out or point the wrong way on such data.
#
# The correct, standard tool is the ORIENTATION TENSOR:
#       T = sum over points of ( v * v^T )      (a 3x3 matrix, no centering)
# Squaring via the outer product removes the sign ambiguity. The eigenvector
# of T with the LARGEST eigenvalue is the mean axis -- our dominant direction.
# The eigenvalues describe how tightly the vectors cluster around it.
T = v.T @ v
evals, evecs = np.linalg.eigh(T)     # eigh returns eigenvalues in ascending order
d = evecs[:, -1]                     # last column = largest eigenvalue = dominant axis
d = d / np.linalg.norm(d)
if d[2] < 0:                         # sign is arbitrary; flip to point +z for consistency
    d = -d

# Fraction of the total "spread" captured by the dominant axis:
# near 1.0 = strongly aligned struts, near 1/3 = nearly random.
alignment = evals[-1] / evals.sum()


# ----------------------------------------------------------------------
# 7. Turn the direction vector into interpretable angles
# ----------------------------------------------------------------------
# `d` is a unit vector [dx, dy, dz]. abs() is used wherever the sign of the
# axis shouldn't matter (an axis and its opposite are the same orientation).
#
#   angle_from_z  : tilt away from the vertical (Z) axis      = acos(|dz|)
#   elev_above_xy : how far the vector rises above the XY plane = 90 - angle_from_z
#   azimuth_xy    : compass heading within the XY plane (from +X), 0-360
#                   -- this is the ImageJ "Directionality"-style number
#   angle_from_x  : angle to the X axis                        = acos(|dx|)
#   angle_from_y  : angle to the Y axis                        = acos(|dy|)
angle_from_z  = np.degrees(np.arccos(abs(d[2])))
elev_above_xy = 90.0 - angle_from_z
azimuth_xy    = np.degrees(np.arctan2(d[1], d[0])) % 360
angle_from_x  = np.degrees(np.arccos(abs(d[0])))
angle_from_y  = np.degrees(np.arccos(abs(d[1])))
closest_axis  = "XYZ"[int(np.argmax(np.abs(d)))]   # which axis d points most along

# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
print("\n--- Dominant orientation ---")
print(f"Direction (unit vector): [{d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}]")
print(f"Closest to axis:         {closest_axis}")
print(f"Alignment strength:      {alignment*100:.1f}%")
print(f"Angle from z-axis:       {angle_from_z:6.2f} deg")
print(f"Elevation above x-y:     {elev_above_xy:6.2f} deg")
print(f"Azimuth in x-y plane:    {azimuth_xy:6.2f} deg  (from +x)")
print(f"Angle from x-axis:       {angle_from_x:6.2f} deg")
print(f"Angle from y-axis:       {angle_from_y:6.2f} deg")


# ----------------------------------------------------------------------
# 8. Optional: 3D visual sanity check (arrow + sampled point cloud)
# ----------------------------------------------------------------------
if SHOW_PLOT:
    import matplotlib.pyplot as plt

    xk, yk, zk = xi[keep], yi[keep], zi[keep]
    origin = np.array([xk.mean(), yk.mean(), zk.mean()])
    arrow  = d * 100.0                       # scaled up just so it's visible

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    step = slice(None, None, 20)             # plot every 20th point to stay light
    ax.scatter(xk[step], yk[step], zk[step], s=3, alpha=0.1, c="tab:blue")
    ax.quiver(*origin, *arrow, color="r", linewidth=3)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Dominant orientation (red) + points")
    plt.tight_layout()
    plt.show()
