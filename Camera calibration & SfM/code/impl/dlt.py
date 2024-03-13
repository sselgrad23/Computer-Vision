import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  for i in range(num_corrs):
    constraint_matrix[2 * i, :] = np.array([0,0,0,0,
      -points3D[i,0], -points3D[i,1], -points3D[i,2], -1, 
      points2D[i,1]*points3D[i,0], points2D[i,1]*points3D[i,1],points2D[i,1]*points3D[i,2], points2D[i,1]])
    
    constraint_matrix[2 * i + 1, :] = np.array([points3D[i,0], points3D[i,1], points3D[i,2], 1,
      0, 0, 0, 0,
      -points2D[i,0]*points3D[i,0], -points2D[i,0]*points3D[i,1], -points2D[i,0]*points3D[i,2], -points2D[i,0]])

  return constraint_matrix