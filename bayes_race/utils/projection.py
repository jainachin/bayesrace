"""	Projection of a point on a line.
"""

__author__ = 'Achin Jain'
__email__ = 'achinj@seas.upenn.edu'


import numpy as np


def Projection(point, line):

	assert len(point)==1
	assert len(line)==2

	x = np.array(point[0])
	x1 = np.array(line[0])
	x2 = np.array(line[len(line)-1])

	dir1 = x2 - x1
	dir1 /= np.linalg.norm(dir1, 2)
	proj = x1 + dir1*np.dot(x - x1, dir1)

	dir2 = (proj-x1)
	dir3 = (proj-x2)

	# check if this point is on the line, otw return closest vertex
	if np.linalg.norm(dir2, 2)>0 and np.linalg.norm(dir3, 2)>0:
		dir2 /= np.linalg.norm(dir2)
		dir3 /= np.linalg.norm(dir3)
		is_on_line = np.linalg.norm(dir2-dir3, 2) > 1e-10
		if not is_on_line:
			if np.linalg.norm(x1-proj, 2) < np.linalg.norm(x2-proj, 2):
				proj = x1
			else:
				proj = x2
	dist = np.linalg.norm(x-proj, 2)
	return proj, dist