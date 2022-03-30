import numpy as np
import json
import matplotlib.pyplot as plt
from mayavi import mlab 
from skimage import measure

def read_json(filename):
	with open(filename, 'r', encoding='utf-8') as fs:
		return json.load(fs)

def normalization(data):
	maxnum = np.max(data)
	print(maxnum)
	minnum = np.min(data)
	dist = maxnum-minnum;
	data = (data - minnum) / dist
	return data

class constructor:

	def __init__(self, jsonlog):
		self.data = np.array(read_json(jsonlog))
		self.data = normalization(self.data)
		self.jsonlog = jsonlog
		print(np.array(self.data).shape)
		verts, faces, norm, val = measure.marching_cubes_lewiner(self.data, 0.5)
		mlab.triangular_mesh([vert[0] for vert in verts],[vert[1] for vert in verts],[vert[2] for vert in verts],faces) 
		mlab.show() 

	def draw_error_curve(self, errorfile):
		self.error = read_json(errorfile)
		x = np.linspace(0, len(self.error)-1, len(self.error))
		y = self.error
		plt.figure(figsize=(6,4))
		plt.title('absolute errors of lambda=0.05')
		plt.plot(x,y,color='red',linewidth=1)
		plt.xlabel('times')
		plt.ylabel('error')
		plt.show()