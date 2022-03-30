import numpy as np
import cv2 as cv
import json
from math import cos, sin, pi, tan, exp
from random import choice
from copy import deepcopy

class dataset:
	def __init__(self, dataset_size, datafile_format):
		self.size = dataset_size
		self.index = -1
		self.data = None
		self.file_format = datafile_format

	def __call__(self):
		while True:
			self.index = (self.index + 1) % self.size
			filename = self.file_format.format(self.index + 1)
			res = yield cv.imread(filename, cv.IMREAD_GRAYSCALE)

	def random(self):
		arr = np.linspace(0, self.size - 1, self.size)
		while True:
			self.index = int(choice(arr))
			filename = self.file_format.format(self.index + 1)
			res = yield cv.imread(filename, cv.IMREAD_GRAYSCALE)

class sart:
	def __init__(self,
				data_generator,
				volume_a = 128,
				kb_threshold = 0.5,
				iter_lambda = 0.01,
				iter_times = 600,
				initial_alpha = 0):
		self.va = volume_a
		self.volume = initial_alpha * np.ones([volume_a, volume_a, volume_a])
		self.dgen = data_generator
		self.kb_threshold = kb_threshold
		self.kb_arr = []
		self.gen_kbl()
		self.iter_lambda = iter_lambda
		self.iter_times = iter_times
		self.error_list = []
		self.calculate()

	def kaiser_bassel_function(self, distance):
		if distance >= self.kb_threshold:
			return 0;
		else:
			return (498*cos(2*pi*distance) + 99*cos(4*pi*distance)
				+ cos(6*pi*distance) + 402) / 1000.0

	def next_data(self):
		return next(self.dgen.random())

	def gen_kbl(self):
		self.kb_arr = []
		dist_map = np.linspace(0, self.kb_threshold, int(self.kb_threshold/0.05 + 1))
		for item in dist_map:
			self.kb_arr.append(self.kaiser_bassel_function(item))

	def forward_projection(self, angle, bg_color):
		dirty_cols = []
		dirty_weights = []
		gen = []
		center_shift = 0.5 * (self.va - 1)
		for ray in range(self.va):
			sum_weights = np.zeros([1,self.va])
			alpha_weights = np.zeros([1,self.va])
			dirty_cols.append([])
			dirty_weights.append([])
			
			for row in range(self.va):
				weights = []
				walphas = []
				dirty_cols[ray].append([])
				x = row-center_shift
				for col in range(self.va):					
					y = col-center_shift
					s_col = sin(angle)*x + cos(angle)*y + center_shift
					
					if s_col < 0 or s_col >= self.va:
						pass
					elif abs(s_col-ray) <= self.kb_threshold:
						dirty_cols[ray][row].append(col)
						weights.append([self.kb_arr[int(abs(s_col-ray)/0.05)]*np.ones(self.va)])
						dirty_weights[ray].append([self.kb_arr[int(abs(s_col-ray)/0.05)]*np.ones(self.va)])
						walphas.append([self.volume[row][col]*self.kb_arr[int(abs(s_col-ray)/0.05)]])
				if len(weights) > 0:
					sum_weights += np.sum(weights, axis=0)
					alpha_weights += np.sum(walphas, axis=0)
			dat = alpha_weights / (sum_weights+np.ones(sum_weights.shape)*1e-8)
			gen.append(dat)
		gen_sample = np.array(gen).reshape([self.va, self.va])
		return dirty_cols, dirty_weights, gen_sample

	def back_projection(self, angle, dirty_cols, dirty_weights, delta_A):
		tmp_volume = deepcopy(self.volume)
		for ray in range(self.va):
			tmp = []
			weights = np.sum(dirty_weights[ray], axis=0)
			if np.array(dirty_weights[ray]).shape[0] != 0:
				for w_list in dirty_weights[ray]:
					tmp.append(np.multiply(w_list, delta_A[ray].reshape([1,self.va]))) 	
				wdeltas = np.sum(tmp, axis=0)
				correctness = (self.iter_lambda * wdeltas / (weights+np.ones(weights.shape)*1e-8)).reshape(self.va)
				for row in range(self.va):
					for col in dirty_cols[ray][row]:
						self.volume[row][col] = tmp_volume[row][col] - correctness

	def save_log(self):
		with open('./log/log.json', 'w+', encoding='utf-8') as fs:
			json.dump(self.volume.tolist(), fs)
		with open('./log/errorlog.json', 'w+', encoding='utf-8') as fs:
			json.dump(self.error_list, fs)

	def calculate(self):
		for t in range(self.iter_times):
			print("iteration: {:>03d}".format(t))
			sample = np.transpose(self.next_data())
			assert(self.va == sample.shape[0])

			sample_id = self.dgen.index - 1
			angle = 2*pi*sample_id/self.dgen.size

			bg_color = 255.0
			sample = np.array(sample + np.ones(sample.shape)) / bg_color
			dirty_cols, dirty_weights, gen_sample = self.forward_projection(angle, bg_color)
			
			log_A = np.log(sample)
			log_AA = - np.array(gen_sample)
			delta_A = log_A - log_AA
			print("error: {:.2f}".format(np.sum(delta_A)))
			self.error_list.append(abs(np.sum(delta_A)))
			self.back_projection(angle, dirty_cols, dirty_weights,  delta_A)

			if t%10 == 0:
				self.save_log()
