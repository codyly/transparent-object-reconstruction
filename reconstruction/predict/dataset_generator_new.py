import cv2
import numpy as np 
import os
import helper
import math
import time
import json
import pandas as pd
import csv
from tqdm import tqdm
import sys

LGT_WIDTH = 64
LGT_HEIGHT = 128


class float3:
    def __init__(self,x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    def __add__(self, other):
        m = float3(self.x+other.x, self.y+other.y, self.z+other.z)
        return m
    def __sub__(self, other):
        m = float3(self.x-other.x, self.y-other.y, self.z-other.z)
        return m
    def __mul__(self, c):
        m = float3(self.x*c, self.y*c, self.z*c)
        return m
    def __str__(self):
        return '('+ str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

def make_float3a(a):
    return float3(a, a, a)

def make_float3(a, b, c):
    return float3(a, b, c)

def dot(a, b):
    return a.x*b.x + a.y * b.y + a.z*b.z

def normalize(x):
    length = math.sqrt(dot(x, x))
    x.x /= length
    x.y /= length
    x.z /= length
    return x


CAMERA = make_float3(5, 0, 0)
OBJECT = make_float3(0, 0, 0)
ATTENUATION_COEF = 1e-4


def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0

class ggxSurface:
    def __init__(self, coord = make_float3(0, 0, 0), norm = make_float3(1, 0, 2), ggx_ag = 0.01):
        self.coord = coord
        self.norm = normalize(norm)
        self.ggx_ag = ggx_ag

    def G(self, m, n, i, o):
        G1 = G2 = 0
        ag = self.ggx_ag
        if dot(i, m) / dot(i, n) > 0:
            G1 = 2.0 / (1.0 + math.sqrt(1 + ag * ag * (1.0/(dot(i,n) * dot(i,n)) - 1.0)))
        if dot(o, m) / dot(o, n) > 0:
            G2 = 2.0 / (1.0 + math.sqrt(1 + ag * ag * (1.0/(dot(o,n) * dot(o,n)) - 1.0)))
        return G1 * G2

    def refl_w(self, i, o):
        m = normalize(i + o)
        n = self.norm
        if dot(i,n)  == 0:
            return 0
        return dot(i,m) * self.G(m, n, i, o) / dot(i,n) / dot(n,m)


    def norm_d(self, i, o):
        m = normalize(i + o)
        n = self.norm
        ag = self.ggx_ag
        cos_theta = dot(m, n)
        ZERO = 1e-6
        tan_thera = math.sqrt(1-cos_theta*cos_theta) / (cos_theta + ZERO)
        if cos_theta <= 0:
            return 0
        else:
            return ag * ag / math.pi / (cos_theta ** 4) / ((ag**2 + tan_thera**2)**2)

class lightZone:
    def __init__(self, rt = make_float3(-4, 0, 2), lb = make_float3(4, 0, 6),
                brightness = 5.0):
        self.rt = rt
        self.lb = lb
        self.precise = (lb.x - rt.x) / LGT_HEIGHT
        self.res_x = LGT_HEIGHT
        self.res_z = LGT_WIDTH
        self.brightness = brightness

    def scatter_trace(self, surface = ggxSurface(), img = None):

        if img is None:
            img = np.zeros([self.res_x, self.res_z]).astype(np.uint8)

        for xi in range(self.res_x):
            for zi in range(self.res_z):
                lpos = make_float3(self.lb.x, self.lb.y, self.lb.z)
                lpos.x += (-self.precise) * xi
                lpos.z += (-self.precise) * zi
                # print(lpos)
                i = lpos - surface.coord
                o = CAMERA - surface.coord
                distance = (math.sqrt(dot(i, i)) + math.sqrt(dot(o, o)) )
                i = normalize(i)
                o = normalize(o)
                crement = surface.norm_d(i, o) * self.brightness * math.exp(-ATTENUATION_COEF*distance)
                tmp = img[xi][zi] + crement
                img[xi][zi] = min(255, tmp)
                if crement > 0:
                    # print(i, o)
                    pass
                # print(img[xi][zi])

        return img

def randomObjPos(center = make_float3a(0), radius = 2):
    theta = np.random.rand() * math.pi * 2
    r = np.random.rand() * radius
    return make_float3(r*math.cos(theta), 0, r*math.sin(theta)) + center

def randomObjNorm():
    theta = np.random.rand() * math.pi / 2
    return make_float3(math.cos(theta), 0, math.sin(theta))

def randomObjAg(l = 1e-5, u = 1e-2 * 2):
    return np.random.rand() * (u-l) + l

def cus_compair(s):
    num = len(s[np.where(s>5)])
    if num <= 0:
        return num
    else:
        return np.sum(s) / num 

def gen_record(rid = 0):
    
    lz = lightZone()
    img = lz.scatter_trace(ggxSurface(coord=randomObjPos(), norm=randomObjNorm(), ggx_ag = randomObjAg()))
    imgs = []
    imgs.append(img)
    for i in range(5):
        ind_lz = lightZone(brightness = np.random.rand() * .025)
        imgs.append(ind_lz.scatter_trace(ggxSurface(coord=randomObjPos(), norm=randomObjNorm(), ggx_ag = randomObjAg())))
    for i in range(6):
        img = img | imgs[i]
        cv2.imshow('img',img)
        cv2.waitKey(0)

    imgs = sorted(imgs, key=cus_compair, reverse=True)[0:5]

    label_img = imgs[0]
    for i in range(5):
        label_img = label_img | imgs[i]

    img = np.array(img)
    # cv2.imshow('lb',label_img)
    # cv2.waitKey(0)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.imshow('imgT',img.T)
    # cv2.waitKey(0)
    sys.exit(0)

    imgs = np.array(imgs)
    img = img.reshape([LGT_HEIGHT * LGT_WIDTH])
    label_img = label_img.reshape([LGT_HEIGHT * LGT_WIDTH])

    
    # imgs = imgs.reshape([5 * LGT_HEIGHT * LGT_WIDTH // 4])
    
    record = {'id':rid, 'input':img.tolist(), 'output': label_img.tolist()}
    # print(img.shape, label_img.shape)
    
    return record

def gen():
    for i in range(2):
        nid = np.random.randint(0, 251)
        rec = gen_record(nid)
        with open('./ds-prac/st_{:d}.json'.format(nid), 'w') as f:
            json.dump(rec, f)

def main():
    # ds = []
    np.random.seed(int(time.time()))
    # csvf = open('./ds-prac/dataset.csv', 'a')
    # writer = csv.writer(csvf)
    # writer.writerow(['filename'])
    for i in tqdm(range(250)):
        rec = gen_record(i)
        # with open('./ds-prac/st_{:d}.json'.format(i), 'w') as f:
            # json.dump(rec, f)
        # writer.writerow(['st_{:d}.json'.format(i)])
    # csvf.close()

if __name__ == '__main__':

    main()
    

