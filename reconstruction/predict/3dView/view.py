#方法一，利用关键字
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import numpy as np
import json
import math

#定义坐标轴
fig = plt.figure()

# a = np.linspace(0, 0.4, 2)
# x =  - a + 0.70
# # y =  - 0.2275 * a + 0.461

# plt.plot(a,x, 'gray')



# a = np.linspace(0, 0.4, 2)
# x =  - a + 0.4255
# # y =  - 0.2275 * a + 0.2305

# plt.plot(a,x, 'gray')


a = np.linspace(0, math.pi/2, 50)
x =  0.4*np.cos(a)
y =  0.4*np.sin(a)

plt.plot(x,y, 'gray')


a = np.linspace(0, math.pi/2, 50)
x =  0.5*np.cos(a)
y =  0.5*np.sin(a)

plt.plot(x,y, 'gray')



a = np.linspace(0, math.pi/2, 50)
x =  0.461*np.cos(a)
y =  0.461*np.sin(a)

plt.plot(x,y, 'red')

jdata = None
tag = sys.argv[1]

with open('..\\result_'+tag+'.json', 'r', encoding='utf-8') as f:
    jdata = json.load(f)


xd = []
yd = []

for i in range(50):
    key = '{:03d}'.format(i)
    coord = jdata[key]['coord']
    xd.append(coord[0])
    yd.append(coord[1])
plt.plot(xd, yd, 'bo')

plt.show()