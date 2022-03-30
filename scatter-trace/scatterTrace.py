import cv2
import numpy as np
import glob
from tqdm import tqdm

cnt = 0
st = np.zeros([128, 64]).astype(np.uint8)

direction = [ [0,1], [1,1], [1,0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

# for filename in tqdm(glob.glob('.\\st1\\*.png')):
#     # print(filename)
#     img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#     brightness= img[256][256]
#     for d in direction:
#         dx, dy = d[0], d[1]
#         brightness = max(brightness, img[256 + dx][256 + dy])
#     st[127-cnt//64][63-cnt%64] = brightness
#     cnt += 1

# cv2.imshow('st', st)
# cv2.waitKey()
# cv2.imwrite('scatterTrace.png', st)
# # t = st


i = 0
j = 1

while True:
    filename = '.\\st1\\image_{:d}_{:d}.png'.format(i, j)
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    brightness= img[256][256]
    for d in direction:
        dx, dy = d[0], d[1]
        brightness = max(brightness, img[256 + dx][256 + dy])
    st[127-cnt//64][63-cnt%64] = brightness
    cnt += 1
    if j == 63:
        i = i + 1
        j = 0
    else:
        j = j + 1
    if i == 128 and j == 1:
        break
cv2.imshow('st', st)
cv2.waitKey()
cv2.imwrite('scatterTrace.png', st)