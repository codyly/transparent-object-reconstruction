import cv2
import numpy as np
import glob
from tqdm import tqdm
import json

cnt = 0

direction = [ [0,1], [1,1], [1,0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]

output_brightness = []

local = 0

for filename in tqdm(glob.glob('.\\simple2\\*.ppm')):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filename[:-4] + '.png', img)

i = 0
j = 1
while True:
    filename = '.\\simple2\\image_{:d}_{:d}.png'.format(i, j)

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite(filename[:-4] + '.png', img)
    brightness= img[256][256]
    for d in direction:
        dx, dy = d[0], d[1]
        brightness = max(brightness, img[256 + dx][256 + dy])
    local += brightness
    cnt += 1
    # if cnt // 64 == 16 and cnt % 64 == 32:
    #     print(brightness/255)
    if cnt % 64 == 0:
        output_brightness.append(local)
        local = 0
    if j == 63:
        i = i + 1
        j = 0
    else:
        j = j + 1
    if i == 32 and j == 1:
        break

output_brightness = np.array(output_brightness)/255/2

jsons = {'data': output_brightness.tolist()}
with open('pred.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f)

print(output_brightness.shape)
print(output_brightness)
    

# cv2.imshow('pattern', pattern)
# cv2.waitKey()
