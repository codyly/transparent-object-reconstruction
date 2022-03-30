import cv2
import numpy as np
import glob
from tqdm import tqdm
from tqdm import trange
import json
import sys

cnt = 0

direction = [ [0,1], [1,1], [1,0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]


tag = sys.argv[1]

for filename in tqdm(glob.glob('.\\'+tag+'\\*.ppm')):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(filename[:-4] + '.png', img)



start = 206
end   = 256

all_brightness = []

for pix in trange(start, end):
    # all_brightness.append([])
    output_brightness = []
    i = 0
    j = 1
    while True:
        filename = '.\\'+tag+'\\image_{:d}_{:d}.png'.format(i, j)

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite(filename[:-4] + '.png', img)
        brightness= img[256][pix]
        for d in direction:
            dx, dy = d[0], d[1]
            brightness = max(brightness, img[256 + dx][pix + dy])
        cnt += 1
        output_brightness.append(brightness)
        if j == 63:
            i = i + 1
            j = 0
        else:
            j = j + 1
        if i == 1 and j == 1:
            break

    # output_brightness = np.array(output_brightness)/255

    all_brightness.append(output_brightness)

all_brightness = np.array(all_brightness) / 255
# print(all_brightness[0][0], all_brightness[1][0])
jsons = {'data': all_brightness.tolist()}
with open('prac_edge_'+tag+'.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f, indent=4)
with open('..\\DAE\\prac_edge_'+tag+'.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f, indent=4)

# print(output_brightness.shape)
# print(output_brightness)
    

# cv2.imshow('pattern', pattern)
# cv2.waitKey()
