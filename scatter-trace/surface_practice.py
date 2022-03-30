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


jdata = {}

all_brightness = np.zeros([20, 100, 64]).astype(np.float32)

cnt = 0

i = 0
j = 1
bmap = np.zeros([4096, 4096]).astype(np.uint8)
while True:
    filename = '.\\'+tag+'\\image_{:d}_{:d}.png'.format(i, j)

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    r = int(cnt/8)
    c = cnt % 8
    # bmap[r*512:(r+1)*512, c*512:(c+1)*512] = img
    # all_brightness[:,:,cnt] = img[300:350, 250:350]
    for p in range(100):
        all_brightness[:,p,cnt] = np.mean(img[300:320, 225+p])
    # cv2.imshow('test', img[300:320, 225:325])
    # cv2.waitKey()
    # break
    # roi = img[128:280, 180:256]
    # r = np.zeros([152]).astype(np.float32)
    # for rr in range(152):
    #     cnt_pix = 0
    #     for pr in range(76):
    #         if roi[rr][pr] > 0:
    #             break
    #         cnt_pix += 1
    #     r[rr] = (76-cnt_pix) * 0.004511 + 0.2 
    # with open('gt.json', 'w', encoding='utf-8') as f:
    #     json.dump({'data': r.tolist()}, f)
    # break
    cnt += 1
    if j == 63:
        i = i + 1 
        j = 0
    else:
        j = j + 1
    if i == 1 and j == 1:
        break
# cv2.imwrite(tag+'.png', bmap)
# cv2.waitKey()
# sys.exit(0)
# for r in trange(128, 384):
#     start = 206
#     end   = 256

#     row_brightness = []

#     for pix in range(start, end):
#         # all_brightness.append([])
#         output_brightness = []
#         i = 0
#         j = 1
#         while True:
#             filename = '.\\'+tag+'\\image_{:d}_{:d}.png'.format(i, j)

#             img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#             # cv2.imwrite(filename[:-4] + '.png', img)
#             brightness= img[r][pix]
#             # for d in direction:
#             #     dx, dy = d[0], d[1]
#             #     brightness = max(brightness, img[r + dx][pix + dy])
#             cnt += 1
#             output_brightness.append(brightness)
#             if j == 63:
#                 i = i + 1 
#                 j = 0
#             else:
#                 j = j + 1
#             if i == 1 and j == 1:
#                 break

#         # output_brightness = np.array(output_brightness)/255

#         row_brightness.append(output_brightness)

#     all_brightness.append(row_brightness)

# all_brightness = np.array(all_brightness) / 255
# print(all_brightness[0][0], all_brightness[1][0])
print(all_brightness.shape)
all_brightness  = all_brightness / 255
jsons = {'data': all_brightness.tolist()}
with open('prac_surface_'+tag+'.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f, indent=4)
with open('..\\DAE\\prac_surface_'+tag+'.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f, indent=4)

# print(output_brightness.shape)
# print(output_brightness)
    

# cv2.imshow('pattern', pattern)
# cv2.waitKey()
