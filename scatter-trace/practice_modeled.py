import cv2
import numpy as np
import json


jstr = None
with open('./practice.json', 'r', encoding='utf-8') as f:
    jstr = json.load(f)

pattern = np.array(jstr['data']).reshape([64, 128])

scatterTrace = cv2.imread('./scatterTrace.png', cv2.IMREAD_GRAYSCALE) / 255


output_brightness = np.zeros([64,]).astype(np.float32)

for i in range(64):
    tmp = 0
    for j in range(128):
        tmp += pattern[i, j] * scatterTrace[127- j, 63-i]
        # print(tmp.shape)
    output_brightness[i] = tmp
        # break

# tmp = 0
# for i in range(128):
#     tmp += pattern[16, i, 32] * scatterTrace[i, 32]

jsons = {'data': output_brightness.tolist()}
with open('prac_pred_st.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f)
with open('..\\DAE\\prac_pred_st.json', 'w', encoding='utf-8') as f:
    json.dump(jsons, f)

print(output_brightness.shape)

print(output_brightness)