import os
import sys
import re
import math
import numpy as np
import json


f = open('surface_bottle_2.xyz', 'r', encoding='utf-8')

data = f.read()

f.close

s = re.findall("\d\.\d+",data)

d = [float(i) for i in s]

rnum = int(len(d) / 6)

lev = -1

f = open('smooth.xyz', 'w', encoding='utf-8')

levcnt = 0

levavg = []

levsum = 0

for i in range(rnum-1):
    flag = True
    sid = int(i * 6)
    clev = d[sid + 1]
    if clev != lev:
        lev = clev
        # flag = False
        levsum = 0
    elif clev != d[sid + 7]:
        levavg.append(levsum/levcnt)
        levcnt = 0
        levsum = 0
        # flag = False
    else:
        levcnt +=1
        levsum += math.sqrt(d[sid]**2 + d[sid+2]**2)

    # if d[sid] > 0.431:
    #     flag = False
    
    # if d[sid] < 0.1 and d[sid + 2] < 0.1:
    #     flag = False
    
    if flag:
        output = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(d[sid], d[sid + 1], d[sid+2], d[sid+3], d[sid+4],d[sid+5])
        f.write(output)


f.close()

# ----------------------------------------------------------------

for ii in range(1):
    f = open('smooth.xyz', 'r', encoding='utf-8')

    data = f.read()

    f.close

    s = re.findall("\d\.\d+",data)

    d = [float(i) for i in s]

    rnum = int(len(d) / 6)

    lev = -1

    new_lev_sum = 0

    new_lev_avg = []

    new_lev_cnt = 0

    new_lev_max = 0

    new_lev_maxs = []



    f = open('smooth.xyz', 'w', encoding='utf-8')

    levid = -1

    for i in range(rnum-1):
        flag = True
        sid = int(i * 6)
        clev = d[sid + 1]
        factor = 1
        if clev != lev:
            lev = clev
            levid += 1
            # flag = False
            new_lev_cnt = 1
            new_lev_max = 0
        # print(len(levavg), levid)
        new_r = 0

        if levid < len(levavg) and (math.sqrt(d[sid]**2 + d[sid+2]**2) - levavg[levid] > 0.1 / 2**ii):
            new_r = 0.5 * (math.sqrt(d[sid]**2 + d[sid+2]**2) + levavg[levid])
            if new_r < d[sid+2]:
                flag = False
            else:
                factor = math.sqrt(new_r**2 - d[sid+2]**2) / d[sid]
        
        elif levid < len(levavg) and math.sqrt(d[sid]**2 + d[sid+2]**2) < levavg[levid]:
            factor = math.sqrt(levavg[levid]**2 - d[sid+2]**2) / d[sid]
            new_r = levavg[levid]

        if flag and levid < len(levavg):
            new_lev_sum += new_r
            new_lev_cnt += 1
            new_lev_max = max(new_r, levavg[levid])

        if clev != d[sid + 7]:
            new_lev_avg.append(new_lev_sum/new_lev_cnt)
            new_lev_cnt = 0
            new_lev_sum = 0
            new_lev_maxs.append(new_lev_max)

        if flag:
            output = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(d[sid] * factor, d[sid + 1], d[sid+2], d[sid+3], d[sid+4],d[sid+5])
            f.write(output)

    f.close()

    levavg = new_lev_avg
    # print(new_lev_avg)


# div = 50
# l = len(levavg)

# x = np.arange(1,div+1, 1)
# y = levavg[:div]
# z1 = np.polyfit(x, y, 3)
# p1 = np.poly1d(z1)
# yvals=p1(x)
# print(yvals)


# x = np.arange(1,(l-div)+1, 1)
# y = levavg[div:]
# z1 = np.polyfit(x, y, 0)
# p1 = np.poly1d(z1)
# yvals1=p1(x)
# print(yvals1)

# levavg[:div] = yvals
# levavg[div:] = yvals1

# ----------------------------------------------------------------

# metrics

f = open('smooth.xyz', 'r', encoding='utf-8')

data = f.read()

f.close

s = re.findall("\d\.\d+",data)

d = [float(i) for i in s]

rnum = int(len(d) / 6)

print(rnum)

f = open('gt.json', 'r', encoding='utf-8')

jdata = json.load(f)['data']

f.close()

print(len(jdata))

lev = -1

levid = -1

valid_num = 0

coord_error = 0

norm_error = 0

total_count = 0

for i in range(rnum):

    sid = i * 6
    clev = d[sid+1]

    if clev != lev:
        lev = clev
        levid += 1

    gt_r = jdata[levid]

    if gt_r**2 - d[sid+2]**2 > 0:
        
        gt_x = math.sqrt(gt_r**2 - d[sid+2]**2)
        norm = math.asin(d[sid+2] / gt_r)
        if norm > math.pi / 6 and norm < math.pi / 3:
            sample_x = d[sid]
            sample_norm = math.asin(d[sid+5])
            d_x = gt_x - sample_x
            d_n = norm - sample_norm
            valid_num += 1
            sample_r = math.sqrt(d[sid+2]**2 + d[sid]**2)
            d_r = gt_r - sample_r
            coord_error += math.sqrt(d_x**2)
            norm_error  += math.sqrt(d_n**2) * (180 / math.pi)

print(levid)
print(valid_num/rnum, coord_error/valid_num, norm_error/valid_num)





