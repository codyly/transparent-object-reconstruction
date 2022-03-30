import math
import numpy as np 
import cv2
import sys
import json
from tqdm import trange
import dataset_generator_new as dgn
from copy import copy

from sympy import *


SIGMA = 1.0

IS_DIRECT_TRACE = 0.9


NEAR_DIST = 3.0

START_ANGLE = 10

END_ANGLE = 80



def dist(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def normalize(v):
    l = dist([0,0], v)
    v[0] /= l
    v[1] /= l
    return v

class camera:
    def __init__(self):
        self.eye       = [ 3, 0.8]
        self.eye_ray   = [-1, 0]
        self.eye_angle = 0
        self.alpha     = 0
    
    def get_eye_coord(self):
        return self.eye

    def get_eye_dir(self):
        return self.eye_ray

    def set_eye_coord(self, eye_coord):
        self.eye[0], self.eye[1] = eye_coord[0], eye_coord[1]

    def set_eye_dir(self, obj_coord):
        ox, oy = obj_coord[0], obj_coord[1]
        self.eye_ray[0], self.eye_ray[1] = ox - self.eye[0], oy - self.eye[1]
        self.eye_ray = normalize(self.eye_ray)

    def update_eye_alpha(self):
        # v = [-self.eye_ray[0], -self.eye_ray[1]]
        # cosa, sina = self.eye_ray[0], self.eye_ray[1]
        # a = math.asin(sina)
        # if cosa < 0:
        #     a += math.pi
        # if cosa > 0 and sina < 0:
        #     a += math.pi * 2
        # self.alpha = a 
        pass

    def get_alpha(self):
        # self.update_eye_alpha()
        return self.alpha

class pattern:
    def __init__(self):
        self.r_b       = [ 4, 2]
        self.l_t       = [-4, 6]
        self.dx        = - 8.0 / 128.0
        self.dy        =   4.0 /  64.0
        self.pattern_angle = 0
        self.pattern_image = None
    
    def get_pattern_coords(self):
        coords = []
        for i in range(8192):
            tx = i // 64
            ty = i  % 64
            x = self.r_b[0] + tx * self.dx
            y = self.r_b[1] + ty * self.dy
            coords.append([x,y])
        return coords


class polar_coord:
    def __init__(self, cx, cy):
        self.c = [cx, cy]
        self.base = [1, 0]
        self.near = NEAR_DIST
        # self.dx = (1/self.near) / 64
        self.dist = 10
        self.dx = (self.dist) / 64
        self.dy = math.pi / 128 * (END_ANGLE - START_ANGLE) / 90

    def get_coord(self, x, y):
        if math.fabs(dist(self.c, [x,y])) < self.near and math.fabs(dist(self.c, [x,y])) > (self.near + self.dist):
            xx = -1
        else:
            # xx = int(((1/self.near)  - 1 / dist(self.c, [x,y]))/ self.dx)
            xx = int((dist(self.c, [x,y]) - self.near) / self.dx)
        v = [x-self.c[0], y-self.c[1]]
        v = normalize(v)
        cost, sint = v[0], v[1]
        yy = math.asin(sint)
        if cost < 0 and sint > 0:
            yy = math.pi - yy
        elif cost > 0 and sint < 0:
            yy += math.pi * 2
        elif cost < 0 and sint < 0:
            yy = math.pi - yy
        if yy < START_ANGLE / 90 * math.pi or yy > END_ANGLE / 90 * math.pi:
            yy = int(-1)
        else:
            yy = int((yy-START_ANGLE / 90 * math.pi) / self.dy)
        return [xx, yy]

def interpolate(trace):
    # for r in range(128):
    #     v = 0
    #     for c in range(12):
    #         if trace[r][c] > 128:
    #             v = trace[r][c]
    #             break
    #     trace[r, 0:c] = v 
    #     for i in range(0, 64):
    #         if trace[r,i] > v:
    #             trace[r,i] = v
        # for i in range(63, c, -1):
        #     if trace[r,i-1] <= trace[r,i]:
        #         trace[r,i-1] = trace[r,i]

    return trace

def process_polar_trace(trace):
    cc = np.zeros([128,]).astype(np.float32)
    offset = 1
    d_list = []
    thresh = 50
    can_list = []
    attenuation = 0
    for r in range(128):
        valid_num = 0
        start = 0
        for start in range(64):
            if trace[r, start] > thresh:
                break
        if start >= 10:
            trace[r,:] = 0
            continue
        for c in range(start, 64):

            # candidate_value = min(trace[r, max(c-1, start)], np.mean(trace[r, start : c+1]))
            # candidate_value = np.mean(trace[r, max(start, c-offset) : min(64,c+offset)])
            candidate_value = min(trace[r,c], np.mean(trace[r, max(start, c-offset) : min(64, c+1)]))
            # candidate_value = trace[r,c]
            if True: 
                # candidate_value = min(trace[r, max(start, c-1)], np.mean(trace[r, max(start, c-offset) : min(64,c+offset)]))
                # candidate_value = trace[r,c]
                # if trace[r,max(0,c-1)] != 0:
                #     candidate_value = min(trace[r,c], trace[r,max(0,c-1)])
                # else:
                # candidate_value = trace[r,c]
                # candidate_value = min(trace[r, max(c-1, start)], np.mean(trace[r, start : c+1]))
                # print(candidate_value)
                t = int(trace[r,c])
                d = - (max(0, candidate_value - attenuation) - t) ** 2 / (SIGMA**2)
                
                d = math.exp(d) * max(0, candidate_value - attenuation)
                # d_list.append(d)
                # can_list.append(can_list)
                trace[r, c] = max(0, candidate_value - attenuation)
                cc[r]+=d
                valid_num += 1
            else:
                trace[r, c] = candidate_value
        # if valid_num == 0:
        #     cc[r] = 0
        # else:
        #     cc[r] = cc[r] / valid_num
    # cc1 = np.zeros([128,]).astype(np.float32)
    # for i in range(128):
    #     cc1[i] = np.mean(cc[max(0, i-1):min(128, i+2)])
    # d_list=np.array(d_list)
    # can_list = np.array(can_list)
    # thresh_can = sorted(can_list[np.where(can_list > 100)], reverse=True)[int(8192*(1-IS_DIRECT_TRACE))]

    # cc = np.sum(d_list[np.where(can_list>=thresh_can)])

    return trace, cc

def get_norm_thetas(eye, trace):  # theta = (y+eye.alpha)/2 + pi

    thetas = np.zeros([128,]).astype(np.float32)

    offset_y = START_ANGLE / 90 * math.pi
    step_trace_y = math.pi / 128 * (END_ANGLE - START_ANGLE) / 90

    step_y = math.pi / 128    

    for i in range(128):
        new_coord = (offset_y + i * step_trace_y + eye.get_alpha()) / 2.0
        if new_coord >= 2 * math.pi:
            new_coord -= 2 * math.pi
        new_coord = math.floor(new_coord/step_y)
        if i == 0:
            data = [trace[0], trace[1]]
        elif i == 127:
            data = [trace[126], trace[127]]
        else:
            data = [trace[i-1], trace[i], trace[i+1]]
        data = np.array(data)
        # data = trace[max(0, i-1):min(128, i+2)]
        valid = data[np.where(data > 0)]
        if len(valid) > 0 :
            thetas[new_coord] = np.mean(valid)

    return thetas


def analyze_coord(ox, oy, pat, cam):

    coord = polar_coord( ox, oy )

    new_trace = np.zeros([128, 64]).astype(np.float32)

    pat_coords = pat.get_pattern_coords()

    for i in range(8192):
        new_xy = coord.get_coord(pat_coords[i][0], pat_coords[i][1])
        nx, ny = new_xy[0], new_xy[1]
        if nx > 0 and nx < 64 and ny > 0 and ny < 128:
            brightness = pat.pattern_image[ 127 - i //64 ][63 - i%64]
            new_trace[ny][nx] += brightness

    # cv2.imshow('tmp', new_trace)
    # cv2.waitKey()

    old_trace = new_trace
    

    new_trace_i = interpolate(new_trace)

    new_trace_i, c = process_polar_trace(new_trace_i)


    thetas = get_norm_thetas(cam, old_trace)
    intensity = np.sum(thetas) 
    angle = np.argmax(c) / 128 * (END_ANGLE - START_ANGLE)  + START_ANGLE
    angle_reg = angle / 180 * math.pi
    angle_idx = np.argmax(c)


    ray_intensity = 200


    pre_trace = new_trace_i

    # print(np.max(new_trace))

    result = { 'ray': ray_intensity, 'intensity': intensity, 'c': c[angle_idx], 'reg':angle_reg,  'angle': angle, 'distribution':thetas, 'coord': [ ox, oy ], 'trace': new_trace_i }

    return result



def single_test():
    cam_x = 3
    cam_y = 0.8
    pred_x = 0.3
    pred_interval = 0.1

    cam = camera()      # set camera
    cam.set_eye_coord([cam_x, cam_y])

    pat = pattern()  # load pattern
    tag = sys.argv[2]
    filename = '.\\' +tag +'.png'
    pat.pattern_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    pred_x = pred_x

    start_point     = [pred_x, cam_y]
    step_along_axis = -0.002
    end_point       = [pred_x - pred_interval, cam_y]
    step_num        = int((end_point[0] - start_point[0]) / step_along_axis)

    results = []

    for i in range(step_num):
        xi = start_point[0] + i * step_along_axis
        yi = start_point[1]
        results.append(analyze_coord(xi, yi, pat, cam))

    def custom_key(record):
        return record['c']

    results = sorted(results, key=custom_key, reverse=True)

    output = {'coord': results[0]['coord'], 'angle': results[0]['distribution']}

    print(output)

    cv2.imshow('new', results[0]['trace'])
    cv2.waitKey()


def edge_test():
    pixel_width = 0.004511 # 0.006766
    start_width = 0.4255   #0.966
    cam_x = 2 # 3

    start_pred_x  = 0.1845     #0.3
    pred_interval = 0.2        #0.1
    pred_step     = 0.00451    #0.007

    jstr = {}

    tag = sys.argv[2]

    for pi in trange(50):
        cam_y = start_width - pi * pixel_width

        cam = camera()      # set camera
        cam.set_eye_coord([cam_x, cam_y])

        pat = pattern()  # load pattern
        # tag = sys.argv[1]
        filename = '.\\'+tag+'\\prac_edge_'+tag+'_{:03d}'.format(pi)+ '.png'
        pat.pattern_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # pred_x = start_pred_x + pred_step * pi
        start_x = math.sqrt(1 - cam_y**2 / 0.5**2) * 0.5
        end_x   = math.sqrt(min(1, max(0, 1 - cam_y**2 / 0.4**2))) * 0.4

        start_point     = [start_x, cam_y]
        step_along_axis = -0.008
        end_point       = [end_x, cam_y]
        step_num        = int((end_point[0] - start_point[0]) / step_along_axis)

        results = []

        for i in range(step_num):
            xi = start_point[0] + i * step_along_axis
            yi = start_point[1]
            results.append(analyze_coord(xi, yi, pat, cam))

        def custom_key(record):
            return record['c']

        results = sorted(results, key=custom_key, reverse=True)



        output = {'coord': results[0]['coord'], 'angle': results[0]['angle'], 
                  'candidate_coord': [results[i]['coord'] for i in range(4)], 'candidate_angle':  [results[i]['angle'] for i in range(4)]}

        # print(pi, output)

        jstr['{:03d}'.format(pi)] = output

    with open('.\\result_'+tag+'.json', 'w+', encoding='utf-8') as f:
        json.dump(jstr, f, indent=4)
        # cv2.imshow('new', results[0]['trace'])
        # cv2.waitKey()


def append_xyz_file(filename, c, n):
    f = open(filename, 'a+', encoding='utf-8')
    rec = '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(c[0], c[1], c[2], n[0], n[1], n[2])

    f.write(rec)
    f.close()

def surface_test():

    last_coord = None

    # bottle test
    pixel_width = 0.004511 # 0.006766
    start_width = 0.4255   #0.966
    cam_x = 2

    # chull failed test

    pixel_width = 0.0225
    start_width = -1.1125
    cam_x = 10

    # chull test 2

    pixel_width = 0.0225
    start_width = -0.8875
    cam_x = 10

    # chull test 3

    pixel_width = 0.0225
    start_width = -0.325
    cam_x = 10

    start_pred_x  = 0.1845     #0.3
    pred_interval = 0.2        #0.1
    pred_step     = 0.00451    #0.007

    start_pred_x  = 0.5        #0.3
    pred_interval = 0.5        #0.1
    pred_step     = 0.00451    #0.007

    jstr = {}

    tag = sys.argv[2]

    xyz_file = '.\\' + tag + '.xyz'

    initial_height = 1.125
    height_step = 0.0225

    row = 20
    col = 100

    first = True

    poolsize = 5

    pool = np.zeros([2 * poolsize]).astype(np.float32)
    pool_id = 0
    pool_prepared = False
    solution = [2100, 1.1, 0.001]

    for pi in trange(row * col):

        if pi % col >= col - 20:
            continue

        if pi % col == 0:
            first = False

        if int(pi / col) % 2 == 0 and pi % 2 == 1:
            continue
            # pass
        if int(pi / col) % 2 == 1 and pi % 2 == 0:
            continue

        cam_y = start_width - (pi % col) * pixel_width

        height = initial_height - (pi // col) * height_step

        cam = camera()      # set camera
        cam.set_eye_coord([cam_x, cam_y])

        pat = pattern()  # load pattern
        # tag = sys.argv[1]
        filename = '.\\'+tag+'\\prac_'+tag+'_{:05d}'.format(pi)+ '.png'
        pat.pattern_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # print(np.sum(pat.pattern_image))
        a = np.sum(pat.pattern_image) 
        # print(a)
        if a > 30000:


            # pred_x = start_pred_x + pred_step * pi
            # r_inside = 0.4
            # r_outside = 0.5
            
            # if pi // 50 < 50:
            #     r_inside  = 0.2 + 0.004 * (pi // 50)
            #     r_outside = 0.3 + 0.004 * (pi // 50)

            # start_x = math.sqrt(min(1, max(0, 1 - cam_y**2 / r_outside **2))) * r_outside
            # end_x   = math.sqrt(min(1, max(0, 1 - cam_y**2 / r_inside **2))) * r_inside

            start_x =  1
            end_x   =  -2

            # if pi / 50 < 50:
            #     start_x =  0.0
            #     end_x   = -1.0
            #     # print('a')

            if math.fabs(start_x - end_x) <= 0.01:
                continue 

            start_point     = [start_x, cam_y]
            step_along_axis = -0.02
            end_point       = [end_x, cam_y]
            step_num        = int((end_point[0] - start_point[0]) / step_along_axis)

            results = []

            for i in range(step_num):
                xi = start_point[0] + i * step_along_axis
                yi = start_point[1]
                results.append(analyze_coord(xi, yi, pat, cam))

            def custom_key(record):
                return record['c']

            results = sorted(results, key=custom_key, reverse=True)

            if results[0]['c'] > 0:
                print('')

                print('{:.3f}'.format(results[0]['coord'][0]), results[0]['angle'], '{:.3f}'.format(results[1]['coord'][0]), results[0]['angle'],
                    '{:.3f}'.format(results[2]['coord'][0]), results[2]['angle'])

                c = [results[0]['coord'][0], height, results[0]['coord'][1]]
                tg_idx = 0
                if last_coord is None or first == False:
                    last_coord = [c[0], c[2]]
                    
                else:
                    for ri in range(2):
                        if results[ri]['c'] > 0 and math.fabs(results[ri]['c'] - results[ri+1]['c']) / results[0]['c'] < 0.05 and \
                            dist(last_coord, results[ri]['coord']) > dist(last_coord, results[ri+1]['coord']):
                            if tg_idx == 0 and dist(last_coord, results[0]['coord']) > dist(last_coord, results[ri+1]['coord']):
                                c = [results[ri+1]['coord'][0], height, results[ri+1]['coord'][1]]
                                tg_idx = ri + 1
                            elif tg_idx == ri:
                                c = [results[ri+1]['coord'][0], height, results[ri+1]['coord'][1]]
                                tg_idx = ri + 1
                    last_coord = [c[0], c[2]]


                n = [math.cos(results[tg_idx]['angle'] / 180 * math.pi), 0, math.sin(results[tg_idx]['angle'] / 180 * math.pi)]
                out = np.hstack((pat.pattern_image,  results[tg_idx]['trace']))
                def ww(ray, reg, ag=0.0023):
                    w = (2 / (1 + math.sqrt(1 + (ag*math.tan(reg))**2))) ** 2
                    return ray / w
                def fr(i, eta=1.5):
                    sini = math.sin(i)
                    cosi = math.cos(i)
                    sint = sini / eta
                    cost = math.sqrt(1-sint**2)
                    etah = (sini*cosi-sint*cost) / (sini*cosi+sint*cost)
                    etav = (sini*cost-sint*cosi) / (sini*cost+sint*cosi)
                    return 0.5*(etah ** 2 + etav ** 2)

                from scipy.optimize import fsolve
                # from sympy import *
                def func_ag(paramlist, args):

                    eta = 1.5

                    K = 2109

                    ag  = paramlist[0]
                    
                    i0, ray0 = args[0], args[1]
                    
                    sini0 = math.sin(i0)
                    cosi0 = math.cos(i0)
                    sint0 = sini0 / eta
                    cost0 = math.sqrt(1-sint0**2)
                    etah0 = (sini0*cosi0-sint0*cost0) / (sini0*cosi0+sint0*cost0)
                    etav0 = (sini0*cost0-sint0*cosi0) / (sini0*cost0+sint0*cosi0)
                    fr0 = 0.5*(etah0 ** 2 + etav0 ** 2)
                    w0 = (2 / (1 + math.sqrt(1 + (ag*math.tan(i0))**2))) ** 2
                    F0 = K  * fr0 * w0 - ray0

                    return [ F0 ]

                def func_eta(paramlist, args):

                    eta = paramlist[0]

                    K = 2109

                    ag  = args[2]
                    
                    i0, ray0 = args[0], args[1]
                    
                    sini0 = math.sin(i0)
                    cosi0 = math.cos(i0)
                    sint0 = sini0 / eta
                    cost0 = math.sqrt(1-sint0**2)
                    etah0 = (sini0*cosi0-sint0*cost0) / (sini0*cosi0+sint0*cost0)
                    etav0 = (sini0*cost0-sint0*cosi0) / (sini0*cost0+sint0*cosi0)
                    fr0 = 0.5*(etah0 ** 2 + etav0 ** 2)
                    w0 = (2 / (1 + math.sqrt(1 + (ag*math.tan(i0))**2))) ** 2
                    F0 = K  * fr0 * w0 - ray0

                    return [ F0 ]

                def func_eta_k(paramlist, args):

                    eta = paramlist[1]

                    K = paramlist[0]

                    ag  = args[2]
                    
                    i0, ray0 = args[0], args[1]
                    
                    sini0 = math.sin(i0)
                    cosi0 = math.cos(i0)
                    sint0 = sini0 / eta
                    cost0 = math.sqrt(1-sint0**2)
                    etah0 = (sini0*cosi0-sint0*cost0) / (sini0*cosi0+sint0*cost0)
                    etav0 = (sini0*cost0-sint0*cosi0) / (sini0*cost0+sint0*cosi0)
                    fr0 = 0.5*(etah0 ** 2 + etav0 ** 2)
                    w0 = (2 / (1 + math.sqrt(1 + (ag*math.tan(i0))**2))) ** 2
                    F0 = K  * fr0 * w0 - ray0

                    i1, ray1 = args[2], args[3]
                    
                    sini1 = math.sin(i1)
                    cosi1 = math.cos(i1)
                    sint1 = sini1 /  eta
                    cost1 = math.sqrt(1-sint1**2)
                    etah1 = (sini1*cosi1-sint1*cost1) / (sini1*cosi1+sint1*cost1)
                    etav1 = (sini1*cost1-sint1*cosi1) / (sini1*cost1+sint1*cosi1)
                    fr1 = 0.5*(etah1 ** 2 + etav1 ** 2)
                    w1 = (2 / (1 + math.sqrt(1 + (ag*math.tan(i1))**2))) ** 2
                    F1 = K  * fr1 * w1 - ray1

                    return [ F0, F1 ]

                ray_intensity = 0

                dx, dy = math.cos(2*results[tg_idx]['reg']), math.sin(2*results[tg_idx]['reg'])
                dstep = .25
                nx, ny = results[tg_idx]['coord'][0], results[tg_idx]['coord'][1]
                cnt = 0
                rayn = 0
                while cnt < 50:
                    cnt += 1
                    nx += dstep * dx
                    ny += dstep * dy
                    if nx < pat.r_b[0] and nx > pat.l_t[0] and ny < pat.l_t[1] and ny > pat.r_b[1]:
                        ptx = math.floor((nx - pat.l_t[0])/pat.dy) 
                        pty = math.floor((ny - pat.r_b[1])/pat.dy)
                        # cv2.imshow('1',pat.pattern_image)
                        # cv2.waitKey()
                        # sys.exit(0)
                        if pat.pattern_image[ptx][63-pty] > 100:
                            ray_intensity = max( ray_intensity,  pat.pattern_image[ptx][63-pty])
                            rayn += 1
                
                if rayn != 0:

                    results[tg_idx]['ray'] = ray_intensity 

                    print('{:.3f}'.format(results[tg_idx]['coord'][0]), results[tg_idx]['angle'], results[tg_idx]['c'], results[tg_idx]['ray'], \
                        ww(results[tg_idx]['ray'],results[tg_idx]['reg']) / fr(results[tg_idx]['reg']))

                    args = [results[tg_idx]['reg'], results[tg_idx]['ray'], 0.23]
                    pool[2:] = pool[0:-2]
                    pool[0:2] = [results[tg_idx]['reg'], results[tg_idx]['ray']]
                    # pool[pool_id * 2 + 1] = 
                    pool_id += 1

                    if pool_prepared == False and pool_id == poolsize:
                        pool_prepared = True
                    pool_id = pool_id % poolsize

                    if True:
                        ss = copy([solution[1]])
                        # s=fsolve(func_ag, ss, args)
                        s = fsolve(func_eta, ss, args)
                        print('Solved eta:{:.3f} GGX_Ag:{:.4f}'.format( math.fabs(s[0]), args[2]))

                        # args1 = pool[0:4]
                        # ss = copy(solution[:2])
                        # s = fsolve(func_eta_k, ss, args1)
                        # print('Solved eta:{:.3f} GGX_Ag:{:.4f}'.format( math.fabs(s[1]), 0.0023))

                        # s[1] = max(1, s[1])
                        # solution = s

                cv2.imwrite('.\\trace' + tag[7:] + '\\prac_'+tag+'_{:05d}'.format(pi)+ '.png', out)           
                    # if pi == 72:
                    #     cnt = 0
                    #     for result in results: 
                    #         out_a = np.hstack((pat.pattern_image,  result['trace']))
                    #         cv2.imwrite('.\\trace_chull_5\\a\\prac_'+tag+'_{:05d}'.format(cnt)+ '.png', out_a)
                    #         print('\n', cnt, '{:.3f}'.format(result['coord'][0]), result['c'],  result['angle'])
                    #         cnt += 1
                    #     sys.exit(0)
                if first == False:
                    first = True
                else:
                    append_xyz_file(xyz_file, c, n)
                # d = math.sqrt(c[0]**2 + c[2]**2)
                # if math.fabs(d - r_outside) > 0.01 and math.fabs(d - r_inside) > 0.01:
                #     append_xyz_file(xyz_file, c, n)

        #     output = {'coord': results[0]['coord'], 'angle': results[0]['angle'], 
        #               'candidate_coord': [results[i]['coord'] for i in range(4)], 'candidate_angle':  [results[i]['angle'] for i in range(4)]}

        #     # print(pi, output)

        #     jstr['{:03d}'.format(pi)] = output

        # with open('.\\result_'+tag+'.json', 'w+', encoding='utf-8') as f:
        #     json.dump(jstr, f, indent=4)
            # cv2.imshow('new', results[0]['trace'])
            # cv2.waitKey()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in 'single_test':
        single_test()
    elif len(sys.argv) > 1 and sys.argv[1] in 'edge_test':
        edge_test()
    elif len(sys.argv) > 1 and sys.argv[1] in 'surface_test':
        surface_test()




