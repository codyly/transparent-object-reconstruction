import numpy as np
import math
import time
import cv2 as cv
import cv2.aruco as aruco
import random 
import os
from copy import deepcopy

def normalization(data):
    maxnum = np.max(data)
    # print(maxnum)
    minnum = np.min(data)
    dist = maxnum-minnum;
    data = (data - minnum) / dist
    return data

class artagTool:

    def __init__(self,
                calibration_folder = 'calibration/newData/photo/',
                standard_path = './calibration/newData/photo/c00.ppm',
                sample_path = './calibration/newData/sample/c00.ppm',
                raw_path = './calibration/newData/set3/raw/',
                dataset_roi = [0,0,0,0],
        ):
        self.board = None
        self.dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.board_dict = aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
        self.cameraMatrix = np.array([[2946.48,       0, 1980.53],
                                    [      0, 2945.41, 1129.25],
                                    [      0,       0,       1]])
        self.distortionCoef = np.array( [0.226317, -1.21478, 0.00170689, -0.000334551, 1.9892])
        self.roi = [0,0,0,0]
        self.calibration_folder = calibration_folder
        self.standard_path = standard_path
        self.sample_path = sample_path
        self.raw_path = raw_path
        self.dataset_roi = dataset_roi

    def show_image(self, image):
        cv.namedWindow('artager_tools_image')
        cv.imshow("artager_tools_image", image)
        cv.waitKey(0)

    def addTag(self, image, pos=(0,0), size=200, border=1, tagid=-1):
        if tagid < 0:
            tagid = random.randint(1, 249)
        (x,y) = pos
        roi = image[x:x+size, y:y+size]
        roi = aruco.drawMarker(self.dict, tagid, size, roi, border)
        image[x:x+size, y:y+size] = roi
        return image

    def warpTagsDefault(self, image, size=20, border=1, margin=10):
        (width,height) = (image.shape[1], image.shape[0])
        w = margin
        h = margin
        offset = margin + size + 2*border
        while w + offset <= width:
            h = margin
            while h + offset <= height:
                self.addTag(image, (w,h), size, border, -1)
                h += offset
            w += offset
        return image

    def addBoard(self):
        num = 9
        self.board = aruco.CharucoBoard_create(num, num, .025, .0125, self.board_dict)
        img = self.board.draw((200 * num, 200 * num))
        # self.show_image(img)

    def reverseImg(self, img):
        (width, height) = (img.shape[0], img.shape[1])
        for w in range(width):
            for h in range(height//2):
                tmp = img[w,h]
                img[w,h] = img[w,height-h-1]
                img[w,height-h-1] = tmp
        return img

    def detectTags(self, img):
        mtx = self.cameraMatrix
        dist = self.distortionCoef
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = \
            aruco.detectMarkers(img, self.board_dict, parameters=parameters)
        if ids is not None:
            ret, charucoCorners, charucoIds = \
                    aruco.interpolateCornersCharuco(corners, ids, img, self.board)
            # aruco.drawDetectedMarkers(img,corners,ids)
            aruco.drawDetectedCornersCharuco(img,charucoCorners)  
        # self.show_image(img)
        return charucoCorners, charucoIds, corners, ids

    def setROI(self, board_img):
        mtx = self.cameraMatrix
        dist = self.distortionCoef
        parameters =  aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = \
            aruco.detectMarkers(board_img, self.board_dict, parameters=parameters)
        if ids is not None:
            ret, charucoCorners, charucoIds = \
                    aruco.interpolateCornersCharuco(corners, ids, board_img, self.board)
            # self.show_image(board_img)
            # print(np.array(corners).shape)
            x_list = np.array(corners)[:,:,:,0]
            # print(x_list)
            y_list = np.array(corners)[:,:,:,1]
            min_x = int(np.min(x_list))
            range_x = int(np.max(x_list)-min_x)
            min_y = int(np.min(y_list))
            range_y = int(np.max(y_list)-min_y)
            self.roi = [min_x,min_y,range_x,range_y]
    
    def getROI(self, img):
        (x,y,w,h) = self.roi
        return img[y:y+h, x:x+w]

    def camera_calibration(self, folder):
        self.addBoard()
        allCorners=[]
        allIds=[]
        dirpath=os.getcwd()
        folder = dirpath + '/'+folder
        os.chdir(folder)
        # print(len(os.listdir(folder)))
        for i in range(len(os.listdir(folder))):
            img = cv.imread("c{:0>2d}.ppm".format(i),0)
            # self.show_image(img)
            parameters =  aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = \
                aruco.detectMarkers(img, self.board_dict, parameters=parameters)
            if corners == None or len(corners) == 0:
                continue
            if ids is not None:
                ret, charucoCorners, charucoIds = \
                    aruco.interpolateCornersCharuco(corners, ids, img, self.board)
                if corners is not  None  and charucoIds is not None:
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)
                aruco.drawDetectedMarkers(img,corners,ids)
                # self.show_image(img)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
        w,h=img.shape[1],img.shape[0]
        ret, K, dist_coef, rvecs, tvecs = \
            aruco.calibrateCameraCharuco(allCorners, allIds, self.board,(w,h),
                None,None)
        self.cameraMatrix = K
        self.distortionCoef = dist_coef
        os.chdir(dirpath)

    def undistortion(self,img):
        h, w = img.shape[:2]
        newcameramtx, roi=cv.getOptimalNewCameraMatrix(self.cameraMatrix, 
            self.distortionCoef,(w,h),1,(w,h))
        dst = cv.undistort(img, self.cameraMatrix, self.distortionCoef,
            None, newcameramtx)
        return dst

    def preprocessing(self,img):
        # img = cv.equalizeHist(img)
        img = self.undistortion(img)
        img = self.tailor_default(img)
        return img

    def printCameraPara(self):
        print('')
        print('------camera calibration result------')
        print('# camera matrix:')
        print(self.cameraMatrix)
        print('')
        print('# distortion coefficient:')
        print(self.distortionCoef)
        print('-------------------------------------')
        print('')

    def solveIteration(self, s, saCorner, refCorner, epsilon, tolerance, maxiter):
        iteration = 0
        error = math.inf
        while error > tolerance and iteration < maxiter:
            iteration += 1
            error = 0
            count = 0
            for i in range(len(saCorner)):
                for j in range(len(saCorner[i])):
                    corner = saCorner[i][j]
                    refcornor = refCorner[i][j]
                    x = corner[0]
                    y = corner[1]
                    dx = s[0] + s[1]*x + s[2]*y + s[3]*x*y + s[4]*x*x + s[5]*y*y + s[6]*x*y*y + s[7]*x*x*y - refcornor[0]
                    dy = s[8] + s[9]*x + s[10]*y + s[11]*x*y + s[12]*x*x + s[13]*y*y + s[14]*x*y*y + s[15]*x*x*y - refcornor[1]
                    gradx = grady = np.array(normalization([1,x,y,x*y,x*x,y*y,x*y*y,x*x*y]))
                    s[0:8] -= gradx * (epsilon * dx)
                    s[8:16] -= grady * (epsilon * dy)
                    error += math.sqrt(dx*dx + dy*dy)
                    count += 1

            error /= count
            # if iteration % 1000 == 0:
            #     print(error)
        print(error)
        return s

    def solveRemapMathod(self, standard, sample, epsilon=1e-3, tolerance=0.0005, maxiter=10000):
        assert(standard.shape == sample.shape)
        stCorner, stId, stc, sti = self.detectTags(standard)
        saCorner, saId, sac, sai = self.detectTags(sample)
        stc = np.array(stc)
        sti = np.array(sti)
        sac = np.array(sac)
        sai = np.array(sai)
        refCorner = []
        stCounter = 0
        saCounter = 0
        for saCounter in range(saId.shape[0]):
            if saId[saCounter] in stId:
                while stId[stCounter] != saId[saCounter]:
                    stCounter += 1
                refCorner.append(stCorner[stCounter])
        stCounter = 0
        saCounter = 0
        saCorner = saCorner.tolist()
        for saCounter in range(sai.shape[0]):
            if sai[saCounter] in sti:
                while stCounter < sti.shape[0] and sti[stCounter] != sai[saCounter]:
                    stCounter += 1
                if stCounter < sti.shape[0]:
                    saCorner.append(sac[saCounter][0])
                    refCorner.append(stc[stCounter][0])
        refCorner = np.array(refCorner)
        saCorner = np.array(saCorner)

        for i in range(len(saCorner)):
            for j in range(len(saCorner[i])):
                saCorner[i][j][0] = (saCorner[i][j][0])/self.roi[2]
                saCorner[i][j][1] = (saCorner[i][j][1])/self.roi[3]
                refCorner[i][j][0] = (refCorner[i][j][0])/self.roi[2]
                refCorner[i][j][1] = (refCorner[i][j][1])/self.roi[3]

        # solution = np.ones([16]) / 2
        solution = np.array([0.15712081,  0.70222784,  0.04595652, -0.22687445,  0.00710069, -0.05140645,
                             0.20044894,  0.05712042,  0.04112884,  0.0735165,   0.89877992, -0.20655421,
                            -0.0716841,   0.04083822,  0.04500059,  0.17699867])
        solution = self.solveIteration(solution, saCorner, refCorner, epsilon, tolerance, maxiter)
        print(solution)
        return solution
    
    def remap(self, img, s):
        new_img = deepcopy(img)
        new_img -= new_img
        new_img += 255
        for h in range(self.roi[2]):
            for w in range(self.roi[3]):
                x = h/self.roi[2]
                y = w/self.roi[3]
                fx = s[0] + s[1]*x + s[2]*y + s[3]*x*y + s[4]*x*x + s[5]*y*y + s[6]*x*y*y + s[7]*x*x*y
                fy = s[8] + s[9]*x + s[10]*y + s[11]*x*y + s[12]*x*x + s[13]*y*y + s[14]*x*y*y + s[15]*x*x*y
                hh = min(int(fx*self.roi[2]), img.shape[1]-1)
                ww = min(int(fy*self.roi[3]), img.shape[0]-1)
                new_img[ww,hh]=img[w,h]
        # self.show_image(new_img)
        new_img = self.tailor_default(new_img)
        new_img = self.tailor(new_img, 1/12)
        # self.show_image(new_img)
        return new_img

    def tailor_default(self, img):
        (h, w) = img.shape
        roi = img[int(0.125*h):int(0.875*h), int(0.125*w):int(0.875*w)]
        return roi

    def tailor(self, img, rate):
        (h, w) = img.shape
        roi = img[int(rate*h):int((1-rate)*h), int(rate*w):int((1-rate)*w)]
        return roi

    def refined_dataset(self, solution):
        raw_path = self.raw_path
        refined_path = raw_path+'refined/'

        dirpath=os.getcwd()
        folder = dirpath + '/'+raw_path
        os.chdir(folder)
        # print(folder)
        for i in range(len(os.listdir(folder))-1):
        # for i in range(1):
            img = cv.imread("myProject_{:0>3d}.ppm".format(i),0)
            
            img = self.preprocessing(img)
            img = self.getROI(img)
            img = self.remap(img, solution)
            
            # img = self.remap(img, solution)
            img = cv.resize(img,(128, 128))
            cv.imwrite("./refined/myProject_{:0>3d}.jpeg".format(i), img, [int(cv.IMWRITE_JPEG_QUALITY),100])
            # self.show_image(img)
        os.chdir(dirpath)

    def run(self):
        calibration_folder = self.calibration_folder
        standard_path = self.standard_path
        sample_path = self.sample_path
       
        self.camera_calibration(calibration_folder)
        img = cv.imread(standard_path, cv.IMREAD_GRAYSCALE)
        img = self.preprocessing(img)
        self.setROI(img)
        standard = self.getROI(img)

        img = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)
        img = self.preprocessing(img)
        
        sample = self.getROI(img)
        solution = self.solveRemapMathod(standard, sample)
        self.show_image(self.remap(sample, solution))
        self.refined_dataset(solution)
        

    
    