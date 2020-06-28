import os
import sys
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.ndimage.filters import maximum_filter

from openvino.inference_engine import IECore

def detect_peak(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image,mask=~(image == local_max))
    
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def decode(hm, displacements, threshold=0.8):
    hm = hm.reshape(hm.shape[2:])     # (40,30)
    peaks = detect_peak(hm)
    peakX = peaks[1]
    peakY = peaks[0]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x,y in zip(peakX, peakY):
        conf = hm[y,x]
        if conf<threshold:
            continue
        points=[]
        for i in range(8):
            dx = displacements[0, i*2  , y, x]
            dy = displacements[0, i*2+1, y, x]
            points.append((x/scaleX+dx, y/scaleY+dy))
        objs.append(points)
    return objs

def show_heatmap(hm):
    h = hm.reshape((hm.shape[2],hm.shape[3],1))
    h = cv2.resize(h, None, fx=10, fy=10)
    cv2.imshow('heatmap', h)

def draw_box(image, pts):
    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0,1), (1,3), (0,2), (3,2), (1,5), (0,4), (2,6), (3,7), (5,7), (6,7), (6,4), (4,5)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]*scaleX), int(pt0[1]*scaleY))
        pt1 = (int(pt1[0]*scaleX), int(pt1[1]*scaleY))
        cv2.line(image, pt0, pt1, (255,0,0))
    
    for i in range(8):
        pt = pts[i]
        pt = (int(pt[0]*scaleX), int(pt[1]*scaleY))
        cv2.circle(image, pt, 8, (0,255,0), -1)
        cv2.putText(image, str(i), pt,  cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

def main(args):

    base,ext = os.path.splitext(args.model)
    if ext != '.xml':
        print('Not .xml file is specified ', args.model)
        sys.exit(-1)

    ie = IECore()
    net = ie.read_network(base+'.xml', base+'.bin')
    exenet = ie.load_network(net, 'CPU')

    inblobs =  (list(net.inputs.keys()))
    outblobs = (list(net.outputs.keys()))
    print(inblobs, outblobs)

    inshapes  = [ net.inputs [i].shape for i in inblobs  ]
    outshapes = [ net.outputs[i].shape for i in outblobs ]
    print(inshapes, outshapes)

    for idx, outshape in enumerate(outshapes):
        if outshape[1]==1:
            hm_idx = idx
        if outshape[1]==16:
            dis_idx = idx

    if args.input == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        if args.input == 'cam':
            _, img_orig = cap.read()
        else:
            img_file = args.input
            img_orig = cv2.imread(img_file)

        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (inshapes[0][3], inshapes[0][2]))
        img = img.transpose((2,0,1))

        res = exenet.infer({inblobs[0]:img})

        hm = res[outblobs[hm_idx]]
        displacements = res[outblobs[dis_idx]]

        # show heatmap
        if args.heatmap == True:
            show_heatmap(hm)

        # decode inference result
        objs = decode(hm, displacements, threshold=0.7)

        # draw bbox
        for obj in objs:
            draw_box(img_orig, obj)
        
        if args.input == 'cam':
            cv2.imshow('output', img_orig)
            if cv2.waitKey(1)==27:
                return
        else:
            cv2.imwrite('output.jpg', img_orig)
            cv2.imshow('output', img_orig)
            cv2.waitKey(0)
            return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='chair.jpg', help='input image file name')
    parser.add_argument('-m', '--model', type=str, default='./objectron_chair/saved_model.xml', help='FBFace IR model file name (*.xml)')
    parser.add_argument('--heatmap', action='store_true', required=False, default=False, help='Display heatmap')
    args = parser.parse_args()

    main(args)
