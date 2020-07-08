from utils import *

import cam
import mtcnn
from alignment import alignment
from arcface import arcface


from PIL import Image

import cv2

import os
import sys

sys.path.append('alignment')
sys.path.append('mtcnn')
sys.path.append('cam')

def get_feature(name,frame):
    cv2.imwrite(os.path.join('images','%s.jpg'%(name)),frame)
    image = Image.open(os.path.join('images','%s.jpg'%(name)))
    bounding_boxes, landmarks = mtcnn.detect_faces(image)
    max_size = 0
    select = None
    for i in range(len(bounding_boxes)):
        size=(bounding_boxes[i][2]-bounding_boxes[i][0])*(bounding_boxes[i][3]-bounding_boxes[i][1])
        if size>max_size:
            select = i
            max_size = size
    select_lm = [0,0,0,0,0,0,0,0,0,0]
    for i in range(5):
        select_lm[i] = landmarks[select][i]
        select_lm[i+5] = landmarks[select][i+5]
    cv2_alignment = alignment(frame, select_lm)
    cv2.imwrite('images/%s_aligned.jpg'%(name),cv2_alignment)
    image_aligned = Image.open('images/%s_aligned.jpg'%(name))
    feature = arcface(image_aligned)
    clear_imgs()
    return feature


def reg(cam):
    print('Please enter name:')
    name = input()
    print('Find your best position, press A to take the photo.')
    frame = cam.get_frame('Register for %s'%(name))
    print('Registering......')
    feature = get_feature(name,frame)
    register(feature,name)
    clear_imgs()
    return

def clear():
    clear_reg()
    return

def test(cam,features,labels):
    print('Find your best position, press A to take the photo.')
    frame = cam.get_frame('Testing')
    feature = get_feature('__tmp__',frame).cpu().squeeze(dim=0)
    result = []
    for f in features:
        _f = f.squeeze(dim=0)
        result.append(torch.dot(feature,_f).item())

    flag = False
    for cos in result:
        if cos > 0.4:
            seq = result.index(cos)
            print('This is %s, cos = %f'%(labels[seq],cos))
            flag = True
            break
    if not flag:
        print('No register image matched.')
    clear_imgs()
    return

if __name__ == "__main__":

    print('Welcome to the SYSTEM')
    
    camera = cam.Camera()
    print('Camera ready')
    
    features,labels = get_reg_feature()
    if len(labels) != 0:
        print('Reg file found')
    else:
        print('Reg file is empty')
    while True:
        print('Press: A to add reg, C to clear reg, T to test a new face, Q to quit.')
        key = input()
        if key in ['A','a']:
            reg(camera)
            features,labels = get_reg_feature()
        elif key in ['C', 'c']:
            clear()
            features,labels = get_reg_feature()
        elif key in ['T', 't']:
            test(camera,features,labels)
        elif key in ['Q', 'q']:
            break
        else:
            print('Wrong input, please try again.')
    
