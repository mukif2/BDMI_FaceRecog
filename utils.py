import cv2
import torch
import os
import shutil

def get_reg_feature():
    '''
    读取reg文件，将注册人脸feature和label返回为两个list
    '''    
    features = []
    labels = []
    if os.path.exists('reg'):
        with open('reg','r') as f:
            text = f.readlines()
        for line in text:
            splits = line.split()
            labels.append(splits[0])
            features.append(torch.unsqueeze(torch.tensor([float(x) for x in splits[1:]]),0))
        return (features, labels)
    else:
        return ([],[])

def register(feature,label):
    '''
    将feature, label写入reg文件
    '''
    line = ''
    line += label
    feature = feature.cpu().numpy().tolist()
    for x in feature[0]:
        line += ' %.8f'%(x)
    line += '\n'
    with open('reg','a') as f:
        f.write(line)

def clear_reg():
    '''
    清空全部reg
    '''
    with open('reg','w') as f:
        f.write('')

def clear_imgs():
    if os.path.exists('images'):
        shutil.rmtree('images')
        os.mkdir('images')
    return
    