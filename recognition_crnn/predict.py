#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文简体识别
@author: chineseocr
"""
__version__='chineseocr'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from recognition_crnn.keys import Lang
from config import RecognitionModelPath2
from PIL import Image
from tensorflow.keras.models import load_model
# from tensorflow.keras.backend import set_session
import tensorflow as tf
ocrModel=''
import numpy as np

ocrModel = { 'chinese': RecognitionModelPath2}

def load_models():
    modelDict = {}
    
    for lan in ocrModel:
        modelPath    = ocrModel[lan]
        characters   = Lang[lan.split('-')[0]]
        if characters is not None:            
            charactersS  = [' ']+list(characters)+[' ',' ']
            
            model = load_model(modelPath)
            model._make_predict_function()
            charDict ={s:ind for ind,s in enumerate(charactersS)} 
            if ' ' in charDict:
                charDict[' '] =1
            modelDict[lan] =[model,charactersS,charDict]
            
    return  modelDict
            

modelDict = load_models()
# model._make_predict_function()
def predict_prob(image,lan,useStr=None):
    ## useStr 用户字库
    
    scale = image.size[1]*1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    if w<1:
        return [],[]
    image   = image.resize((w,32),Image.BILINEAR)
    
    image = (np.array(image.convert('L'))/255.0-0.5)/0.5
    tmp  =  np.zeros((32,w+32))
    tmp[:] = (128/255-0.5)/0.5
    tmp[:,16:-16] = image

        
    image = tmp.reshape((1,32,-1,1))
    
    model,charactersS,charDict =  modelDict.get(lan,[None,None,None])
    
    out =[' ']
    prob = [1.0]
    if model is not None:
          y_pred = model.predict(image)
          y_pred = y_pred[0][2:,]
          if useStr is not None and 'chinese' in lan:
              tmpuseStr=''
              indexList=[0]
              tmpuseStr = set(charactersS)&set(useStr)
              tmpuseStr = list(tmpuseStr)
              indexList+=[ charDict[s] for s in tmpuseStr]
              indexList+=[len(charactersS)-2,len(charactersS)-1]
              charactersS  = [' ']+list(tmpuseStr)+[' ',' ']
              y_pred=y_pred[:,indexList]
              
          prob = y_pred
          index = prob.argmax(axis=1)
          prob  = [ prob[ind,pb] for ind,pb in enumerate(index)]
          out,prob = decode_prob(index,prob,charactersS)
          
    return ''.join(out),prob

def decode_prob(t,prob,charactersS):
        length = len(t)
        char_list = []
        prob_list = []
        n = len(charactersS)
        for i in range(length):
            if t[i] not in [n-1,n-2] and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(charactersS[t[i] ])
                        prob_list.append(prob[i])
        
        return char_list,[float(round(x,2)) for x in prob_list]
    

if __name__=='__main__':
    import time
    test_path = '/'
    f = open('result.txt', 'w', encoding='utf-8')
    for file in os.listdir(test_path):
        image = Image.open(os.path.join(test_path, file))
        t = time.time()
        out,pro = predict_prob(image, lan='chinese', useStr=None)
        f.writelines('{} {} {}\n'.format(file,out,pro))
        print('{} {}'.format(file,out))
