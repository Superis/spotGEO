import time
import matplotlib.pyplot as plt


import numpy as np

from collections import defaultdict

import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

import itertools

import random
import pandas as pd
from skimage import measure
import csv


def read_image(path):
    return plt.imread(path)

def read_annotation_file(path):
    with open(path) as annotation_file:
        annotation_list = json.load(annotation_file)
    # Transform list of annotations into dictionary
    annotation_dict = {}
    for annotation in annotation_list:
        sequence_id = annotation['sequence_id']
        if sequence_id not in annotation_dict:
            annotation_dict[sequence_id] = {}
        annotation_dict[sequence_id][annotation['frame']] = annotation['object_coords']
    return annotation_dict


def random_different_coordinates(coords, size_x, size_y, pad,cond):
    """ Returns a random set of coordinates that is different from the provided coordinates, 
    within the specified bounds.
    The pad parameter avoids coordinates near the bounds."""
    good = False
    while not good:
        good = True
        c1 = random.randint(pad + 1, size_x - (pad + 1))
        c2 = random.randint(pad + 1, size_y -( pad + 1))
        if cond:
            for c in coords:
                coordset_0 = range(int(c[0]/radius)-1,int(c[0]/radius)+2)
                coordset_1 = range(int(c[1]/radius)-1,int(c[1]/radius)+2)
                #if c1 in coordset_0 and c2 in coordset_1:
                if int(c1/radius) in coordset_0 and int(c2/radius) in coordset_1:
                    good = False
                    break
        else:
            for c in coords:
                if c1==c[0] and c2==c[1]:
                    good = False
                    break
    return (c1,c2)

def extract_neighborhood(x, y, arr, radius):
    """ Returns a 1-d array of the values within a radius of the x,y coordinates given """
    if x < radius or y < radius or x>=480-radius or y>=640-radius:
        return np.ones((radius*2+1,radius*2+1)).ravel()
    return arr[(x - radius) : (x + radius + 1), (y - radius) : (y + radius + 1)].ravel()

def out_extract_neighborhood(x, y, arr, radius,xx1,yy1):
    xx2 = xx1+radius
    yy2 = yy1+radius
    if xx1<0 or yy1<0 or xx2>480 or yy2>640:
        #print("0")
        return np.zeros((radius+2,radius+2)).ravel()
    if xx1>1 and xx1<480-radius-1 and yy1>1 and yy1<640-radius-1:
        #print("m")
        myarr = arr[(xx1- 1) : (xx2 + 1), (yy1- 1) : (yy2 + 1)]
    vec = np.zeros((1,radius+2))
    vec2 = np.zeros((1,radius+1))
    if xx1>=480-radius-1 and yy1>1 and yy1<640-radius-1:
        #print("1")
        myarr = arr[(xx1- 1) : xx2, (yy1- 1) : (yy2 + 1)]
        myarr = np.append(myarr,vec,axis=0)
    if xx1<=1 and yy1>1 and yy1<640-radius-1:
        #print("2")
        myarr = arr[xx1 : (xx2+1), (yy1- 1) : (yy2 + 1)]
        myarr = np.append(vec,myarr,axis=0)
    if yy1>=640-radius-1 and xx1>1 and xx1<480-radius-1:
        #print("3")
        myarr = arr[(xx1- 1) : (xx2+1), (yy1- 1) : yy2]
        myarr = np.append(myarr,vec.T,axis=1)
    if yy1<=1 and xx1<480-radius-1 and xx1>1:
        #print("4")
        myarr = arr[(xx1-1) : (xx2+1), yy1 : (yy2 + 1)]
        myarr = np.append(vec.T,myarr,axis=1)
    if yy1<=1 and xx1<=1:
        #print("5")
        myarr = arr[xx1 : (xx2+1), yy1 : (yy2 + 1)]
        myarr = np.append(vec2.T,myarr,axis=1)
        myarr = np.append(vec,myarr,axis=0)
    if yy1>=640-radius-1 and xx1>=480-radius-1:
        #print("6")
        myarr = arr[(xx1-1) : xx2, (yy1-1) : yy2]
        myarr = np.append(myarr,vec2.T,axis=1)
        myarr = np.append(myarr,vec,axis=0)
    if yy1>=640-radius-1 and xx1<=1:
        #print("7")
        myarr = arr[xx1 : (xx2+1), (yy1-1) : yy2]
        myarr = np.append(myarr,vec2.T,axis=1)
        myarr = np.append(vec,myarr,axis=0)
    if xx1>=480-radius-1 and yy1<=1:
        #print("8")
        myarr = arr[(xx1-1) : xx2, yy1 : (yy2 + 1)]
        myarr = np.append(myarr,vec2,axis=0)
        myarr = np.append(vec.T,myarr,axis=1)
    #print(myarr.shape)
    return myarr.ravel()    
            #return arr[ xx1 : xx2 , yy1 : yy2 ].ravel()

def check_coordinate_validity(x, y, size_x, size_y, pad):
    """ Check if a coordinate is not too close to the image edge """
    return x >= pad and y >= pad and x + pad < size_x and y + pad < size_y

def generate_labeled_data(image_path, annotation, nb_false, radius,cond):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    features,labels = [],[]
    im_array = read_image(image_path)
    # True samples
    for obj in annotation:
        obj = [int(x + .5) for x in obj] #Project the floating coordinate values onto integer pixel coordinates.
        # For some reason the order of coordinates is inverted in the annotation files
        if True:#check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            x1 = int(obj[1]/radius)
            y1 = int(obj[0]/radius)
            print(obj[1],obj[0])
            if obj[1] % radius ==0:
                xx1range = range((x1*radius)-3, (x1*radius)+1)
            elif obj[1] % radius == 1 :
                xx1range = range(x1*radius-2, (x1*radius)+2)
            elif obj[1] % radius == 2:
                xx1range = range(x1*radius-1, (x1*radius)+3)
            else:
                xx1range = range(x1*radius, (x1*radius)+4)
            if obj[0] % radius == 0:
                yy1range = range((y1*radius)-3, (y1*radius)+1)
            elif obj[0] % radius == 1:
                yy1range = range((y1*radius)-2, (y1*radius)+2)
            elif obj[0] % radius == 2:
                yy1range = range((y1*radius)-1, (y1*radius)+3)
            else:
                yy1range = range(y1*radius, (y1*radius)+4)
            for xx1 in xx1range:
                for yy1 in yy1range:
                    features.append(out_extract_neighborhood(obj[1],obj[0],im_array,radius,xx1,yy1))
                    labels.append(1)
            #features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            #labels.append(1)
        if False:
            krange = [obj[0]-4,obj[0],obj[0]+4]
            lrange = [obj[1]-4,obj[1],obj[1]+4]
            for k in krange:
                for l in lrange:
                    if check_coordinate_validity(l,k,im_array.shape[0],im_array.shape[1],radius):
                        #if k!=obj[0] or l!=obj[1]:
                        randn = random.randint(1,9)
                        if randn % 2 == 0:
                            features.append(out_extract_neighborhood(l,k,im_array,radius))
                            labels.append(1)
    # False samples
    for i in range(nb_false):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius,cond)
        x1 = int(c[1]/radius)
        y1 = int(c[0]/radius)
        xx1 = x1*radius
        yy1 = y1*radius
        #print(c[1],c[0])
        features.append(out_extract_neighborhood(c[1],c[0],im_array,radius,xx1,yy1))
        labels.append(0)
    return np.array(labels),np.stack(features,axis=1)

def generate_labeled_set(annotation_array, path, sequence_id_list, radius, nb_false,cond):
    # Generate labeled data for a list of sequences in a given path
    labels,features = [],[]
    for seq_id in sequence_id_list:
        for frame_id in range(1,6):
            d = generate_labeled_data(f"{path}{seq_id}/{frame_id}.png",
                                    annotation_array[seq_id][frame_id],
                                    nb_false,
                                    radius,cond)
            labels.append(d[0])
            features.append(d[1])
    return np.concatenate(labels,axis=0), np.transpose(np.concatenate(features,axis=1))


def generate_labeled_testset(annotation_array, path, sequence_id_list, radius, nb_false,cond):
    # Generate labeled data for a list of sequences in a given path
    labels,features = [],[]
    for seq_id in sequence_id_list:
        for frame_id in range(1,6):
            d = generate_labeled_testdata(f"{path}{seq_id}/{frame_id}.png",
                                    annotation_array[seq_id][frame_id],
                                    nb_false,
                                    radius,cond)
            labels.append(d[0])
            features.append(d[1])
    return np.concatenate(labels,axis=0), np.transpose(np.concatenate(features,axis=1))

def generate_labeled_testdata(image_path, annotation, nb_false, radius,cond):
    """ For one frame and one annotation array, returns a list of labels 
    (1 for true object and 0 for false) and the corresponding features as an array.
    nb_false controls the number of false samples
    radius defines the size of the sliding window (e.g. radius of 1 gives a 3x3 window)"""
    features,labels = [],[]
    im_array = read_image(image_path)
    # True samples
    for obj in annotation:
        obj = [int(x + .5) for x in obj] #Project the floating coordinate values onto integer pixel coordinates.
        # For some reason the order of coordinates is inverted in the annotation files
        if True:#check_coordinate_validity(obj[1],obj[0],im_array.shape[0],im_array.shape[1],radius):
            x1 = int(obj[1]/radius)
            y1 = int(obj[0]/radius)
            #print(obj[1],obj[0])
            xx1 = x1*radius
            yy1 = y1*radius
            features.append(out_extract_neighborhood(obj[1],obj[0],im_array,radius,xx1,yy1))
            labels.append(1)
            #features.append(extract_neighborhood(obj[1],obj[0],im_array,radius))
            #labels.append(1)
        if False:
            krange = [obj[0]-4,obj[0],obj[0]+4]
            lrange = [obj[1]-4,obj[1],obj[1]+4]
            for k in krange:
                for l in lrange:
                    if check_coordinate_validity(l,k,im_array.shape[0],im_array.shape[1],radius):
                        #if k!=obj[0] or l!=obj[1]:
                        randn = random.randint(1,9)
                        if randn % 2 == 0:
                            features.append(out_extract_neighborhood(l,k,im_array,radius))
                            labels.append(1)
    # False samples
    for i in range(nb_false):
        c = random_different_coordinates(annotation,im_array.shape[1],im_array.shape[0],radius,cond)
        x1 = int(c[1]/radius)
        y1 = int(c[0]/radius)
        xx1 = x1*radius
        yy1 = y1*radius
        #print(c[1],c[0])
        features.append(out_extract_neighborhood(c[1],c[0],im_array,radius,xx1,yy1))
        labels.append(0)
    return np.array(labels),np.stack(features,axis=1)


import joblib
import sklearn


def classify_image(im, model, radius):
    n_features=(radius**2+radius*2+(radius+2)*2) #Total number of pixels in the neighborhood
    feat_array=np.zeros((int(im.shape[0]/radius),int(im.shape[1]/radius),n_features))
    for x in range(0,int(im.shape[0]/radius)):
        for y in range(0,int(im.shape[1]/radius)):
            #x1 = int(x/radius)
            #y1 = int(y/radius)
            xx1 = x*radius
            yy1 = y*radius
            feat_array[x,y,:]=out_extract_neighborhood(x,y,im,radius,xx1,yy1)
    all_pixels=feat_array.reshape(int(im.shape[0]/radius)*int(im.shape[1]/radius),n_features)
    #pred_pixels=model.predict(all_pixels).astype(np.bool_)
    pred_pixels=model.predict_proba(all_pixels)#.astype(np.bool_)
    #pred_image=pred_pixels.reshape(im.shape[0],im.shape[1])
    svc_labels_temp = np.where(pred_pixels>0.98)
    svc_labels_f = []
    last = 0 
    #print(svc_labels_temp[0])
    for i,val in enumerate(svc_labels_temp[0]):
        for j in range(last,val):
            svc_labels_f.append(0)
        if svc_labels_temp[1][i]==1:
            svc_labels_f.append(1)
        else:
            svc_labels_f.append(0)
        last = val+1
    for j in range(last,pred_pixels.shape[0]):
        svc_labels_f.append(0)
    predicted_pixels = np.array(svc_labels_f)
    pred_image = predicted_pixels.reshape(int(im.shape[0]/radius),int(im.shape[1]/radius))
    return pred_image

def extract_centroids(pred, bg):
    conn_comp=measure.label(pred, background=bg)
    object_dict=defaultdict(list) #Keys are the indices of the connected components and values are arrrays of their pixel coordinates 
    for (x,y),label in np.ndenumerate(conn_comp):
            if label != bg:
                object_dict[label].append([x,y])
    # Mean coordinate vector for each object, except the "0" label which is the background
    centroids={label: np.mean(np.stack(coords),axis=0) for label,coords in object_dict.items()}
    object_sizes={label: len(coords) for label,coords in object_dict.items()}
    return centroids, object_sizes

def filter_large_objects(centroids,object_sizes, min_size,max_size):
    small_centroids={}
    for label,coords in centroids.items():
            if object_sizes[label] <= max_size and object_sizes[label]>min_size:
                small_centroids[label]=coords
    return small_centroids


def corrange(x,k,dist):
    diff1 = x[1][0]-x[0][0]
    diff2 = x[1][1]-x[0][1]
    diff3 = k[1][0]-k[0][0]
    diff4 = k[1][1]-k[0][1]
    if abs(diff1-diff3)<2 and abs(diff2-diff4)<2:
        if k[0][0]-x[0][0] > (dist*diff1)-5 and k[0][0]-x[0][0] < (dist*diff1)+5 and k[0][1]-x[0][1] > (dist*diff2)-5 and k[0][1]-x[0][1] < (dist*diff2)+5: 
            return True,(diff1+diff3)/2,(diff2+diff4)/2
        else:
            return False,None,None
    else:
        return False,None,None
    
def corrange1(x,k,dist,diff1,diff2):
    cmp1 = k[0]-x[0]
    cmp2 = k[1]-x[1]
    if  cmp1 > (dist*diff1)-5 and cmp1 < (dist*diff1)+5 and cmp2 > (dist*diff2)-5 and cmp2 < (dist*diff2)+5: 
        return True
    else:
        return False


def in_list(arr,lst):
    for l in lst:
        if np.array_equal(arr,l):
            return True
    return False
    


def not_sim(coordlist,newc):
    flg = True
    for c in coordlist:
        if abs(newc[0]-c[0])<3 or abs(newc[1]-c[1])<3:
            flg = False
            break
    return flg

def build_test_set(pr_res,im,sclf,radius):
    pred_pixels = np.zeros((im.shape[0],im.shape[1]))
    ind = np.where(pr_res==1)
    #features,labels = [],[]
    #print(ind[0],ind[1])
    #print(ind[0].shape,ind[1].shape)
    indexx = []
    features=[]
    priorl = []
    for i,val in enumerate(ind[0]):
        for k in range(val*radius,((val+1)*radius)):
            for l in range(ind[1][i]*radius,((ind[1][i]+1)*radius)):
                r = extract_neighborhood(k,l,im,3)
                features.append(r)
                priorl.append([k,l])
    features = np.stack(features, axis=0)
    pred_pixels_md = sclf.predict_proba(features)
    labtemp = np.where(pred_pixels_md[:,1] > 0.945)
    for indexn in labtemp[0]:
        indu = priorl[indexn]
        pred_pixels[indu[0],indu[1]]=1
    return pred_pixels
    

from itertools import product


wnd = joblib.load('wnd1sclf.pkl')
sclf = joblib.load('StackingActUncWH.pkl')
wnd.drop_last_proba=False
sclf.drop_proba_col = False

def final_pr(myseq):
    #sub_list = []
    seq_time = time.time()
    prdlist = []
    for myframe in range(1,6):
        test_image = plt.imread(f"test/{myseq}/{myframe}.png")
        pr_res = classify_image(test_image, wnd, 4)
        predf = build_test_set(pr_res,test_image,sclf,4)
        prdlist.append(predf)
    print('Sequence: '+str(myseq)+' '+str(time.time()-seq_time))
    mlst = []
    for predimg in prdlist:
        centroids,sizes = extract_centroids(predimg,0)
        centroids = filter_large_objects(centroids, sizes,0,20)
        #print(len(centroids.values()))
        x = list(centroids.values())

        mlst.append(x)

    combs = []
    for i in range(0,4):
        combs.append([])
        combs[i].append(list(product(mlst[i],mlst[i+1])))


    #print(time.time()-thet)
    xcombs = []
    for i,x in enumerate(combs):
        xcombs.append([])
        for u in x:
            for k in u:
                if abs(k[0][0]-k[1][0]) < 45 and abs(k[0][1]-k[1][1]) < 50:
                    xcombs[i].append(k)

    res=[[],[],[],[],[]]
    for i in range(0,3):
        for j in range(i+1,4):
            for x in xcombs[i]:
                for k in xcombs[j]:
                    boolr, d1,d2 = corrange(x,k,j-i)
                    if boolr:
                        if not in_list(x[0],res[i]):
                            res[i].append(x[0])
                        if not in_list(x[1],res[i+1]):
                            res[i+1].append(x[1])
                        if not in_list(k[0],res[j]):
                            res[j].append(k[0])
                        if not in_list(k[1],res[j+1]):
                            res[j+1].append(k[1])
                        

    
    combs = []
    for i in range(0,4):
        combs.append([])
        combs[i].append(list(product(res[i],res[i+1])))
   
    xcombs = []
    for i,x in enumerate(combs):
        xcombs.append([])
        for u in x:
            for k in u:
                if abs(k[0][0]-k[1][0]) < 45 and abs(k[0][1]-k[1][1]) < 50:
                    xcombs[i].append(k)

    #combs = []
    mlstX = [[],[],[],[],[]]
    bool_seq = False
    for i in range(0,3):
        for j in range(i+1,4):
            for x in xcombs[i]:
                for k in xcombs[j]:
                    boolr, d1,d2 = corrange(x,k,j-i)
                    if boolr:
                        listc = [c for c in range(0,5) if c!=i and c!=j and c!=i+1 and c!=j+1]
                        for pnum in listc:
                            for y in res[pnum]:
                                if corrange1(x[0],y,pnum-i,d1,d2):
                                    bool_seq=True
                            if not bool_seq:
                                    u1 = x[0][0]+d1*(pnum-i)
                                    u2 = x[0][1]+d2*(pnum-i)
                                    if u1<-0.5:
                                        u1=-0.5
                                    if u2<-0.5:
                                        u2=-0.5
                                    if u1 > 479.5:
                                        u1=479.5
                                    if u2 >639.5:
                                        u2=639.5
                                    added = np.array([u1,u2])
                                    if not_sim(mlstX[pnum],added):
                                        mlstX[pnum].append(added)
                            bool_seq=False
    for j,k in enumerate(res):
        for kk in k:
            mlstX[j].append(np.array(kk))
    sub_list = []
    for i in mlstX:
            sub_list.append(i)
    #print('Filter: '+str(time.time()-seq_time))
    return sub_list

from multiprocessing import Pool
import multiprocessing

def main():
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    
    start_time = time.time()
    print("strt")
    nprocs = multiprocessing.cpu_count()
    print(nprocs)
    p = Pool(nprocs)
    sub_sequence = p.map(final_pr, range(1801,3001))
    p.close()
    #print(sub_sequence)
    sub_list = []
    for u in sub_sequence:
        for lst in u:
            sub_list.append(lst)
    print(time.time()-start_time)
    new_list2 = []
    for i,val in enumerate(sub_list):
        if len(val)>0:
            sub2 = np.concatenate([c[np.array([1,0])].reshape((1,2)) for c in val])
            new_list2.append(sub2.tolist()[0:30])
        else:
            new_list2.append([])
    #submit to file
    sequencerange = range(1801,3001)
    framerange = range(1,6)
    submission=[]
    for s in range(1801,3001):
        #print(s)
        for fr in range(1,6):
            if s in sequencerange:
                submission.append({"sequence_id" : s, 
                                        "frame" : fr, 
                                        "num_objects" : len(new_list2[(s-1801)*5 + fr-1]), 
                                        "object_coords" : new_list2[(s-1801)*5 + fr-1]})
            else:
                submission.append({"sequence_id" : s,
                                        "frame" : fr,
                                        "num_objects" : 0,
                                        "object_coords" : []})
    with open('my_submission1801-3000.json', 'w') as outfile:
        json.dump(submission, outfile)

if __name__ ==  '__main__': 
    main()
