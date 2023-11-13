import glob
import os
from segmentation.settings import VEHICLE_TRAIN_PATH,RBNR_VEHICLE_PATH,RODSOL_PATH,PERSON_TRAIN_PATH,OUTPUT_MASK
from segmentation.dataloader.datagen import DataGenerator
import random
import numpy as np

datagen=DataGenerator()

def listofFiles():
    vehiclefiles=[]
    rodsol_files=[]
    rbnr_files=[]
    for tracks in os.listdir(VEHICLE_TRAIN_PATH):
        filepath=os.path.join(VEHICLE_TRAIN_PATH,tracks)
        vehiclefiles.extend(glob.glob(f"{filepath}/*.png"))
    for tracks in os.listdir(RBNR_VEHICLE_PATH):
        filepath=os.path.join(RBNR_VEHICLE_PATH,tracks)
        rbnr_files.extend(glob.glob(f"{filepath}/*.JPG"))
    for tracks in os.listdir(RODSOL_PATH):
        #print(tracks)
        if "motorcycles-me" not in tracks and "motorcycles-br" not in tracks :
            filepath=os.path.join(RODSOL_PATH,tracks)
            rodsol_files.extend(glob.glob(f"{filepath}/*.jpg"))
        
    personfiles=glob.glob(f"{PERSON_TRAIN_PATH}/*.jpg")
    all_files=vehiclefiles + personfiles +rodsol_files + rbnr_files
    random.shuffle(all_files)
    return all_files


def generate_mask():
    ls=listofFiles()
    for i,filename in enumerate(ls):
        readimg=datagen.read_image(filename,target_size=(256,256))
        prcssimg=datagen.process(readimg)
        outimg=datagen.generate_mask(prcssimg,readimg.size)
        name=os.path.join(OUTPUT_MASK,str(i)+".npy")
        #print(outimg.shape,np.unique(outimg))
        np.save(name,outimg)
    #break