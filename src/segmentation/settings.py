import json
import os
"""
The Dataset Paths are respective to the kaggle directory
Check the Drive link in the README.md to access the dataset

"""

PERSON_TRAIN_PATH='./PersonVehicle-Reidentification/person-reid/train_image/train_image'
PERSON_TEST_PATH='./PersonVehicle-Reidentification/person-reid/test_image/test_image'
VEHICLE_TRAIN_PATH='./PersonVehicle-Reidentification/ufpralpr/UFPR-ALPR dataset/training'
VEHICLE_TEST_PATH='./PersonVehicle-Reidentification/ufpralpr/UFPR-ALPR dataset/testing'
RBNR_VEHICLE_PATH='./PersonVehicle-Reidentification/rbnr-datasets/RBNR_Datasets/datasets'
RODSOL_PATH='./PersonVehicle-Reidentification/rodosolalpr/RodoSol-ALPR/images'
OUTPUTIMG_PATH='./PersonVehicle-Reidentification/output/download.png'
MODEL_WEIGHTS='PersonVehicle-Reidentification/output/weights/myvehi_model.keras'
OUTPUT_MASK=".PersonVehicle-Reidentification/Dataset/image_mask"