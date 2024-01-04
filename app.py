import os
from src.segmentation.inference import Inference
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# os.environ["SM_FRAMEWORK"] = "tf.keras"

def main():

    st.title("Segmentation Analysis")
    st.write("Upload an image")
    uploaded_file = st.file_uploader("Choose a file",type=["jpg", "png", "jpeg"])
    IMAGE_DIR = os.path.join(os.getcwd(),"output")
    OUT_PATH = os.path.join(os.getcwd(),"output/download.png")
    WRITEPATH = os.path.join(os.getcwd(),"output/output.jpg")
    if uploaded_file is not None:
            # To read file as bytes:
            bytes_data = uploaded_file.getvalue()

            IMAGE_PATH = os.path.join(IMAGE_DIR,uploaded_file.name)
            with open(IMAGE_PATH,"wb") as f:
                f.write(uploaded_file.getbuffer())
    
        
                inf = Inference(os.path.join(os.getcwd(),"output/weights/res_humanid_model.keras"),OUT_PATH)               
                a,b,c = inf.predictions(IMAGE_PATH)
                inf.plot(a,b,c)
                #cv2.imwrite(WRITEPATH , c)
                image=Image.open(OUT_PATH)
                st.image(image, caption="Result", use_column_width=True)
                
                             


if __name__ == "__main__":
     main()