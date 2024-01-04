from src.segmentation.inference import Inference
import os



inf = Inference(os.path.join(os.getcwd(),"output/weights/res_humanid_model.keras"),os.path.join(os.getcwd(),"output/download.png"))
a,b,img= inf.predictions(os.path.join(os.getcwd(),"output/121.jpg"))
inf.plot(a,b,img)