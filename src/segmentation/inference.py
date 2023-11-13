import os
import numpy as np
import matplotlib.pyplot as plt
from segmentation.dataloader.datagen import DataGenerator
from segmentation.model.architecture import get_model



IMG_SIZE=(256,256,3)

class Inference(DataGenerator):
    
    def __init__(self,model_path,output_path,model=None):
        #Datagenerator as parent class
        super().__init__()
        self.model_path=model_path
        self.output_path=output_path
        #if self.model_path:
        self.model=get_model(IMG_SIZE,2)
        self.model.load_weights(model_path)
        #else:
        #    self.model=model

    def predictions(self,path):
        """
        Args:
            path - Image path
            seg_model- segmentation model
        """
        img=self.read_image(path,target_size=(256,256))
        img=np.array(img)/255.0
        imagee=np.expand_dims(img,axis=0)
        pred=np.argmax(self.model.predict(imagee)[0],axis=2)
        image_rgb= imagee[0].copy()
        image_rgb[pred == 0] = 255

        return img,pred,image_rgb
    
    def plot(self,imagee,pred,image_rgb,fig_dim=(15,15),save_fig=True):
        """
            Args: 
                original image,prediction_mask,background_removed image --> numpy array
                figure dimensions --> Tuple
                save_fig --> boolean
                
        """
        plt.figure(figsize=fig_dim)
        plt.subplot(1,3,1)
        plt.imshow(imagee)
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(image_rgb)
        plt.title("After Background Removal")
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(pred,cmap="gray")
        plt.axis('off')
        plt.title("Predicted Mask")
        
        if save_fig==True:
            plt.savefig(self.output_path)