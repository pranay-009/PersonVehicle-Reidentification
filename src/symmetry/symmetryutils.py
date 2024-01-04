import matplotlib.pyplot as plt
import numpy as np
from image_similarity_measures.evaluate import evaluation
from image_similarity_measures.quality_metrics import ssim,fsim,rmse,uiq,issm
from torch.nn import CosineSimilarity
import cv2
from PIL import Image
class Symmetry(SiameseNetwork):

    def __init__(self,threshold,params,height ,width ,channels,model=None):
        super().__init__(height,width,channels)
        #self.model=model
        self.cos=CosineSimilarity(dim=1, eps=1e-6)
        self.threshold=threshold
        self.patch_size=params["PATH"]
        self.stride=params["STRIDE"]

    def read_image(self,path,type_format="PIL"):
        
        image=Image.open(path)
        if type_format!="PIL":
            image=np.array(image)
        return image
    
    
    def generate_patch(self,image):
        
        patches = []
        height, width, _ = image.shape
        patch_height = height
        patch_width = 96 #96
        stride = 32 #32
        
        for x in range(0, width, stride):
            patch = image[:, x:x+patch_width]
            
            if patch.shape[1] < patch_width:
                padded_patch = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
                padded_patch[:, :patch.shape[1]] = patch
                patch = padded_patch

            patches.append(patch)
        patch_array = np.array(patches)
        return patch_array
    def structural_similarity(self,image1,image2):
        
        return ssim(image1,image2)

    def feature_based_similarity(self,feature1,feature2):
        
        return fsim(feature1,feature2)
    
    def get_embeddings(self,image):
        siames_net=self.Architecture()
        out = siames_net.predict(np.expand_dims(image,axis=0))
        return out
        
    def detect_contours(self,image):
        seg_value = 1

        if image is not None:
            np_seg = np.array(image)
            segmentation = np.where(np_seg == seg_value)

            # Bounding Box
            bbox = 0, 0, 0, 0
            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))

                bbox = x_min, x_max, y_min, y_max
             
        else:
            print("Error: Segmentation image could not be read or is empty.")
        return cv2.rectangle(image, (x_min, y_min), (x_max, y_max),(0, 255, 0), 2)
    
    def check_similarity(self,feature1,feature2):
        
        #check feature similarity
        feature1,feature2 = np.array(feature1[0]),np.array(feature2[0])
        output=dot(feature1, feature2)/(norm(feature1)*norm(feature2))
        
        return output
    
    def symmetry_matrix(self,images,similarity_measure="ssim"):
        n=len(images)
        mat=np.zeros((n,n))
        
        for i in range(n):
            for j in range(n):
                if similarity_measure == "ssim":
                    mat[i][j]=self.structural_similarity(images[i],images[j])
                elif similarity_measure == "fsi":
                    mat[i][j]=self.feature_based_similarity(images[i],images[j])
                elif similarity_measure == "embedding":
                    feat1,feat2=self.get_embeddings(images[i]),self.get_embeddings(images[j])
                    mat[i][j]=self.check_similarity(feat1,feat2)
        return mat
    
    def symmetry_check(self,images):
        
        n=len(images)-2
        lst=[]
        for i in range(n-1):
            feat1,feat2=self.get_embeddings(images[i]),self.get_embeddings(images[i+1])
            measure = self.check_similarity(feat1,feat2)
            if measure >= self.threshold:
                print(measure)
                image=cv2.hconcat([images[i][:,:self.stride],images[i+1]])
                lst.append(image)
        
        return lst
            
                            
            
    def plot(self,images):
        ROW,COL=2,len(images)//2
        fig, axes = plt.subplots(nrows=ROW, ncols=COL, figsize=(12, 6))
        axes = axes.flatten()
        for i, (image, ax) in enumerate(zip(images, axes)):
            ax.imshow(image, cmap='gray')  # Change cmap if your images are not grayscale
            ax.set_title(f'Image {i + 1}')
            ax.axis('on')  # Show axis labels
            ax.set_xticks([])  # Hide x-axis ticks
            ax.set_yticks([])  # Hide y-axis ticks
            
        plt.tight_layout()
        plt.show()

        
        