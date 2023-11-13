import torch
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from torchvision import transforms



CAR=[7,6,14]
PERSON=[15]
ALL=[0,1]
PALLETE = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
COLORS = torch.as_tensor([i for i in range(21)])[:, None] * PALLETE
COLORS = (COLORS % 255).numpy().astype("uint8")

class DataGenerator:
    def __init__(self,batch_size=16):
        #training image directory
        self.batch_size=batch_size
        self.model=torch.hub.load('pytorch/vision:v0.10.0', "deeplabv3_resnet101", pretrained=True)
        self.model.eval()
        
    def read_image(self,imgPath,target_size=(256,256),fill_color=(255, 255, 255)):
        
        input_image = Image.open(imgPath)
        input_image = input_image.convert("RGB")
        original_size = input_image.size
        #print(original_size)
        if "UFPR-ALPR dataset" in imgPath:
            input_image=input_image.crop((0,0,original_size[0],650))
    
        # Calculate the padding size
        width_padding = max(0, target_size[0] - original_size[0])
        height_padding = max(0, target_size[1] - original_size[1])

        # Calculate the padding values (left, top, right, bottom)
        padding = (width_padding // 2, height_padding // 2, (width_padding + 1) // 2, (height_padding + 1) // 2)

        # Resize the image to the target size and pad it
        new_image = input_image.resize(target_size)
        return new_image
    
    def process(self,img):

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        return input_batch
    def post_processing(self,img):
        # 
        for val in CAR:
            img=np.where(img==val,1,img)
        img=np.where(img==15,1,img)
        mask = np.isin(img, ALL , invert=True)
        img[mask] = 0
        return img
        
    #generate mask
    def generate_mask(self,img,shape):
        if torch.cuda.is_available():
            img = img.to('cuda')
            self.model.to('cuda')

        with torch.no_grad():
            output = self.model(img)['out'][0]
        output = output.argmax(0)
        out = Image.fromarray(output.byte().cpu().numpy()).resize(shape)
        #print(output)
        out.putpalette(COLORS)
        out=self.post_processing(np.array(out))
        return out
    #forward
    def batch_forward(self,ls,ls2,start_indx):
        image,mask=[],[]
        out="/kaggle/working/image_mask"
        for file in ls2[start_indx:start_indx+self.batch_size]:
            indx=int(file[:-4])
            #print(indx)
            img=self.read_image(ls[indx],target_size=(256,256))
            msk=np.load(os.path.join(out,file))
            img=np.array(img)/255.0
            image.append(img)
            mask.append(msk)
        image=np.stack(image,axis=0)
        mask=np.stack(mask,axis=0)
        #mask=to_categorical(mask,num_classes=2)
        return image,mask
    def forward(self,ls,start_indx):
        image,mask=[],[]
        for file in ls[start_indx:start_indx+self.batch_size]:
            img=self.read_image(file)
            img=np.array(img)/255.0
            processed_img=self.process(img)
            msk=self.generate_mask(processed_img,img.size)
            image.append(np.array(img))
            mask.append(msk)
        image=np.stack(image,axis=0)
        mask=np.stack(mask,axis=0)
        mask=to_categorical(mask,num_classes=2)
        #print(mask.shape)
        return image,mask
 
       
        