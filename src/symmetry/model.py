#from tensorflow.keras.applications import VGG16
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf


class SiameseNetwork:
    def __init__(self, height : int,width : int , channels : int):
        self.height = height
        self.width = width
        self.channel = channels

        input_shape = (height,width, channels)
        input_tensor = Input(shape=input_shape)
        self.vgg = tf.keras.applications.vgg19.VGG19(
                                                    include_top=False,
                                                    weights='imagenet',
                                                    input_tensor=input_tensor,
                                                    )
        
    
    def Architecture(self):

        model = Sequential() 
        for layer in self.vgg.layers:
            model.add(layer)
        model.add(Flatten())
        
        return model
        
    def compilation(self,model):
        
        pass
    
    def train(self,*args):
        
        pass
                