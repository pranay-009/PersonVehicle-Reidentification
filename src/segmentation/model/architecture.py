import tensorflow as tf

def encoder(prev_block,filters):
    """
    Args:
        Description: Its an Encoder Module of the Residual Unet
        prev_block :contains the previous block layer
        filters: number of channels
    
        returns encoder block
    
    """
    x = tf.keras.layers.Activation("relu")(prev_block)
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

    return x
def encoder_residual_block(filters,prev_block):
    """
        Args:
        Description: Its an Residual Encoder Module of the Residual Unet
        prev_block :contains the previous block layer
        filters: number of channels
    
        returns block
    
    """
    residual = tf.keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
                prev_block)
    return residual

def decoder(prev_block,filters,skip_block):
    """
        Args:
        Description: Its an Decoder Module of the Residual Unet
        prev_block :contains the previous block layer
        filters: number of channels
        skip_block : skip connections that joins the encoder to decoder of unet
    
        returns block
    
    """

    x = tf.keras.layers.Activation("relu")(prev_block)
    x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = tf.keras.layers.concatenate([x,skip_block])
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.UpSampling2D(2)(x)

    return x

def decoder_residual(filters,prev_block):
    """
        Args:
        Description: Its an Residual decoder Module of the Residual Unet
        prev_block :contains the previous block layer
        filters: number of channels
    
        returns block
    
    """
    residual = tf.keras.layers.UpSampling2D(2)(prev_block)
    residual = tf.keras.layers.Conv2D(filters, 1, padding="same")(residual)

    return residual

def get_model(img_size, num_classes):
    """
        Args:
        Description: Residual Unet Architecture
        img_size : (None,None,3)
        num_classes: number of classes for segmentations
    
        returns block
    
    """
    inputs = tf.keras.Input(shape=img_size)

    x1 = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("relu")(x1)

    previous_block_activation = x1  
    #encoder
    #for filters in [64, 128, 256]:
    x22=encoder(x1,64)
    res1=encoder_residual_block(64,previous_block_activation)
    x2= tf.keras.layers.add([x22, res1])  
    previous_block_activation = x2  
    
    x33=encoder(x2,128)
    res2=encoder_residual_block(128,previous_block_activation)
    x3= tf.keras.layers.add([x33, res2])  
    previous_block_activation = x3
    
    x44=encoder(x3,256)
    res3=encoder_residual_block(256,previous_block_activation)
    x4= tf.keras.layers.add([x44, res3])  
    previous_block_activation = x4
    
    
    x=x4
    c1=decoder(x,256,x44)
    res_dec1=decoder_residual(256,previous_block_activation)
    c11 = tf.keras.layers.add([c1, res_dec1])  # Add back residual
    previous_block_activation = c11
    
    c2=decoder(c11,128,x33)
    res_dec2=decoder_residual(128,previous_block_activation)
    c22 = tf.keras.layers.add([c2, res_dec2])  # Add back residual
    previous_block_activation = c22

    
    c3=decoder(c22,64,x22)
    res_dec3=decoder_residual(64,previous_block_activation)
    c33 = tf.keras.layers.add([c3, res_dec3])  # Add back residual
    previous_block_activation = c33
    
    c4=decoder(c33,32,x1)
    res_dec4=decoder_residual(32,previous_block_activation)
    c44 = tf.keras.layers.add([c4, res_dec4])  # Add back residual
    previous_block_activation = c44
    ### [Second half of the network: upsampling inputs] ###
    outputs = tf.keras.layers.Conv2D(num_classes,3, activation="softmax", padding="same")(
        c44
    )

    # Define the model
    model = tf.keras.Model(inputs, outputs)
    return model

