import numpy as np
import wandb
import time
import tqdm
import tensorflow as tf
import random
from tensorflow.keras.utils import to_categorical
from segmentation.dataloader.datagen import DataGenerator
from segmentation.model.architecture import get_model
from segmentation.dataloader.mask import listofFiles
from segmentation.model.loss import  WeightedCategoricalCrossentropy
from segmentation_models.losses import bce_jaccard_loss,jaccard_loss
from tensorflow.keras.optimizers.legacy import Adam
from segmentation_models.metrics import iou_score,f1_score,precision,recall


LEARNING_RATE = 1e-4
EPOCHS=10
metrics=[iou_score,f1_score,precision,recall,'accuracy']
class_weights=[0.1,0.9]
model=get_model(num_classes=2)

weighted_loss = WeightedCategoricalCrossentropy(class_weights)

model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=metrics
)
ls= listofFiles()
def log_evaluate_batch(imag ,msks,test_table,ids):
    img_id = ids
    for img,maask in zip(imag,msks):
        
        #eval_loss,eval_iouscore,eval_f1score,eval_preci,eval_recal,eval_acc=model.evaluate(np.expand_dims(img,axis=0),np.expand_dims(maask,axis=0))
        pred=np.argmax(model.predict(np.expand_dims(img,axis=0))[0],axis=2)
        image_rgb= img.copy()
        image_rgb[pred == 1] = 255
        test_table.add_data(str(img_id), wandb.Image(img), wandb.Image(image_rgb),wandb.Image(pred))
        img_id+=1
        #print("evaluation")
        #print(eval_loss,eval_iouscore,eval_f1score,eval_preci,eval_recal,eval_acc)
    
    return img_id
def train(batch_size,epochs,train_list,val_list):

    data = DataGenerator(batch_size)
    eval_dataset=DataGenerator(batch_size=3)
    columns=["id","image","background_removed","mask"]
    test_table = wandb.Table(columns=columns)
    x=1
    #try:
    for epoch in range(1,epochs+1):
        train_loss=0
        val_loss=0
        start_indx=0
        loss,iouscore,f1score,preci,recal,accuracy=0,0,0,0,0,0
        batch_step=int(len(train_list)/batch_size)
        start_time = time.perf_counter()
        for i in tqdm(range(batch_step)):
            img,mask=data.batch_forward(ls,train_list,i*batch_size)
            #print(np.unique(mask))
            mask=to_categorical(mask,num_classes=2)
            model.trainable=True
            #mask=np.expand_dims(mask,axis=3)
            #print(mask.shape)
            with tf.device('/GPU:0'):

                loss,iouscore,f1score,preci,recal,_=model.train_on_batch(img,mask)
                #print("ok")
        end_time = time.perf_counter()
        eval_img,eval_mask=eval_dataset.batch_forward(ls,val_list,random.randint(0,len(val_list)-1))
        eval_mask=to_categorical(eval_mask,num_classes=2)
        #x=log_evaluate_batch(eval_img,eval_mask,test_table,x)
        #x+=1
        #wandb.log({"test_predictions" : test_table})
        #print(f"Training batch time: {end_time-start_time:.3f} seconds, ")
        print('epoch [%.2f] loss[%.3f] iouscore[%.3f] f1score[%.3f] preci[%.3f] recall[%.3f]' % (epoch,loss,iouscore,f1score,preci,recal)   )
        #wandb.log({"epoch" : epoch, "loss" : loss,"iou": iouscore, "f1": f1score, "preci": precision, "recall": recal })

    #wandb.finish()


            
            