from losses_metrics import losses
from WAT_stacked_uents import StackedUNets
from dataloader import DataLoader

import math
import pandas as pd
import tensorflow as tf
import argparse
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import plot_model

def exponential_lr(epoch, LEARNING_RATE):
    if epoch < 10:
        return LEARNING_RATE
    else:
        return LEARNING_RATE * math.exp(0.1 * (10 - epoch)) # lr decreases exponentially by a factor of 10
    
def total_loss(y_true, y_pred):
    perceptual = losses().perceptual_loss(y_true, y_pred)
    ssim = losses().ssim_loss(y_true, y_pred)
    
    scaled_perceptual = (perceptual*0.05807468295097351)
    adjusted_perceptual = (scaled_perceptual+0.009354699403047562)
    
    total = (ssim+adjusted_perceptual)/2
    return total

def main():
        print('---------------------------------')
        print('Model Training ...')
        print('---------------------------------')
        HEIGHT = 256
        WIDTH = 256
        LEARNING_RATE = 0.001
        BATCH_SIZE = 64
        NB_EPOCH = 10
        parser = argparse.ArgumentParser(description="Process a variable.")
        parser.add_argument('-d', type=str, nargs='?', default=None, help="The variable to process (optional)")
        args = parser.parse_args()

        dataset_path = args.d
        data_ids = {"/kaggle/input/mmmai-simulated-data/ds004795-download":"Motion_Simulated","/kaggle/input/mmmai-regist-data/MR-ART-Regist":"Motion","/kaggle/input/brats-motion-data/new_Brats_motion_data":"BraTS"}
        model = StackedUNets().Correction_Multi_input(HEIGHT, WIDTH)
        # load("/kaggle/input/wavelet-style/WAT_style_stacked_05_val_loss_0.0595.h5")
#         print(model.summary())
        
        # model = load_model("/kaggle/input/stackedunet-regist-final-wavtf-normal-dataset/stacked_model_45_val_loss_0.1759.h5",
                           # custom_objects={'total_loss':total_loss, 'ssim_score': ssim_score, 'psnr':psnr, 'K':K})
        
        WEIGHTS_PATH = "/kaggle/working/"
        csv_logger = CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv', append=True, separator=',')
        reduce_lr = LearningRateScheduler(exponential_lr)
        
        model.compile(loss=total_loss, optimizer=Adam(learning_rate=LEARNING_RATE),
                      metrics=[losses().ssim_score, 'mse', losses.psnr])
        
        checkpoint_path = '/kaggle/working/WAT_style_stacked_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
        model_checkpoint = ModelCheckpoint(checkpoint_path,
                                   monitor='val_loss',
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='min',
                                   verbose=1)
        data_loader_Motion_Simulated = DataLoader(
            data_path=dataset_path,
            split_ratio=[0.7, 0.2, 0.1],
            view="Axial",
            data_id=data_ids[dataset_path],
            crop=False,
            batch_size=BATCH_SIZE,
            split_json_path=None
        )

        data_loader_Motion_Simulated.split_data()
            

        train_dataset = data_loader_Motion_Simulated.generator('train')
        test_dataset_Motion_Simulated = data_loader_Motion_Simulated.generator('test')
        validation_dataset= data_loader_Motion_Simulated.generator('validation') 
        for (mb,m,ma),f in train_dataset:
             print(m.shape)
             break
        hist = model.fit(train_dataset,
                         epochs=NB_EPOCH,
                         verbose=1,
                         validation_data=validation_dataset,
                         callbacks=[csv_logger, reduce_lr, model_checkpoint])


if __name__ == "__main__":
    main()