from dataloader import DataLoader
from losses_metrics import losses
from adapti_multi_loss_normalization import multi_loss
from WAT_stacked_uents import stacked_unets

import math
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import plot_model


BATCH_SIZE = 64
# argument parser to select the dataset we will work on 
parser = argparse.ArgumentParser(description="Process a variable.")
parser.add_argument('-d', type=str, nargs='?', default=None, help="The variable to process (optional)")
args = parser.parse_args()

dataset_path = args.d
# data_pathes = {"mr-sim":"/kaggle/input/mmmai-simulated-data/ds004795-download","mr":"/kaggle/input/mmmai-regist-data/MR-ART-Regist","brats":"/kaggle/input/brats-motion-data/new_Brats_motion_data"}
data_ids = {"/kaggle/input/mmmai-simulated-data/ds004795-download":"Motion_Simulated","/kaggle/input/mmmai-regist-data/MR-ART-Regist":"Motion","/kaggle/input/brats-motion-data/new_Brats_motion_data":"BraTS"}

# if args.variable  not in list(data_pathes.keys()):
    # print(f"The  dataset {args.variable} isn't supported ")
    # dataset_path = "mr-sim"
# else:
    # print("seleceted dataset done")
    # dataset_path = data_pathes[args.variable]
    # data_id = data_ids[args.variable]

Model = StackedUNets.Correction_Multi_input(256,256)
Model.compile(loss=losses.ssim_loss(), optimizer=Adam(learning_rate=LEARNING_RATE),
                      metrics=[ssim_score, 'mse', psnr])
        
checkpoint_path = '/kaggle/working/WAT_style_stacked_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
model_checkpoint = ModelCheckpoint(checkpoint_path,
                                   monitor='val_loss',
                                   save_best_only=False,
                                   save_weights_only=False,
                                   mode='min',
                                   verbose=1)
        
img_true = np.random.randn(10, 256, 256, 1)
img_pred = np.random.randn(10, 256, 256, 1)
hist = Model.fit(img_true,img_pred,
                         epochs=NB_EPOCH,
                         verbose=1,
                         validation_data=validation_dataset,
                         callbacks=[csv_logger, reduce_lr, model_checkpoint])
losses_class = losses(img_true,img_pred)
multi = multi_loss()
print(f"Psnr for the Random images is = {losses_class.psnr()}")
print(f"SSIM_Loss for the Random images is = {losses_class.ssim_loss()}")
print(f"SSIM_socre for the Random images is = {losses_class.ssim_score()}")
print(f"Per_loss for the Random images is = {losses_class.perceptual_loss()}")
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
    

train_dataset_Motion_Simulated = data_loader_Motion_Simulated.generator('train')
test_dataset_Motion_Simulated = data_loader_Motion_Simulated.generator('test')
validation_dataset_Motion_Simulated = data_loader_Motion_Simulated.generator('validation')


base_losses = []
comb_losses = []

for (motion_before, motion, motion_after), free in tqdm(train_dataset_Motion_Simulated):
    losses_class.update_values(free,motion)
    ssim = losses_class.ssim_loss()# Tensor
    ssim = tf.math.reduce_mean(ssim)
    perceptual = losses_class.perceptual_loss()  # Tensor
    
    # print(f"SSIM shape: {ssim.shape[0]}")
    # print(f"Perceptual shape: {perceptual.shape}")
    # if ssim.shape[0] != 64:
        # print("Error here")
        # print(f"The shape of motion batch{motion.shape}")
        # continue 
        
    base_losses.append(ssim)
    comb_losses.append(perceptual)
# print(len(base_losses))
# Convert all at once — now losses become NumPy arrays
base_losses = tf.stack(base_losses).numpy()
comb_losses = tf.stack(comb_losses).numpy()

# print(comb_losses)
# ✅ Save arrays to disk
np.save("base_losses_.npy", base_losses)
np.save("comb_losses_.npy", comb_losses)

# ✅ Load later like this (if needed):
base_losses = np.load("/kaggle/working/Training/base_losses_.npy")
comb_losses = np.load("/kaggle/working/Training/comb_losses_.npy")

# ✅ Adaptive loss normalization
try:
    total_loss, w_comb, b_comb = multi.adaptive_multi_loss_normalization(base_losses, comb_losses)
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Weight (w_comb): {w_comb:.4f}")
    print(f"Bias (b_comb): {b_comb:.4f}")
except ValueError as e:
    print(f"Error: {e}")
