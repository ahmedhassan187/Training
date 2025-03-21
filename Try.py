from dataloader import DataLoader
from losses_metrics import losses
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from adapti_multi_loss_normalization import multi_loss
BATCH_SIZE = 64



img_true = np.random.randn(10, 256, 256, 1)
img_pred = np.random.randn(10, 256, 256, 1)

losses_class = losses(img_true,img_pred)
multi = multi_loss()
print(f"Psnr for the Random images is = {losses_class.psnr()}")
print(f"SSIM_Loss for the Random images is = {losses_class.ssim_loss()}")
print(f"SSIM_socre for the Random images is = {losses_class.ssim_score()}")
print(f"Per_loss for the Random images is = {losses_class.perceptual_loss()}")
data_loader_Motion_Simulated = DataLoader(
    data_path="/kaggle/input/mmmai-simulated-data/ds004795-download",
    split_ratio=[0.7, 0.2, 0.1],
    view="Axial",
    data_id="Motion_Simulated",
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
    losses_class(free,motion)
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
base_losses = np.load("/kaggle/working/base_losses_.npy")
comb_losses = np.load("/kaggle/working/comb_losses_.npy")

# ✅ Adaptive loss normalization
try:
    total_loss, w_comb, b_comb = multi.adaptive_multi_loss_normalization(base_losses, comb_losses)
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Weight (w_comb): {w_comb:.4f}")
    print(f"Bias (b_comb): {b_comb:.4f}")
except ValueError as e:
    print(f"Error: {e}")
