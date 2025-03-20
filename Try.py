from dataloader import DataLoader
from losses_metrics import losses
import numpy as np
BATCH_SIZE = 64


img_true = np.random.randn(10, 256, 256, 1)
img_pred = np.random.randn(10, 256, 256, 1)

losses_class = losses(img_true,img_pred)
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

