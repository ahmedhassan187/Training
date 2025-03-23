import math
import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from losses_metrics import losses
from WAT_stacked_uents import StackedUNets
from dataloader import DataLoader
from adapti_multi_loss_normalization import multi_loss

# Initialize loss object
loss_obj = losses()
multi = multi_loss()

# Dataset paths and identifiers
data_ids = {
    "/kaggle/input/mmmai-simulated-data/ds004795-download": "Motion_Simulated",
    "/kaggle/input/mmmai-regist-data/MR-ART-Regist": "Motion",
    "/kaggle/input/brats-motion-data/new_Brats_motion_data": "BraTS"
}

# Default values for w_comb and b_comb
w_comb = 1  # Default weight
b_comb = 0  # Default bias

def exponential_lr(epoch, LEARNING_RATE):
    """Learning rate scheduler function."""
    if epoch < 10:
        return LEARNING_RATE
    else:
        return LEARNING_RATE * math.exp(0.1 * (10 - epoch))  # lr decreases exponentially by a factor of 10

def total_loss(w_comb, b_comb):
    """Wrapper function to create a total_loss function with dynamic w_comb and b_comb."""
    def loss(y_true, y_pred):
        """Custom loss function combining perceptual and SSIM losses."""
        perceptual = loss_obj.perceptual_loss(y_true, y_pred)
        ssim = loss_obj.ssim_loss(y_true, y_pred)
        # print(f"W_comb is {w_comb} and b_comb is {b_comb}")
        
        scaled_perceptual = perceptual * w_comb
        adjusted_perceptual = scaled_perceptual + b_comb
        
        total = (ssim + adjusted_perceptual) / 2
        return total
    return loss

def load_data_loader(dataset_path, batch_size):
    """Load and split data using DataLoader."""
    try:
        data_loader = DataLoader(
            data_path=dataset_path,
            split_ratio=[0.7, 0.2, 0.1],
            view="Axial",
            data_id=data_ids[dataset_path],
            crop=False,
            batch_size=batch_size,
            split_json_path=None
        )
        data_loader.split_data()
        return data_loader
    except KeyError:
        raise ValueError(f"Dataset path {dataset_path} not found in data_ids.")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def wat_unet(dataset_path):
    """Train the WAT U-Net model."""
    print('---------------------------------')
    print('Model Training ...')
    print('---------------------------------')
    
    HEIGHT = 256
    WIDTH = 256
    LEARNING_RATE = 0.001
    BATCH_SIZE = 10
    NB_EPOCH = 10

    try:
        # Initialize model
        model = StackedUNets().Correction_Multi_input(HEIGHT, WIDTH)
        
        # Load data
        data_loader = load_data_loader(dataset_path, BATCH_SIZE)
        train_dataset = data_loader.generator('train')
        validation_dataset = data_loader.generator('validation')

        # Compute base_losses and comb_losses
        base_losses = []
        comb_losses = []
        for (motion_before, motion, motion_after), free in tqdm(train_dataset):
            # print(f"shape of motion is {motion.shape} and the shape of free is {free.shape}")
            if free.shape[0] != 10 or motion.shape[0] != 10:
                print(f"Wrong shape detected free shape  =  {free.shape} while motion shape = {motion.shape}")
                # continue
            ssim = loss_obj.ssim_loss(free, motion)  # Tensor
            # ssim = tf.math.reduce_mean(ssim)
            perceptual = loss_obj.perceptual_loss(free, motion)  # Tensor
            base_losses.append(ssim)
            comb_losses.append(perceptual)

        # Convert lists to NumPy arrays
        base_losses = tf.stack(base_losses).numpy()
        comb_losses = tf.stack(comb_losses).numpy()

        # Save arrays to disk (optional)
        np.save("base_losses_.npy", base_losses)
        np.save("comb_losses_.npy", comb_losses)

        # Calculate adaptive weights and biases
        try:
            _, w_comb, b_comb = multi.adaptive_multi_loss_normalization(base_losses, comb_losses)
            print(f"Weight (w_comb): {w_comb:.4f}")
            print(f"Bias (b_comb): {b_comb:.4f}")
        except ValueError as e:
            print(f"Error: {e}")
            # Fallback to default values if adaptive normalization fails
            w_comb = 1
            b_comb = 0

        # Compile model with updated total_loss function
        model.compile(
            loss=total_loss(w_comb, b_comb),  # Pass w_comb and b_comb to total_loss
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=[loss_obj.ssim_score, 'mse', loss_obj.psnr]
        )

        # Callbacks
        WEIGHTS_PATH = "/kaggle/working/"
        csv_logger = CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv', append=True, separator=',')
        reduce_lr = LearningRateScheduler(lambda epoch: exponential_lr(epoch, LEARNING_RATE))
        checkpoint_path = f'{WEIGHTS_PATH}WAT_style_stacked_{{epoch:02d}}_val_loss_{{val_loss:.4f}}.h5'
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        # Train model
        history = model.fit(
            train_dataset,
            epochs=NB_EPOCH,
            verbose=1,
            validation_data=validation_dataset,
            callbacks=[csv_logger, reduce_lr, model_checkpoint]
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")

def main():
    """Main function to parse arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train WAT U-Net model.")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to the dataset.")
    args = parser.parse_args()

    if args.dataset in data_ids:
        wat_unet(args.dataset)
    else:
        print(f"Invalid dataset path: {args.dataset}. Available paths are: {list(data_ids.keys())}")

if __name__ == "__main__":
    main()