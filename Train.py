import math
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from losses_metrics import losses
from WAT_stacked_uents import StackedUNets
from dataloader import DataLoader

# Initialize loss object
loss_obj = losses()

# Dataset paths and identifiers
data_ids = {
    "/kaggle/input/mmmai-simulated-data/ds004795-download": "Motion_Simulated",
    "/kaggle/input/mmmai-regist-data/MR-ART-Regist": "Motion",
    "/kaggle/input/brats-motion-data/new_Brats_motion_data": "BraTS"
}

def exponential_lr(epoch, LEARNING_RATE):
    """Learning rate scheduler function."""
    if epoch < 10:
        return LEARNING_RATE
    else:
        return LEARNING_RATE * math.exp(0.1 * (10 - epoch))  # lr decreases exponentially by a factor of 10

def total_loss(y_true, y_pred):
    """Custom loss function combining perceptual and SSIM losses."""
    perceptual = loss_obj.perceptual_loss(y_true, y_pred)
    ssim = loss_obj.ssim_loss(y_true, y_pred)
    
    scaled_perceptual = perceptual * 0.05807468295097351
    adjusted_perceptual = scaled_perceptual + 0.009354699403047562
    
    total = (ssim + adjusted_perceptual) / 2
    return total

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
    BATCH_SIZE = 5
    NB_EPOCH = 10

    try:
        # Initialize model
        model = StackedUNets().Correction_Multi_input(HEIGHT, WIDTH)
        
        # Compile model
        model.compile(
            loss=total_loss,
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

        # Load data
        data_loader = load_data_loader(dataset_path, BATCH_SIZE)
        train_dataset = data_loader.generator('train')
        validation_dataset = data_loader.generator('validation')

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



# from losses_metrics import losses
# from WAT_stacked_uents import StackedUNets
# from dataloader import DataLoader

# import math
# import pandas as pd
# import tensorflow as tf
# import argparse
# from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import model_from_json, load_model
# from tensorflow.keras.utils import plot_model

# loss_obj = losses()

# data_ids = {"/kaggle/input/mmmai-simulated-data/ds004795-download":"Motion_Simulated","/kaggle/input/mmmai-regist-data/MR-ART-Regist":"Motion","/kaggle/input/brats-motion-data/new_Brats_motion_data":"BraTS"}
# def exponential_lr(epoch, LEARNING_RATE):
#     if epoch < 10:
#         return LEARNING_RATE
#     else:
#         return LEARNING_RATE * math.exp(0.1 * (10 - epoch)) # lr decreases exponentially by a factor of 10
    
# def total_loss(y_true, y_pred):
#     perceptual = loss_obj.perceptual_loss(y_true, y_pred)
#     ssim = loss_obj.ssim_loss(y_true, y_pred)
    
#     scaled_perceptual = (perceptual*0.05807468295097351)
#     adjusted_perceptual = (scaled_perceptual+0.009354699403047562)
    
#     total = (ssim+adjusted_perceptual)/2
#     return total
# def main():
#         parser = argparse.ArgumentParser(description="Process a variable.")
#         parser.add_argument('-d', type=str, nargs='?', default=None, help="The variable to process (optional)")
#         args = parser.parse_args()
#         dataset_path = args.d
# def wat_unet():
#         print('---------------------------------')
#         print('Model Training ...')
#         print('---------------------------------')
#         HEIGHT = 256
#         WIDTH = 256
#         LEARNING_RATE = 0.001
#         BATCH_SIZE = 5
#         NB_EPOCH = 10


#         model = StackedUNets().Correction_Multi_input(HEIGHT, WIDTH)
#         # load("/kaggle/input/wavelet-style/WAT_style_stacked_05_val_loss_0.0595.h5")
# #         print(model.summary())
        
#         # model = load_model("/kaggle/input/stackedunet-regist-final-wavtf-normal-dataset/stacked_model_45_val_loss_0.1759.h5",
#                            # custom_objects={'total_loss':total_loss, 'ssim_score': ssim_score, 'psnr':psnr, 'K':K})
        
#         WEIGHTS_PATH = "/kaggle/working/"
#         csv_logger = CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv', append=True, separator=',')
#         reduce_lr = LearningRateScheduler(exponential_lr)
        
#         model.compile(loss=total_loss, optimizer=Adam(learning_rate=LEARNING_RATE),
#                       metrics=[loss_obj.ssim_score, 'mse', loss_obj.psnr])
        
#         checkpoint_path = '/kaggle/working/WAT_style_stacked_{epoch:02d}_val_loss_{val_loss:.4f}.h5'
#         model_checkpoint = ModelCheckpoint(checkpoint_path,
#                                    monitor='val_loss',
#                                    save_best_only=False,
#                                    save_weights_only=False,
#                                    mode='min',
#                                    verbose=1)
#         data_loader_Motion_Simulated = DataLoader(
#             data_path=dataset_path,
#             split_ratio=[0.7, 0.2, 0.1],
#             view="Axial",
#             data_id=data_ids[dataset_path],
#             crop=False,
#             batch_size=BATCH_SIZE,
#             split_json_path=None
#         )

#         data_loader_Motion_Simulated.split_data()
            

#         train_dataset = data_loader_Motion_Simulated.generator('train')
#         test_dataset_Motion_Simulated = data_loader_Motion_Simulated.generator('test')
#         validation_dataset= data_loader_Motion_Simulated.generator('validation') 
#         for (mb,m,ma),f in train_dataset:
#              print(m.shape)
#              break
#         hist = model.fit(train_dataset,
#                          epochs=NB_EPOCH,
#                          verbose=1,
#                          validation_data=validation_dataset,
#                          callbacks=[csv_logger, reduce_lr, model_checkpoint])


# if __name__ == "__main__":
#     if args.d == "W":
#          wat_unet()
#     # main()