from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from pytorch_lightning.loggers import WandbLogger  # Add this import
from pytorch_lightning.callbacks import ModelCheckpoint  # Add this import

from torch.utils.data import random_split

label = 'controlnet_roomprompt_46k'
wandb_logger = WandbLogger(name=label,
                           log_model="all",
                           save_dir='logs',
                           project='your_wandb_project_name',  # Add your wandb project name here
                           id=label)
checkpoint_callback = ModelCheckpoint(
                dirpath="./checkpoints",
                every_n_train_steps=1000,
                save_weights_only=False,

                filename='{epoch:02d}-{step}',
                save_top_k=-1,
            )
# Configs
resume_path = './models/epoch=15-step=42351.ckpt'
batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset()
# Determine the length of the validation set
val_size = int(len(dataset) * 0.05)  # 5% of the dataset for validation

# Determine the length of the training set
train_size = len(dataset) - val_size

# Use random_split to create the validation and training datasets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the dataloaders for the training and validation datasets
train_dataloader = DataLoader(train_dataset, num_workers=64, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, num_workers=64, batch_size=batch_size, shuffle=False)

logger = ImageLogger(batch_frequency=logger_freq)

trainer = pl.Trainer(
    gpus=8,
    precision=16,
    callbacks=[logger, checkpoint_callback],  # Add checkpoint_callback here
    check_val_every_n_epoch=1,
    # logger=wandb_logger  # Add the wandb logger here
)

if __name__ == '__main__':
    # Pass the dataloaders to the trainer
    trainer.fit(model, train_dataloader, val_dataloader) 