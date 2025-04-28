import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 256
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 300
#OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
#OPTIMIZER_PARAMS    = {'type': 'Adam', 'lr': 0.001 }
OPTIMIZER_PARAMS    = {'type': 'RAdam', 'lr': 2e-5, 'weight_decay': 1e-2}
#SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingLR', 'T_max' : NUM_EPOCHS}





# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

# Network
#MODEL_NAME          =  'resnet18' 
#MODEL_NAME          =  'resnet152'
#MODEL_NAME          =  'MY_AlexNet' 
#MODEL_NAME          =  'MyNetwork' 
MODEL_NAME          =  'Sota' 
#MODEL_NAME          =  'resnet34' 

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [2]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = 'su980818-hanyang-university' # os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
