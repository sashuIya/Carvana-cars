from pathlib import Path


DATA_ROOT = Path('/home/alexander/kaggle/proj_cars/data')

TRAIN_DIR = DATA_ROOT / 'train'
MASKS_DIR = DATA_ROOT / 'train_masks'
TEST_DIR = DATA_ROOT / 'test'

SAVED_MODELS_ROOT = Path('/home/alexander/kaggle/proj_cars/alapin/output')

INPUT_SIZE = 512
THRESHOLD = 0.5

TRAIN_BATCH_SIZE = 10 
TEST_BATCH_SIZE = 12 

ORIGINAL_HEIGHT = 1280
ORIGINAL_WIDTH = 1918

SAVED_MODELS_ROOT = Path('/home/alexander/kaggle/proj_cars/alapin/output')
SAVED_MODEL = Path('model_best.pth.tar')

LOAD_MODEL = True
