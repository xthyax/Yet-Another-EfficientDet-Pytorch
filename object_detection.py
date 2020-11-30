import os
os.environ["MKL_NUM_THREADS"] = "3" # "6"
os.environ["OMP_NUM_THREADS"] = "2" # "4"
os.environ["NUMEXPR_NUM_THREADS"] = "3" # "6"
import sys
import json
import time
import shutil
import glob
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

from detection_module import config_loader
from detection_module.od_wrapper import EfficientDetWrapper
from utils.utils import set_GPU
from datetime import datetime
###################
# Global Constant #
###################


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train EfficientDet.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="C:\\DLCApplication\\Dataset",
                        help='Directory of the DLModel dataset')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--weight', required=False,
                        metavar="/home/simon/logs/weights.h5",
                        help="Path to weights .h5")
    parser.add_argument('--config', default=os.path.join(ROOT_DIR, 'efficient_module/default_config.json'),
                        help='Path to config json file')
    parser.add_argument('--savePath', default="NONE", required=False,
                        help='Path to save predicted images')
    parser.add_argument('--au_list', required=False,
                        help='a list of used augment technique')
    parser.add_argument('--binary', required=False, default=0, type=int,
                        help='binary/mutli classification option 1/0')
    parser.add_argument('--gpu', required=False, default=None, type=int,
                        help='declare number of gpus')
    args = parser.parse_args()
    colors = [(255,0,0),(255,215,0),(255,140,0),(255,69,0),(0,255,0),(255,255,0),(0,255,255),(0,0,255),]
    
    try:
    
        if args.command == "train":
            param = config_loader.LoadConfig(args.config)

            class TrainConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = param.NUM_GPU if args.gpu == None else args.gpu
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.IMAGE_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                AU_LIST = param.AUGMENT_LIST
                if AU_LIST == [] or AU_LIST == None:
                    AU_LIST = None
                else:
                    AU_LIST = param.AUGMENT_LIST
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary)
                IS_SAVE_BEST_MODELS = json.loads(param.IS_SAVE_BEST_MODELS.lower())
                NUM_WORKERS = param.NUM_WORKERS
                ANCHOR_SCALES = param.ANCHOR_SCALES
                ANCHOR_RATIOS = param.ANCHOR_RATIOS
                MEAN = param.MEAN
                STD = param.STD
            
            # Save config at log path
            os.makedirs(args.logs, exist_ok=True)
            shutil.copy(args.config, os.path.join(args.logs, args.config.split("\\")[-1]))
            config = TrainConfig()
            set_GPU(config.GPU_COUNT)
            model = EfficientDetWrapper(config)
            model.prepare_data()
            # _init_t =  input("[DEBUG] Init train ?(Y/N)\nYour answer: ")
            # if _init_t.lower() == "y":
                # if config.WEIGHT_PATH:
                #     model.resume_training()
                # else:

            model.train()

            # else:
            #     pass
            print("\nTrain Done")

        elif args.command == "cm":
            param = config_loader.LoadConfig(args.config)
            
            class InferConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                
                AU_LIST = None

                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary) # Hardcode
                NUM_WORKERS = param.NUM_WORKERS

            config = InferConfig()
            set_GPU(config.GPU_COUNT)
            model = EfficientDetWrapper(config)
            model.load_weight()
            # Test with 1 img
            img_path = r
            model.inference(img_path)
            # output = model.predict_one(img)
            # print(f"Result : {output}")
            # model.confusion_matrix_evaluate()

        elif args.command == "testing":
            param = config_loader.LoadConfig(args.config)
            
            class InferConfig:
                NO_EPOCH = param.NO_EPOCH
                GPU_COUNT = 1
                LEARNING_RATE = param.LEANING_RATE
                LEARNING_MOMENTUM = param.MOMENTUM
                WEIGHT_DECAY = param.DECAY
                OPTIMIZER = param.OPTIMIZER
                NUM_CLASSES = len(param.CLASS_NAME)
                CLASS_NAME = param.CLASS_NAME
                INPUT_SIZE = param.CHANGE_BOX_SIZE
                IMAGES_PER_GPU = param.BATCH_SIZE
                CLASS_THRESHOLD = param.CLASS_THRESHOLD
                AU_LIST = param.AUGMENT_LIST
                if AU_LIST == [] or AU_LIST == None:
                    AU_LIST = None
                else:
                    AU_LIST = param.AUGMENT_LIST
                ARCHITECTURE = param.ARCHITECTURE
                BATCH_SIZE = param.BATCH_SIZE
                LOGS_PATH = args.logs
                DATASET_PATH = args.dataset
                WEIGHT_PATH = args.weight if args.weight else None
                FAIL_CLASSNAME = param.FAILCLASS_NAME
                PASS_CLASSNAME = param.PASSCLASS_NAME
                BINARY = bool(args.binary)  # Hardcode
            
            config = InferConfig()
            set_GPU(config.GPU_COUNT)
            model = EfficientDetWrapper(config)
            model.prepare_data()
        
    except Exception as e:
        print(traceback.format_exc())
        print(f'[ERROR] {e}', file=sys.stderr)
    finally:
        print('[INFO] End of process.')