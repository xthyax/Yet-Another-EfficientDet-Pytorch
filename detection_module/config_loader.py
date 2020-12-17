import numpy as np
import json


class LoadConfig:
    ''' 
    json_file: path to json config file 
    '''

    def __init__(self, json_file):
        self.CENTER = None
        self.json_file = json_file
        self.config = self.__read_json()
        self.BATCH_SIZE = self.config['BatchSize']
        self.LEANING_RATE = self.config['LearningRate']
        try:
            self.MOMENTUM = self.config['Momentum']
        except:
            self.MOMENTUM = 0.9
        try:
            self.DECAY = self.config['Decay']
        except:
            self.DECAY = 0.0001
        self.NO_EPOCH = self.config['NoEpoch']
        self.CHANGE_BOX_SIZE = self.config['ChangeBoxSize']
        self.ANCHOR_SCALES = self.config['Anchor_Scales']
        self.ANCHOR_RATIOS = self.config['Anchor_Ratios']
        self.MEAN = self.config['Mean']
        self.STD = self.config['Std']
        self.IMAGE_SIZE = self.config['Image_Size']
        try:
            self.IS_SAVE_BEST_MODELS = self.config['isSaveBestModels']
        except:
            self.IS_SAVE_BEST_MODELS = "False"

        try:
            self.AUGMENT_LIST = self.config['AugmentList']
        except:
            self.AUGMENT_LIST = []
        self.OPTIMIZER = self.config['Optimizer']
        try :
            self.ARCHITECTURE = self.config['Architecture']
        except:
            self.ARCHITECTURE = "0"
        self.CLASS_NAME =  [class_name for class_name in self.config['DictionaryClassName']]
        try:
            self.FAILCLASS_NAME = self.config['FailClassName']
        except:
            self.FAILCLASS_NAME = []

        try:        
            self.PASSCLASS_NAME = self.config['PassClassName']
        except:
            self.PASSCLASS_NAME = []
        try:
            self.NUM_GPU = self.config['NUM_GPU']
        except:
            self.NUM_GPU = 1
        try:
            self.CLASS_THRESHOLD = self.config['ClassThreshold']
        except:
            self.CLASS_THRESHOLD = [0, 0.1, 0.1, 0.95]
        try:
            self.TRAINING_LAYER = self.config['TrainingLayer']
        except:
            self.TRAINING_LAYER = "All"

        try:
            self.NUM_WORKERS = self.config['Num_Workers']
        except:
            self.NUM_WORKERS = 0

    def __read_json(self):
        with open(self.json_file) as f:
            config = json.load(f)
        return config
