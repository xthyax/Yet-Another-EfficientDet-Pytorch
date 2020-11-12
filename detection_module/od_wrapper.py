import argparse
import traceback
import random

import torch
import yaml
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import DataGenerator, Resizer, Normalizer, Augmenter, collater
from backbone import EfficientDetBackbone
from tensorboardX import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

from .callback import SaveModelCheckpoint
from efficientdet.custom_dataloader import FastDataLoader
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from datetime import datetime

def seed_torch(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(1)

class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


class EfficientDetWrapper:
    def __init__(self, config):
        self.config = config
        self.classes = config.CLASS_NAME
        self.input_size = config.INPUT_SIZE
        self.binary_option = config.BINARY
        self.failClasses = config.FAIL_CLASSNAME
        self.passClasses = config.PASS_CLASSNAME
        self.pytorch_model = None
        self.num_of_classes = len(self.classes)
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.id_class_mapping = None
        self.class_weights = None
        self.evaluate_generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.padding_crop = round(self.input_size / 5) if round(self.input_size / 5)  % 2 == 0 else  round(self.input_size / 5) - 1
        self.padding_crop = 0
        self.anchors_ratios = config.ANCHOR_RATIOS
        self.anchors_scales = config.ANCHOR_SCALES
    
    def _build_model(self):
        return EfficientDetBackbone(num_classes=self.num_of_classes, compound_coef= eval(self.config.ARCHITECTURE),
                                 ratios=eval(self.anchors_ratios), scales=eval(self.anchors_scales))

    def load_classes(self):
        pass

    def check_path(self, list_Directory, list_Generator, path_to_check):
        return [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if path_to_check in set_path.split("\\")[-1].lower()]]]
    
    def prepare_data(self):
        self.load_classes()

        list_Directory = [
            os.path.join(self.config.DATASET_PATH, 'Train'),
            os.path.join(self.config.DATASET_PATH, 'Validation'),
            # os.path.join(self.config.DATASET_PATH, 'Test'),
        ]
        
        # Remove empty folder from default folder list
        list_Generator = []
        for diRectory in list_Directory.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                list_Directory.remove(diRectory)

        # Make generator for every available directory
        for diRectory in list_Directory:
            generator = DataGenerator(dataset_dir=diRectory, classes= self.classes, \
                transform=transforms.Compose([Normalizer(mean=self.config.MEAN, std=self.config.STD), Resizer(self.input_size)]))
            
            list_Generator.append(generator)
        
        check_train = self.check_path(list_Directory, list_Generator, "train")
        self.train_generator = check_train[0] if len(check_train) > 0 else None
        
        check_val = self.check_path(list_Directory, list_Generator, "validation")
        self.val_generator = check_val[0] if len(check_val) > 0 else None
        
        check_test = self.check_path(list_Directory, list_Generator, "test")
        self.test_generator = check_test[0] if len(check_test) > 0 else None
            
        # self.evaluate_generator =  DataGenerator(list_Directory,\
        # self.classes, self.failClasses, self.passClasses,\
        # self.input_size + self.padding_crop, self.binary_option, testing=test_time)

        # self.class_weights = compute_class_weight('balanced',self.train_generator.metadata[0], self.train_generator.metadata[1])

        return list_Generator
    def optimizer_chosen(self, model_param):
        try:
            optimizer_dict = {
                'sgd': optim.SGD(params= model_param, lr=self.config.LEARNING_RATE, momentum=0.9, nesterov=True),
                'adam': optim.Adam(params=model_param, lr=self.config.LEARNING_RATE),
                'adadelta': optim.Adadelta(params=model_param, lr=self.config.LEARNING_RATE),
                'adagrad': optim.Adagrad(params=model_param, lr=self.config.LEARNING_RATE),
                'adamax': optim.Adamax(params=model_param, lr=self.config.LEARNING_RATE),
                'adamw': optim.AdamW(params=model_param, lr=self.config.LEARNING_RATE),
                'asgd': optim.ASGD(params=model_param, lr=self.config.LEARNING_RATE),
                'rmsprop': optim.RMSprop(params=model_param, lr=self.config.LEARNING_RATE, weight_decay=1e-5, momentum=0.9),
                'radam': torch_optimizer.RAdam(params=model_param, lr=self.config.LEARNING_RATE),
                'ranger': torch_optimizer.Ranger(params=model_param, lr=self.config.LEARNING_RATE)
            }[self.config.OPTIMIZER.lower()]

            return optimizer_dict
        except KeyError:
            print("Invalid optimizers")

    def load_weight(self):
        if self.config.WEIGHT_PATH is not None:
            if self.config.WEIGHT_PATH.endswith('.pth'):
                weights_path = self.config.WEIGHT_PATH
            else:
                weights_path = get_last_weights(self.config.LOGS_PATH)
            try:
                last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
            except:
                last_step = 0

            try:
                ret = self.pytorch_model.load_state_dict(torch.load(weights_path), strict=False)
            except RuntimeError as e:
                print(f'[Warning] Ignoring {e}')
                print(
                    '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

            print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
        else:
            last_step = 0
            print('[Info] initializing weights...')
            init_weights(self.pytorch_model)

    def train(self):
        os.makedirs(self.config.LOGS_PATH,exist_ok=True)

        seed_torch()
        trainloader = FastDataLoader(self.train_generator, pin_memory=False, \
            worker_init_fn= _init_fn,\
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=True, num_workers=self.config.NUM_WORKERS)
        
        seed_torch()
        valloader = FastDataLoader(self.val_generator, pin_memory=False,\
            worker_init_fn= _init_fn,\
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=False, num_workers=self.config.NUM_WORKERS)

        self.load_weight():

        if self.config.GPU_COUNT > 1 and self.config.BATCH_SIZE // self.config.GPU_COUNT < 4:
            self.pytorch_model.apply(replace_w_sync_bn)
            use_sync_bn = True
        else:
            use_sync_bn = False

        writer = SummaryWriter(self.config.LOGS_PATH)

        self.pytorch_model = ModelWithLoss(self.pytorch_model, debug=False)

        if self.config.GPU_COUNT > 0:
            self.pytorch_model = self.pytorch_model.cuda()
            if self.config.GPU_COUNT > 1:
                self.pytorch_model = CustomDataParallel(self.pytorch_model, self.config.GPU_COUNT)
                if use_sync_bn:
                    patch_replication_callback(self.pytorch_model)

        model_parameters = list(self.pytorch_model.parameters())

        optimizer = self.optimizer_chosen(model_parameters)

        start_time = datetime.now()

        epoch = 0

        try:
            for epoch in range( self.config.NO_EPOCH):

                epoch_loss = []
                loss_classification_ls = []
                loss_regression_ls = []

                seed_torch(epoch)

                self.pytorch_model.train()

                progress_bar = tqdm(trainloader)
                for iter, data in enumerate(progress_bar):

                    try:
                        imgs = data['img']
                        annot = data['annot']

                        if self.config.GPU_COUNT == 1:
                            # if only one gpu, just send it to cuda:0
                            # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        optimizer.zero_grad()
                        cls_loss, reg_loss = self.pytorch_model(imgs, annot, obj_list=params.obj_list)

                        # print(f"\n[DEBUG] sample_per_batch: {imgs.size(0)}")

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                        # print(f"[DEBUG] cls_loss_mean: {cls_loss}")
                        # print(f"[DEBUG] reg_loss_mean: {reg_loss}")

                        loss = cls_loss + reg_loss

                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                        optimizer.step()

                        # TODO: remove epoch_loss list
                        epoch_loss.append(float(loss))

                        # print(f"[DEBUG] epoch loss: {np.mean(epoch_loss)}")
                        progress_bar.set_description(
                            # 'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            #     step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            #     reg_loss.item(), loss.item()))
                        'Epoch: {}/{}. Cls loss: {:.2f}. Reg loss: {:.2f}. Total loss: {:.2f}'.format(
                            epoch + 1, opt.num_epochs,\
                            np.mean(loss_classification_ls),
                            np.mean(loss_regression_ls), np.mean(loss_classification_ls) + np.mean(loss_regression_ls)))

                        progress_bar.update()

                        # if step % opt.save_interval == 0 and step > 0:
                        #     save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        #     print('checkpoint...')
                    except Exception as e:
                        print('[Error]', traceback.format_exc())
                        print(e)
                        continue
                    
                loss = np.mean(loss_classification_ls) + np.mean(loss_regression_ls)
                writer.add_scalars('Loss', {'train': loss}, epoch)
                writer.add_scalars('Regression_loss', {'train': np.mean(loss_regression_ls)}, epoch)
                writer.add_scalars('Classfication_loss', {'train': np.mean(loss_classification_ls)}, epoch)
                # print(f"[DEBUG] Epoch loss: {np.mean(epoch_loss)}")           
                # scheduler.step(np.mean(epoch_loss))

                # Save model after each epoch instead of after certain step

                self.pytorch_model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(valloader):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = self.pytorch_model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                # writer.add_scalars('Loss', {'val': loss}, step)
                # writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                # writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
                writer.add_scalars('Loss', {'Val': loss}, epoch)
                writer.add_scalars('Regression_loss', {'Val': reg_loss}, epoch)
                writer.add_scalars('Classfication_loss', {'Val': cls_loss}, epoch)
                
                writer.flush()
                SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, epoch)
                # Early stopping
                
        except KeyboardInterrupt:
            SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, epoch)

        writer.close()
        end_time = datetime.now()
        print("Training time: {}".format(end_time-start_time))

    def inference(self):
        pass