import sys
import os

# For fixing relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from typing import Callable
from tqdm import tqdm
from utils.utils import *
from typing import List, Callable
import torchmetrics
import copy


class SupervisedTrainer():
    def __init__(
            self,
            model:nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            optim: torch.optim.Optimizer,
            loss_fn: Callable,
            num_classes: int,
            metrics: List[torchmetrics.Metric],
            save_filename:str='model.pt',
            device='cuda',
            # amp: bool = True
            ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.train_metrics = copy.deepcopy(metrics)
        self.val_metrics = copy.deepcopy(metrics)
        self.save_filename = save_filename
        self.device = device
        self.scaler = torch.amp.grad_scaler.GradScaler()

        for metric in self.train_metrics:
            metric.to(self.device)
        for metric in self.val_metrics:
            metric.to(self.device)


    def train_epoch(self):
        losses = []
        self.model.train()

        for metric in self.train_metrics:
            metric.reset()

        with tqdm(self.train_loader, ncols=120) as progress_bar:
            for images, labels in progress_bar:
                # continue
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
                labels_onehot = labels_onehot.type(torch.float32)

                self.optim.zero_grad()
                with torch.autocast(self.device):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels_onehot)
                self.scaler.scale(loss).backward()
                
                self.scaler.step(self.optim)
                self.scaler.update()

                losses.append(loss.item())

                for metric in self.train_metrics:
                    metric.update(torch.softmax(outputs, dim=1), labels)

                progress_bar.set_postfix(
                    loss=f"{np.mean(losses).item():0.4f}", 
                    mem_use=f'{(torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) / 1024**2:.2f} MB',
                    )
        
        out_dict = {}
        out_dict['loss'] = np.mean(losses).item()
        for metric in self.train_metrics:
            out_dict[f'{metric.__class__.__name__.replace("Multiclass", "")}'] = metric.compute().item()
        return out_dict 


    def __validation_epoch(self, loader: torch.utils.data.DataLoader=None):
        losses = []
        ext = ''
        if loader == None:
            loader = self.val_loader
            ext = 'val_'

        for metric in self.val_metrics:
            metric.reset()

        self.model.eval()
        with torch.inference_mode():
            for images, labels in tqdm(loader, ncols=65):
                images = images.to(self.device)
                labels = labels.to(self.device)
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
                labels_onehot = labels_onehot.type(torch.float32)

                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels_onehot)

                for metric in self.val_metrics:
                    metric.update(torch.softmax(outputs, dim=1), labels)
                losses.append(loss.item())

        out_dict = {}
        out_dict[f'{ext}loss'] = np.mean(losses).item()
        for metric in self.val_metrics:
            out_dict[f'{ext}{metric.__class__.__name__.replace("Multiclass", "")}'] = metric.compute().item()
        return out_dict
    
    def evaluate(self, loader):
        return self.__validation_epoch(loader)


    def save_state(self, epoch, fname, val_loss):
        save_dict = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
        }
        torch.save(save_dict, fname)


    def load_state(self, fname):
        load_state(fname, self.model, self.optim)


    def train_model(
            self, 
            num_epochs=1, 
            early_stop_patience=10,
            lr_reduce_patience=3,
            reset_lr_after_training=True
            ):
        initial_lr = self.optim.param_groups[0]['lr']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min', factor=0.5, patience=lr_reduce_patience, min_lr=1e-5)
        losses = []
        val_losses = []

        # For early stopping
        no_improve_epoch_count = 0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch:2d}/{num_epochs:2d}:')
            train_dict = self.train_epoch()
            val_dict = self.__validation_epoch()
            scheduler.step(val_dict['val_loss'])

            if len(val_losses) == 0 or val_dict['val_loss'] < np.min(val_losses):
                # Save model state
                filename = f'{self.save_filename}'
                self.save_state(epoch, filename, val_dict['val_loss'])
                print(f'Best State Saved at {filename} || val_loss: {val_dict["val_loss"]:0.4f}')
                # Reset early stopping counter
                no_improve_epoch_count = 0
            else:
                no_improve_epoch_count += 1

            losses.append(train_dict['loss'])
            val_losses.append(val_dict['val_loss'])
            
            train_log_strings = []
            val_log_strings = []
            for key in train_dict.keys():
                train_log_strings.append(f'{key} : {train_dict[key]:0.4f}')
            for key in val_dict.keys():
                val_log_strings.append(f'{key} : {val_dict[key]:0.4f}')
            logs = train_log_strings + val_log_strings
            
            msg = ' | '.join(logs)
            print(f'{msg} | lr: {scheduler.get_last_lr()[-1]:.4e}')
            
            if no_improve_epoch_count > early_stop_patience:
                print('Early Stopping')
                break
        
        if reset_lr_after_training:
            self.__reset_lr(initial_lr)
        
        return losses, val_losses

    def __reset_lr(self, new_lr):
        for param_group in self.optim.param_groups:
            param_group['lr'] = new_lr

    def get_model(self):
        return self.model
