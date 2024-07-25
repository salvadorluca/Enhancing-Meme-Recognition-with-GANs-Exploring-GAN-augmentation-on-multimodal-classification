import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict
from utils import *
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

class BimodalTrainer():
    def __init__(self,
                model: Any,
                optimizer: Any,
                scheduler: Any,
                criterion: Any,
                metrics: Dict[str, Callable],
                lambda_L1: float = 0,
                lambda_L2: float = 0,
                device: Any = None
                ):
        
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = optimizer
        self.criterion = criterion
        self.lambda_L1 = lambda_L1
        self.lambda_L2 = lambda_L2
        self.device = device
        self.scheduler = scheduler

        # Logs and metrics
        self.metrics = metrics
        self.train_metrics_log = {key: [] for key in self.metrics.keys()}
        self.val_metrics_log = {key: [] for key in self.metrics.keys()}

        # Move metrics to device
        for metric in self.metrics.values():
            metric = metric.to(self.device)

        self.train_loss_log = []
        self.val_loss_log = []

    def step(self, img_feature, text_feature, labels):
        self.optimizer.zero_grad()
        predictions = self.model(img_feature, text_feature)
        loss = self.criterion(predictions, labels)

        if self.lambda_L1 > 0:
            loss += self.lambda_L1 * L1_regularization(self.model_params)
        
        if self.lambda_L2 > 0:
            loss += self.lambda_L2 * L2_regularization(self.model_params)
        
        loss.backward()
        self.optimizer.step()

        # Update metrics
        for metric in self.metrics.values():
            metric.update(F.sigmoid(predictions), labels)
        
        # Update logs
        self.train_loss_log.append(loss.item())
        return loss.item()
    
    def train_epoch(self, dataloader):
        # Reset training metrics
        for metric in self.metrics.values():
            metric.reset()

        self.model.train()
        for i, (img_feature, text_feature, labels) in enumerate(tqdm(dataloader, leave=True, desc='Train', colour='blue')):
            # Move data to device
            img_feature = img_feature.to(self.device)
            text_feature = text_feature.to(self.device)
            labels = labels.to(self.device).float()

            self.step(img_feature, text_feature, labels)

        # Update logs
        for k in self.metrics.keys():
            self.train_metrics_log[k].append(self.metrics[k].compute().cpu().item())
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        self.model.eval()
        for i, (img_feature, text_feature, labels) in enumerate(tqdm(dataloader, leave=True, desc='eval', colour='green')):
            # Move data to device
            img_feature = img_feature.to(self.device)
            text_feature = text_feature.to(self.device)
            labels = labels.to(self.device).float()

            predictions = self.model(img_feature, text_feature)
            loss = self.criterion(predictions, labels)

            # Update metrics
            for metric in self.metrics.values():
                metric.update(F.sigmoid(predictions), labels)
            
            # Update logs
            self.val_loss_log.append(loss.item())
        
        # Update metrics logs
        for metric in self.metrics.keys():
            self.val_metrics_log[metric].append(self.metrics[metric].compute().cpu().item())


    def plot(self, fig, dh):
        ax = fig.axes
        ax[0].clear()
        ax[0].plot(self.train_loss_log, label='Train', color='orange')
        # ax[0].plot(self.val_loss_log, label='Val', color='blue')
        # Set y-limits to include last 100 iterations
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_title('Loss')

        # Print all metrics
        ax[1].clear()
        for key, value in self.train_metrics_log.items():
            ax[1].plot(value, label='Train '+key)

        for key, value in self.val_metrics_log.items():
            ax[1].plot(value, label='Val '+key)

        ax[1].set_ylim([0, 1])
        ax[1].set_xlabel('Epoch')
        ax[1].legend()
        ax[1].set_title('Metrics')
        dh.update(fig)
    
    def train(self, train_dataloader, val_dataloader, epochs, fig, dh):
        for epoch in range(epochs):
            self.train_epoch(train_dataloader)
            self.evaluate(val_dataloader)
            self.scheduler.step()

            loss_train = np.mean(self.train_loss_log[-len(train_dataloader):])
            loss_val = np.mean(self.val_loss_log[-len(val_dataloader):])
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {loss_train:.4f} - Val Loss: {loss_val:.4f}')
            print('Train: ' + ''.join([f' {k}: {v[-1]:.4f}' for k, v in self.train_metrics_log.items()]))
            print('Val: ' + ''.join([f' {k}: {v[-1]:.4f}' for k, v in self.val_metrics_log.items()]))

            self.plot(fig, dh)
        

    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss_log': self.train_loss_log,
            'val_loss_log': self.val_loss_log,
            'train_metrics_log': self.train_metrics_log,
            'val_metrics_log': self.val_metrics_log
        }, path)

            








