# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy , MyF1Score
import src.config as cfg
from src.util import show_setting
from astroformer import model_cfgs , MaxxVit


# [TODO: Optional] Rewrite this class if you want
class MyNetwork(AlexNet):

    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        # [TODO] Modify feature extractor part in AlexNet
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # (64 64 3) -> (64 64 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # (64 64 64) -> (32 32 64)

            nn.Conv2d(64, 192, kernel_size=3, padding=1), # (32 32 64) -> (32 32 192)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1), # (32 32 192) -> (16 16 192)


            nn.Conv2d(192, 384, kernel_size=3), # (16 16 192) -> (14 14 384)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3), # (14 14 384) -> (14 14 256)
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3), # (14 14 256) -> (12 1124 256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # (12 12 256) -> (6 6 256)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [TODO: Optional] Modify this as well if you want
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class MY_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

    




# Sota
class Sota(nn.Module):
    pass


class SEBlock(nn.Module):
    def __init__(self, in_channels):
        super(SEBlock, self).__init__()

        # Squeeze Operation
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        # Excitation Operation 
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.squeeze(x) # AVG [B, H, W, C] -> [B, C] 
        x = x.view(x.size(0), -1)   # flatten  (B, C, 1, 1) → (B, C)
        x = self.excitation(x) # Attention Score for each channels [B, C] -> [B, C] 
        x = x.view(x.size(0), x.size(1), 1, 1)  # reshape (B, C) -> (B, C, 1, 1)
        return x
    
























class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        
        if model_name == 'MyNetwork':
            self.model = MyNetwork()
        elif model_name == 'MY_AlexNet':
            self.model = AlexNet()
        elif model_name == 'Sota':
            self.model = MaxxVit(model_cfgs['astroformer_5'], img_size=64, num_classes=cfg.NUM_CLASSES)  
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.train_accuracy = MyAccuracy()
        self.train_f1_score = MyF1Score(num_classes=num_classes)
        self.val_accuracy = MyAccuracy()
        self.val_f1_score = MyF1Score(num_classes=num_classes)
        self.test_accuracy = MyAccuracy()
        self.test_f1_score = MyF1Score(num_classes=num_classes)
        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)




    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.train_accuracy.update(scores, y)
        self.train_f1_score.update(scores, y)

        self.log_dict({'loss/train': loss, 'accuracy/train': self.train_accuracy.compute(), 'fl_score/train': self.train_f1_score.compute()},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.train_accuracy.reset()
        self.train_f1_score.reset()

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.val_accuracy.update(scores, y)
        self.val_f1_score.update(scores, y)
        
        self.log_dict({'loss/val': loss, 'accuracy/val': self.val_accuracy.compute() , 'fl_score/val': self.val_f1_score.compute()},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()
        self.val_f1_score.reset()

    def test_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        self.test_accuracy.update(scores, y)
        self.test_f1_score.update(scores, y)

        self.log_dict({
            'loss/test': loss,
            'accuracy/test': self.test_accuracy.compute(),
            'fl_score/test': self.test_f1_score.compute(),
            'Total params' : sum(p.numel() for p in self.model.parameters()),
            'trainable_params' : sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            
        }, on_step=False, on_epoch=True, prog_bar=False, logger=True)


    def on_test_epoch_end(self):
        print("[Test Step] End.")
        self.test_accuracy.reset()
        self.test_f1_score.reset()


    def _common_step(self, batch):  
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
