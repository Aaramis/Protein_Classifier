import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.skip = nn.Sequential()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, bias=False, padding=1)

    def forward(self, x):
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))
        return x2 + self.skip(x)


class ProtCNN(pl.LightningModule):
    def __init__(self, momentum: int = 0.9, weight_decay: int = 1e-2, lr: int = 1e-2, num_classes: int = 16652):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(22, 128, kernel_size=1, padding=0, bias=False),
            ResidualBlock(128, 128, dilation=2),
            ResidualBlock(128, 128, dilation=3),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            Lambda(lambda x: x.flatten(start_dim=1)),
            torch.nn.Linear(7680, num_classes),
        )

        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        pred = torch.argmax(y_hat, dim=1)
        self.train_acc(pred, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['sequence'], batch['target']
        y_hat = self(x)
        pred = torch.argmax(y_hat, dim=1)
        acc = self.valid_acc(pred, y)
        self.log('valid_acc', self.valid_acc, on_step=False, on_epoch=True)

        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5, 8, 10, 12, 14, 16, 18, 20], gamma=0.9
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
