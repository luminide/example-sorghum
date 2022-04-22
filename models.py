import torch.nn as nn
import timm


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        self.model = timm.create_model(
            conf.arch, conf.pretrained,
            num_classes=num_classes, drop_rate=conf.dropout_rate)

    def forward(self, x):
        x = self.model(x)
        return  x
