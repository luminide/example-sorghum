import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

class LossHistory:
    def __init__(self):
        self.data = []

    def add_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, val, np.nan])

    def add_train_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, val, np.nan, np.nan])

    def add_epoch_val_loss(self, epoch, sample_count, val):
        self.data.append(
            [len(self.data), epoch, sample_count, np.nan, np.nan, val])

    def save(self):
        columns = [
            'index', 'epoch', 'sample_count',
            'train_loss', 'val_loss', 'epoch_val_loss']
        df = pd.DataFrame(self.data, columns=columns)
        df.to_csv('history.csv', index=False)

def get_class_names(df):
    labels = df['cultivar']
    return labels.unique()

def save_dist(dist, dist_file):
    fig = plt.figure(figsize=(16, 24))
    sns.barplot(
        y=dist.sort_values(ascending=False).index,
        x=dist.sort_values(ascending=False).values, palette='Reds')
    plt.title('Class Distribution in Predictions')
    plt.savefig(dist_file)

def search_layer(module, layer_type, reverse=True):
    if isinstance(module, layer_type):
        return module

    if not hasattr(module, 'children'):
        return None

    children = list(module.children())
    if reverse:
        children = reversed(children)
    # search for the first occurence recursively
    for child in children:
        res = search_layer(child, layer_type)
        if res:
            return res
    return None

def make_test_augmenter(conf):
    crop_size = round(conf.image_size*conf.crop_size)
    return  A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        A.Normalize(),
        ToTensorV2()
    ])


