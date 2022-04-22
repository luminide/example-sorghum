import os
import argparse
import random
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.utils.data as data
from torch import autocast
from torch.cuda.amp import GradScaler

from augment import make_train_augmenter
from dataset import VisionDataset
from models import ModelWrapper
from config import Config
from util import LossHistory, get_class_names, make_test_augmenter


parser = argparse.ArgumentParser()
parser.add_argument(
    '-j', '--num-workers', default=mp.cpu_count(), type=int, metavar='N',
    help='number of data loading workers')
parser.add_argument(
    '--epochs', default=40, type=int, metavar='N',
    help='number of total epochs to run')
parser.add_argument(
    '-p', '--print-interval', default=100, type=int, metavar='N',
    help='print-interval in batches')
parser.add_argument(
    '--seed', default=None, type=int,
    help='seed for initializing the random number generator')
parser.add_argument(
    '--resume', default='', type=str, metavar='PATH',
    help='path to saved model')
parser.add_argument(
    '-s', '--subset', default=100, type=int, metavar='N',
    help='use a percentage of the data for training and validation')
parser.add_argument(
    '-n', '--num-patches', default=9, type=int, metavar='N',
    help='number of patches per image')
parser.add_argument(
    '--input', default='../input', metavar='DIR',
    help='input directory')

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

class Trainer:
    def __init__(
            self, conf, input_dir, device, num_workers,
            checkpoint, num_patches=9, print_interval=100, subset=100):
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.num_patches = num_patches
        self.max_patience = 10
        self.print_interval = print_interval
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()

        self.create_dataloaders(num_workers, subset)

        self.model = ModelWrapper(conf, self.num_classes)
        self.model = self.model.to(device)
        self.optimizer = self.create_optimizer(conf, self.model)
        assert  self.optimizer is not None, f'Unknown optimizer {conf.optim}'
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=conf.gamma)
        self.history = None

    def expand(self, df, image_col, label_col):
        # expand each image into patches
        ex_df = pd.DataFrame()
        ex_df[image_col] = [
            f'{fn[:-4]}_{i}.jpg' for fn in df[image_col] for i in range(self.num_patches)]
        ex_df[label_col] = df[label_col].repeat(self.num_patches).values
        return ex_df

    def split(self, df, class_names, subset, image_col, label_col):
        train_list = []
        val_list = []
        for name in class_names:
            # group according to class name and then sort according to file name
            # this is to make the validation set dissimilar to the training set
            sel = df[df[label_col] == name].sort_values(by=[image_col])
            if subset < 100:
                # train and validate on subsets
                num_rows = sel.shape[0]*subset//100
                sel = sel.iloc[:num_rows]

            split = sel.shape[0]*90//100
            train_list.append(sel.iloc[:split])
            val_list.append(sel.iloc[split:])

        train_df = self.expand(pd.concat(train_list, ignore_index=True), image_col, label_col)
        val_df = self.expand(pd.concat(val_list, ignore_index=True), image_col, label_col)

        # shuffle the patches in the training set
        train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
        return train_df, val_df

    def create_dataloaders(self, num_workers, subset):
        conf = self.conf
        meta_file = os.path.join(self.input_dir, 'train_cultivar_mapping.csv')
        assert os.path.exists(meta_file), f'{meta_file} not found on Compute Server'
        df = pd.read_csv(meta_file, dtype=str)
        df.dropna(inplace=True)
        class_names = get_class_names(df)
        self.num_classes = len(class_names)

        # split into train and validation sets
        train_df, val_df = self.split(
            df, class_names, subset,
            'image', 'cultivar')

        train_aug = make_train_augmenter(conf)
        test_aug = make_test_augmenter(conf)

        train_dataset = VisionDataset(
            train_df, conf, self.input_dir, 'train_images',
            class_names, train_aug)
        val_dataset = VisionDataset(
            val_df, conf, self.input_dir, 'train_images',
            class_names, test_aug)
        drop_last = (len(train_dataset) % conf.batch_size) == 1
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=False,
            worker_init_fn=worker_init_fn, drop_last=drop_last)
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=conf.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False)

    def create_optimizer(self, conf, model):
        if conf.optim == 'sgd':
            return torch.optim.SGD(
                model.parameters(), lr=conf.lr, momentum=0.9,
                weight_decay=conf.weight_decay)
        if conf.optim == 'adam':
            return torch.optim.AdamW(
                model.parameters(), lr=conf.lr,
                weight_decay=conf.weight_decay)
        return None

    def fit(self, epochs):
        best_loss = None
        patience = self.max_patience
        self.sample_count = 0
        self.history = LossHistory()

        print(f'Running on {device}')
        print(f'{len(self.train_loader.dataset)} examples in training set')
        print(f'{len(self.val_loader.dataset)} examples in validation set')
        trial = os.environ.get('TRIAL')
        suffix = f"-trial{trial}" if trial is not None else ""
        log_dir = f"runs/{datetime.now().strftime('%b%d_%H-%M-%S')}{suffix}"
        writer = SummaryWriter(log_dir=log_dir)

        print('Training in progress...')
        for epoch in range(epochs):
            # train for one epoch
            print(f'Epoch {epoch}:')
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()
            self.scheduler.step()
            writer.add_scalar('Training loss', train_loss, epoch)
            writer.add_scalar('Validation loss', val_loss, epoch)
            writer.add_scalar('Validation accuracy', val_score, epoch)
            writer.flush()
            print(f'training loss {train_loss:.5f}')
            print(f'Validation accuracy {val_score:.2f}% loss {val_loss:.4f}\n')
            self.history.add_epoch_val_loss(epoch, self.sample_count, val_loss)
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                state = {
                    'epoch': epoch, 'model': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                    'conf': self.conf.as_dict()
                }
                torch.save(state, 'model.pth')
                patience = self.max_patience
            else:
                patience -= 1
                if patience == 0:
                    print(
                        f'Validation loss did not improve for '
                        f'{self.max_patience} epochs')
                    break

            self.history.save()
        writer.close()

    def train_epoch(self, epoch):
        loss_func = nn.BCEWithLogitsLoss()
        model = self.model
        optimizer = self.optimizer

        val_iter = iter(self.val_loader)
        val_interval = len(self.train_loader)//len(self.val_loader)
        assert val_interval > 0
        train_loss_list = []
        model.train()
        for step, (images, labels) in enumerate(self.train_loader):
            if (step + 1) % val_interval == 0:
                model.eval()
                # collect validation history for tuning
                try:
                    with torch.no_grad():
                        val_images, val_labels = next(val_iter)
                        val_images = val_images.to(device)
                        val_labels = val_labels.to(device)
                        with autocast(device_type, enabled=self.use_amp):
                            val_outputs = model(val_images)
                        val_loss = loss_func(val_outputs, val_labels)
                        self.history.add_val_loss(epoch, self.sample_count, val_loss.item())
                except StopIteration:
                    pass
                # switch back to training mode
                model.train()

            images = images.to(device)
            labels = labels.to(device)
            # compute output
            # use AMP
            with autocast(device_type, enabled=self.use_amp):
                outputs = model(images)
                loss = loss_func(outputs, labels)

            train_loss_list.append(loss.item())
            self.sample_count += images.shape[0]
            self.history.add_train_loss(epoch, self.sample_count, loss.item())
            if (step + 1) % self.print_interval == 0:
                print(f'Batch {step + 1}: training loss {loss.item():.5f}')
            # compute gradient and do SGD step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

        mean_train_loss = np.array(train_loss_list).mean()
        return mean_train_loss

    def validate(self):
        loss_func = nn.BCEWithLogitsLoss()
        sigmoid = nn.Sigmoid()
        losses = []
        all_labels = np.zeros(len(self.val_loader.dataset), dtype=np.float32)
        preds = np.zeros((len(self.val_loader.dataset), self.num_classes), dtype=np.float32)
        start_idx = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with autocast(device_type, enabled=self.use_amp):
                    outputs = self.model(images)
                end_idx = start_idx + outputs.shape[0]
                all_labels[start_idx:end_idx] = labels.argmax(axis=1).cpu().numpy()
                preds[start_idx:end_idx] = outputs.cpu().numpy()
                start_idx = end_idx
                losses.append(loss_func(outputs, labels).item())

        assert preds.shape[0] % self.num_patches == 0
        preds = preds.reshape((preds.shape[0]//self.num_patches, self.num_patches, -1))
        all_labels = all_labels.reshape((all_labels.shape[0]//self.num_patches, self.num_patches, -1))
        all_labels = all_labels.mean(axis=1)
        # average predictions from patches
        preds = preds.mean(axis=1).argmax(axis=1)
        if np.isfinite(preds).all():
            score = 100*accuracy_score(all_labels, preds)
        else:
            score = np.nan
        return np.mean(losses), score


def worker_init_fn(worker_id):
    random.seed(random.randint(0, 2**32) + worker_id)
    np.random.seed(random.randint(0, 2**32) + worker_id)


def main():
    args = parser.parse_args()
    if args.subset < 100:
        print(f'\nWARNING: {args.subset}% of the data will be used for training\n')
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    model_file = args.resume
    if model_file:
        print(f'Loading model from {model_file}')
        checkpoint = torch.load(model_file)
        conf = Config(checkpoint['conf'])
    else:
        checkpoint = None
        conf = Config()

    print(conf)
    trainer = Trainer(
        conf, input_dir, device, args.num_workers,
        checkpoint, args.num_patches, args.print_interval, args.subset)
    trainer.fit(args.epochs)


if __name__ == '__main__':
    main()
    print('Done')
