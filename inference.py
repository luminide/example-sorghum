import os
import argparse
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenter, save_dist
from dataset import VisionDataset
from models import ModelWrapper
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    '-n', '--num-patches', default=9, type=int, metavar='N',
    help='number of patches per image')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_aug = make_test_augmenter(conf)

    image_dir = 'test'
    test_df = pd.DataFrame()
    image_files = sorted(glob(f'{input_dir}/{image_dir}/*.*'))
    assert len(image_files) > 0, f'No files inside {input_dir}/{image_dir}'
    image_files = [os.path.basename(filename) for filename in image_files]
    test_df['image'] = image_files
    test_df['cultivar'] = class_names[0]
    test_dataset = VisionDataset(
        test_df, conf, input_dir, image_dir,
        class_names, test_aug)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_dir, num_classes):
    checkpoint = torch.load(f'{model_dir}/model.pth', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def test(loader, model, num_classes, num_patches):
    sigmoid = nn.Sigmoid()
    preds = np.zeros((len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            pred_batch = outputs.cpu().numpy()
            num_rows = pred_batch.shape[0]
            end_idx = start_idx + num_rows
            preds[start_idx:end_idx] = pred_batch
            start_idx = end_idx

    assert preds.shape[0] % num_patches == 0
    preds = preds.reshape((preds.shape[0]//num_patches, num_patches, -1))
    # average predictions from patches
    preds = preds.mean(axis=1).argmax(axis=1)
    return preds

def collapse(filenames, num_patches):
    filenames = filenames.iloc[::num_patches]
    # rename from xxx_x.jpg to xxx.png
    return [f'{fn[:-6]}.png' for fn in filenames]

def save_results(df, preds, class_names, num_patches):
    pred_names = [class_names[int(pred)] for pred in preds]
    results = pd.DataFrame()
    results['filename'] = collapse(df['image'], num_patches)
    results['cultivar'] = pred_names
    results.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

    dist_file = 'predicted-distribution.png'
    save_dist(results['cultivar'].value_counts(), dist_file)
    print(f'\nSaved predicted class distribution to {dist_file}')

def run(args, input_dir, model_dir):
    meta_file = os.path.join(input_dir, 'train_cultivar_mapping.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    train_df.dropna(inplace=True)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model, conf = create_model(model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    preds = test(loader, model, num_classes, args.num_patches)
    save_results(df, preds, class_names, args.num_patches)

if __name__ == '__main__':
    args = parser.parse_args()
    run(args, '../input', './')

