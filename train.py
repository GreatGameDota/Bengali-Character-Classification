import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm,trange
from sklearn.model_selection import train_test_split
import sklearn.metrics
import gc
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

from dataloader import get_loader
from models import load_model
from optimizers import get_optimizer
from schedulers import get_scheduler
from transforms import get_transform
from losses import get_criterion

from utils import *

def main():
#     args = parse_args()
    IMAGE_PATH = 'data/images/'
    num_classes_1 = 168
    num_classes_2 = 11
    num_classes_3 = 7
    stats = (0.0692, 0.2051)

    train_df = pd.read_csv('data/train_with_folds.csv')
    # train_df = train_df.set_index(['image_id'])
    # train_df = train_df.drop(['grapheme'], axis=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Data Loaders
    # df_train, df_val = train_test_split(train_df, test_size=0.2, random_state=2021)

    # train_transform = get_transform(128)
    train_transform = A.Compose([
                                A.CoarseDropout(max_holes=1, max_width=64, max_height=64, p=0.9),
                                A.ShiftScaleRotate(rotate_limit=5, p=0.9),
                                A.Normalize(mean=stats[0], std=stats[1], always_apply=True)
    ])
    val_transform = A.Compose([
                            A.Normalize(mean=stats[0], std=stats[1], always_apply=True)
    ])

    BATCH_SIZE = 50
    folds = [
        {
            'train': [1,2,3,4],
            'val': [0]
        },
        {
            'train': [0,2,3,4],
            'val': [1]
        },
        {
            'train': [1,0,3,4],
            'val': [2]
        },
        {
            'train': [1,2,0,4],
            'val': [3]
        },
        {
            'train': [1,2,3,0],
            'val': [4]
        }
    ]

    # Loop over folds
    for fld in range(1):
        fld = 4
        print(f'Train fold: {fld}')

        train_loader = get_loader(train_df, IMAGE_PATH, folds=folds[fld]['train'], batch_size=BATCH_SIZE, workers=4, shuffle=True, transform=train_transform)
        val_loader = get_loader(train_df, IMAGE_PATH, folds=folds[fld]['val'], batch_size=BATCH_SIZE, workers=4, shuffle=False, transform=val_transform)

        # Build Model
        model = load_model('seresnext50_32x4d', pretrained=True)
        model = model.cuda()

        # Optimizer
        optimizer = get_optimizer(model, lr=.00016)

        # Loss
        criterion1 = get_criterion()

        # Training
        history = pd.DataFrame()
        history2 = pd.DataFrame()

        torch.cuda.empty_cache()
        gc.collect()

        best = 0
        best2 = 1e10
        n_epochs = 100
        early_epoch = 0

        # Scheduler
        scheduler = get_scheduler(optimizer, train_loader=train_loader, epochs=n_epochs)

        # print('Loading previous training...')
        # state = torch.load('model.pth')
        # model.load_state_dict(state['model_state'])
        # best = state['kaggle']
        # best2 = state['loss']
        # print(f'Loaded model with kaggle score: {best}, loss: {best2}')
        # optimizer.load_state_dict(state['opt_state'])
        # scheduler.load_state_dict(state['scheduler_state'])
        # early_epoch = state['epoch'] + 1
        # print(f'Beginning at epoch {early_epoch}')
        # print('')

        for epoch in range(n_epochs-early_epoch):
            epoch += early_epoch
            torch.cuda.empty_cache()
            gc.collect()

            # ###################################################################
            # ############## TRAINING ###########################################
            # ###################################################################

            model.train()
            total_loss = 0
            total_loss_1 = 0
            total_loss_2 = 0
            total_loss_3 = 0
            
            # ratio = pow(.5,epoch/50)
            # ratio = 0.7
            ratio = 1.0
            
            t = tqdm(train_loader)
            for batch_idx, (img_batch, y_batch) in enumerate(t):
                img_batch = img_batch.cuda().float()
                y_batch = y_batch.cuda().long()
                
                optimizer.zero_grad()
                
                label1 = y_batch[:,0]
                label2 = y_batch[:,1]
                label3 = y_batch[:,2]
                rand = np.random.rand()
                if rand < 0.5:
                    images, targets = mixup(img_batch, label1, label2, label3, 0.4)
                    output1, output2, output3 = model(images)
                    l1,l2,l3 = mixup_criterion(output1, output2, output3, targets, rate=ratio)
                elif rand < 1:
                    images, targets = cutmix(img_batch, label1, label2, label3, 0.4)
                    output1, output2, output3 = model(images)
                    l1,l2,l3 = cutmix_criterion(output1, output2, output3, targets, rate=ratio)
                # else:
                #     output1,output2,output3 = model(img_batch)
                #     l1, l2, l3 = criterion1(output1,output2,output3, y_batch)

                loss = l1*.4 + l2*.3 + l3*.3
                total_loss += loss
                total_loss_1 += l1*.4
                total_loss_2 += l2*.3
                total_loss_3 += l3*.3
                t.set_description(f'Epoch {epoch+1}/{n_epochs}, LR: %6f, Ratio: %.4f, Loss: %.4f, Root loss: %.4f, Vowel loss: %.4f, Consonant loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],ratio,total_loss/(batch_idx+1),total_loss_1/(batch_idx+1),total_loss_2/(batch_idx+1),total_loss_3/(batch_idx+1)))
                # t.set_description(f'Epoch {epoch}/{n_epochs}, LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(batch_idx+1)))

                if history is not None:
                    history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
                    history.loc[epoch + batch_idx / len(train_loader), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
                    
                loss.backward()
                optimizer.step()
                # if scheduler is not None:
                #     scheduler.step()
            
            # ###################################################################
            # ############## VALIDATION #########################################
            # ###################################################################

            model.eval()
            loss = 0
            
            preds_1 = []
            preds_2 = []
            preds_3 = []
            tars_1 = []
            tars_2 = []
            tars_3 = []
            with torch.no_grad():
                for img_batch, y_batch in val_loader:
                    img_batch = img_batch.cuda().float()
                    y_batch = y_batch.cuda().long()

                    o1, o2, o3 = model(img_batch)

                    l1, l2, l3 = criterion1(o1, o2, o3, y_batch)
                    loss += l1*.4 + l2*.3 + l3*.3

                    for j in range(len(o1)):
                        preds_1.append(torch.argmax(F.softmax(o1[j]), -1))
                        preds_2.append(torch.argmax(F.softmax(o2[j]), -1))
                        preds_3.append(torch.argmax(F.softmax(o3[j]), -1))
                    for i in y_batch:
                        tars_1.append(i[0].data.cpu().numpy())
                        tars_2.append(i[1].data.cpu().numpy())
                        tars_3.append(i[2].data.cpu().numpy())
            
            preds_1 = [p.data.cpu().numpy() for p in preds_1]
            preds_2 = [p.data.cpu().numpy() for p in preds_2]
            preds_3 = [p.data.cpu().numpy() for p in preds_3]
            preds_1 = np.array(preds_1).T.reshape(-1)
            preds_2 = np.array(preds_2).T.reshape(-1)
            preds_3 = np.array(preds_3).T.reshape(-1)

            scores = []
            scores.append(sklearn.metrics.recall_score(
                tars_1, preds_1, average='macro'))
            scores.append(sklearn.metrics.recall_score(
                tars_2, preds_2, average='macro'))
            scores.append(sklearn.metrics.recall_score(
                tars_3, preds_3, average='macro'))
            final_score = np.average(scores, weights=[2,1,1])
            
            loss /= len(val_loader)
            
            if history2 is not None:
                history2.loc[epoch, 'val_loss'] = loss.cpu().numpy()
                history2.loc[epoch, 'acc'] = final_score
                history2.loc[epoch, 'root_acc'] = scores[0]
                history2.loc[epoch, 'vowel_acc'] = scores[1]
                history2.loc[epoch, 'consonant_acc'] = scores[2]
            
            if scheduler is not None:
                scheduler.step(final_score)

            print(f'Dev loss: %.4f, Kaggle: {final_score}, Root acc: {scores[0]}, Vowel acc: {scores[1]}, Consonant acc: {scores[2]}'%(loss))
            
            if epoch > 0:
                history2['acc'].plot()
                plt.savefig(f'epoch%03d_{fld}_acc.png'%(epoch+1))
                plt.clf()
            
            if loss < best2:
                best2 = loss
                print(f'Saving best model... (loss)')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'kaggle': final_score,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }, f'model-1_{fld}.pth')
            
            if final_score > best:
                best = final_score
                print(f'Saving best model... (acc)')
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'kaggle': final_score,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict()
                }, f'model_{fld}.pth')

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--batch_size', default=128, type=int)
    # parser.add_argument('--checkpoint' default=None, stye=str)
#     return parser.parse_args()

if __name__ == '__main__':
    main()