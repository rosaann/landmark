from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import argparse
import pprint
import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from datasets import get_dataloader
from transforms import get_transform
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
import utils
import utils.config
import utils.checkpoint
import utils.metrics
import glob

def inference(model, images):
    logits = model(images)
  #  print('logits ', logits)
    if isinstance(logits, tuple):
        logits, aux_logits = logits
    else:
        aux_logits = None
    probabilities = F.sigmoid(logits)
  #  print('probabilities ', probabilities)
    return logits, aux_logits, probabilities


def evaluate_single_epoch(config,gi, model, dataloader, criterion,
                          epoch, writer, postfix_dict):
    model.eval()

    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        probability_list = []
        label_list = []
        loss_list = []
        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image']
            labels = data['key']
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            logits, aux_logits, probabilities = inference(model, images)

            loss = criterion(logits, labels)
            if aux_logits is not None:
                aux_loss = criterion(aux_logits, labels)
                loss = loss + 0.4 * aux_loss
            loss_list.append(loss.item())

            probability_list.extend(probabilities)
            label_list.extend(labels)

           # f_epoch = epoch + i / total_step
           # desc = '{:5s}'.format('val')
           # desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
           # tbar.set_description(desc)
           # tbar.set_postfix(**postfix_dict)

        log_dict = {}
        labels = np.array(label_list)
       # probabilities = np.array(probability_list)

        predictions = torch.argmax(probabilities, 1)
        predictions = np.array(predictions)
        accuracy = np.sum((predictions == labels).astype(float)) / float(predictions.size)

        log_dict['acc'] = accuracy
        log_dict['f1'] = utils.metrics.f1_score(labels, predictions)
        log_dict['loss'] = sum(loss_list) / len(loss_list)

        if writer is not None:
            for l in range(len(predictions)):
                f1 = utils.metrics.f1_score(labels[:,l], predictions[:,l], 'binary')
                writer.add_scalar('val/f1_{:02d}'.format(l), f1, epoch)

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('val/{}/{}'.format(gi, key), value, epoch)
            postfix_dict['val/{}/{}'.format(gi, key)] = value

        return f1


def train_single_epoch(config, gi, model, dataloader, criterion, optimizer,
                       epoch, writer, postfix_dict):
    model.train()

    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {'loss' : 0, 'acc':0}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, data in tbar:
        images = data['image']
        labels = data['key']
    #    print('images ', images.shape)
   #     print('labels ', labels)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        logits, aux_logits, probabilities = inference(model, images)
     #   print('logits ', logits.shape, logits)
     #   print('labels ', labels)
    #    print('probabilities ', probabilities.shape, probabilities)
        loss = criterion(logits, labels)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, labels)
            loss = loss + 0.4 * aux_loss
        log_dict['loss'] += loss.item()
        loss.backward()
        
        if config.train.num_grad_acc is None:
            optimizer.step()
            optimizer.zero_grad()
        elif (i+1) % config.train.num_grad_acc == 0:
            optimizer.step()
            optimizer.zero_grad()

        predictions = torch.argmax(probabilities, 1)
        
       # predictions = predictions.tolist()
        accuracy = (predictions == labels).sum().float() / float(predictions.numel())
        log_dict['acc'] += accuracy.item()
        
     #   f_epoch = epoch + i / total_step

        
      #  for key, value in log_dict.items():
      #      postfix_dict['train/{}/{}'.format(gi, key)] = value

      #  desc = '{:5s}'.format('train')
      #  desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
      #  tbar.set_description(desc)
      #  tbar.set_postfix(**postfix_dict)
    log_dict['lr'] = optimizer.param_groups[0]['lr']
        
    if writer is not None:
        for key, value in log_dict.items():
            writer.add_scalar('train/{}/{}'.format(gi, key), value, epoch)


def train(config,gi, model, dataloaders, criterion, optimizer, scheduler, writer, start_epoch):
    num_epochs = config.train.num_epochs
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    postfix_dict = {'train/lr': 0.0,
                    'train/acc': 0.0,
                    'train/loss': 0.0,
                    'val/f1': 0.0,
                    'val/acc': 0.0,
                    'val/loss': 0.0}

    f1_list = []
    best_f1 = 0.0
    best_f1_mavg = 0.0
    for epoch in range(start_epoch, num_epochs):
        # train phase
        train_single_epoch(config, gi, model, dataloaders['train'],
                           criterion, optimizer, epoch, writer, postfix_dict)

        # val phase
        f1 = evaluate_single_epoch(config,gi, model, dataloaders['val'],
                                   criterion, epoch, writer, postfix_dict)

        if config.scheduler.name == 'reduce_lr_on_plateau':
          scheduler.step(f1)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
          scheduler.step()

        utils.checkpoint.save_checkpoint(config, gi, model, optimizer, epoch, 0)

        f1_list.append(f1)
        f1_list = f1_list[-10:]
        f1_mavg = sum(f1_list) / len(f1_list)

        if f1 > best_f1:
            best_f1 = f1
        if f1_mavg > best_f1_mavg:
            best_f1_mavg = f1_mavg
    return {'f1': best_f1, 'f1_mavg': best_f1_mavg}


def run(config):
    train_group_csv_dir = './data/group_csv/'
    writer = SummaryWriter(config.train.dir)
    train_filenames = list(glob.glob(os.path.join(train_group_csv_dir, 'data_train_group_*')))
    
    for train_file in train_filenames:
        gi_tr = train_file.replace('data_train_group_', '')
        gi_tr = gi_tr.split('/')[-1]
        gi_tr = gi_tr.replace('.csv', '')
        group_idx = int(gi_tr)
        utils.prepare_train_directories(config, group_idx)
        
        model = get_model(config, group_idx)
        if torch.cuda.is_available():
            model = model.cuda()
        criterion = get_loss(config)
        optimizer = get_optimizer(config, model.parameters())
        
    

        checkpoint = utils.checkpoint.get_initial_checkpoint(config, group_idx)
        if checkpoint is not None:
            last_epoch, step = utils.checkpoint.load_checkpoint(model, optimizer, checkpoint)
        else:
            last_epoch, step = -1, -1

        print('from checkpoint: {} last epoch:{}'.format(checkpoint, last_epoch))
        scheduler = get_scheduler(config, optimizer, last_epoch)
    
        dataloaders = {split:get_dataloader(config, group_idx, split, get_transform(config, split))
                   for split in ['train', 'val']}
    

    
        train(config,group_idx, model, dataloaders, criterion, optimizer, scheduler,
          writer, last_epoch+1)


def parse_args():
    parser = argparse.ArgumentParser(description='HPA')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('train landmark Image Classification Challenge.')
    args = parse_args()
    if args.config_file is None:
      raise Exception('no configuration file')

    config = utils.config.load(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    
    run(config)

    print('success!')


if __name__ == '__main__':
    main()
