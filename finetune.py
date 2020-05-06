import os
import sys
import math
import time
import logging
import sys
import argparse
import glob
import torch
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
from model import resnet50
import utils
from utils import adjust_learning_rate, save_checkpoint
import numpy as np
from random import shuffle


parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=30, help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--resume_train', action='store_true', default=False, help='resume training')
parser.add_argument('--resume_dir', type=str, default='./weights/checkpoint.pth.tar', help='save weights directory')
parser.add_argument('--weights_dir', type=str, default='./weights/', help='save weights directory')
parser.add_argument('--learning_step', type=list, default=[15,25,30], help='learning rate steps')


args = parser.parse_args()
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if not os.path.exists(args.save):
    os.makedirs(args.save)
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


# Image Preprocessing 
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])


    num_epochs = args.epochs
    batch_size = args.batch_size


    train_dataset = datasets.folder.ImageFolder(root='/data/imagenet-pytorch/train/', transform=train_transform)
    test_dataset = datasets.folder.ImageFolder(root='/data/imagenet-pytorch/val/', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True, num_workers=10, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False, num_workers=10, pin_memory=True)

    

    num_train = train_dataset.__len__()
    n_train_batches = math.floor(num_train / batch_size)

    criterion = nn.CrossEntropyLoss().cuda()
    bitW = 2
    bitA = 2
    model = resnet50(bitW, bitA, pretrained=True)
    model = utils.dataparallel(model, 4)

    print("Compilation complete, starting training...")   

    test_record = []
    train_record = []
    learning_rate = args.learning_rate
    epoch = 0
    best_top1 = 0

    optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,25,30])


    while epoch < num_epochs:

        epoch = epoch + 1
    # resume training    
        if (args.resume_train) and (epoch == 1):   
            checkpoint = torch.load(args.resume_dir)
            epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            model.load_state_dict(checkpoint['state_dict'])
            test_record = list(
                np.load(args.weights_dir + 'test_record.npy'))
            train_record = list(
                np.load(args.weights_dir + 'train_record.npy'))
            for i in range(epoch):
                scheduler.step()

        logging.info('epoch %d', epoch)

        for param_group in optimizer.param_groups:
            cur_lr = param_group['lr']
        print('learning_rate:', cur_lr)

    # training
        
        train_acc_top1, train_acc_top5, train_obj = train(train_loader, model, criterion, optimizer)
        logging.info('train_acc %f', train_acc_top1)
        train_record.append([train_acc_top1, train_acc_top5])
        np.save(args.weights_dir + 'train_record.npy', train_record)   

    # test
        test_acc_top1, test_acc_top5, test_obj = infer(test_loader, model, criterion)
        is_best = test_acc_top1 > best_top1
        if is_best:
            best_top1 = test_acc_top1

        scheduler.step()

        logging.info('test_acc %f', test_acc_top1)
        test_record.append([test_acc_top1, test_acc_top5])
        np.save(args.weights_dir + 'test_record.npy', test_record)


        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'best_top1': best_top1,
                }, args, is_best)


def train(train_queue, model, criterion, optimizer):


    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.train()
 
    for step, (input, target) in enumerate(train_queue):
   
        n = input.size(0)
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()

        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg



def infer(valid_queue, model, criterion):

    
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)
 
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, top5.avg, objs.avg
 
 

if __name__ == '__main__':
    utils.create_folder(args)       
    main()
