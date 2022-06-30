# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import copy
import logging


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--lam', default=1, type=float)
parser.add_argument('--momentum_1',default=0.5,type=float)
parser.add_argument('--momentum_2',default=0.8,type=float)
parser.add_argument('--momentum_3',default=0.9,type=float)
parser.add_argument('--method',default='CT',type=str)

# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
    print('lr = ', alpha_plan[epoch])
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0

    for i, (images, _, _, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
       
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

consistency_criterion = nn.KLDivLoss(reduction='batchmean').cuda()
# Train the Model
def train_ours5(epoch, train_loader, model1, model2, model3, optimizer1, optimizer2, optimizer3, args):
    train_total=0
    train_correct=0
    EMA1 = copy.deepcopy(model1)
    EMA2 = copy.deepcopy(model2)
    EMA3 = copy.deepcopy(model3)

    for i, (images, images_s1, images_s2, images_s3, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
       
        images = Variable(images).cuda()
        images_s1 = Variable(images_s1).cuda()
        images_s2 = Variable(images_s2).cuda()
        images_s3 = Variable(images_s3).cuda()
        labels = Variable(labels).cuda()
       
        # Forward + Backward + Optimize
        logits1 = model1(images)
        logits2 = model2(images)
        logits3 = model3(images)
        prec, _ = accuracy(logits1, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec

        loss_ce1 = F.cross_entropy(logits1, labels, reduce = True)
        loss_ce2 = F.cross_entropy(logits2, labels, reduce = True)
        loss_ce3 = F.cross_entropy(logits3, labels, reduce = True)


        # Consistency Loss
        
        logits_s1 = model1(images_s1)
        logits_s2 = model2(images_s2)
        logits_s3 = model3(images_s3)
        
        outputs = torch.pow(torch.softmax(logits1, dim=-1).detach(), 1 / (3 + 3)) \
                * torch.pow(torch.softmax(logits2, dim=-1).detach(), 1 / (3 + 3)) \
                * torch.pow(torch.softmax(logits3, dim=-1).detach(), 1 / (3 + 3)) \
                * torch.pow(torch.softmax(logits_s1, dim=-1).detach(), 1 / (3 + 3)) \
                * torch.pow(torch.softmax(logits_s2, dim=-1).detach(), 1 / (3 + 3)) \
                * torch.pow(torch.softmax(logits_s3, dim=-1).detach(), 1 / (3 + 3)) #outputs = torch.softmax(logits, dim=-1).detach() #outputs = torch.softmax((logits + logits_s1 + logits_s2) / 3, dim=-1).detach()

        log_outputs_s1 = torch.log_softmax(logits_s1, dim=-1)
        log_outputs_s2 = torch.log_softmax(logits_s2, dim=-1)
        log_outputs_s3 = torch.log_softmax(logits_s3, dim=-1)

        consist_loss1 = consistency_criterion(log_outputs_s1, outputs)
        consist_loss2 = consistency_criterion(log_outputs_s2, outputs)
        consist_loss3 = consistency_criterion(log_outputs_s3, outputs)

        log_outputs1 = torch.log_softmax(logits1, dim=-1)
        log_outputs2 = torch.log_softmax(logits2, dim=-1)
        log_outputs3 = torch.log_softmax(logits3, dim=-1)

        consist_loss01 = consistency_criterion(log_outputs1, outputs)
        consist_loss02 = consistency_criterion(log_outputs2, outputs)
        consist_loss03 = consistency_criterion(log_outputs3, outputs)

        lam = min((epoch/100)*args.lam, args.lam)
        
        loss1 = (1 - lam) * loss_ce1 + lam * (consist_loss01 + consist_loss1)
        loss2 = (1 - lam) * loss_ce2 + lam * (consist_loss02 + consist_loss2)
        loss3 = (1 - lam) * loss_ce3 + lam * (consist_loss03 + consist_loss3)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()


        loss = (loss1 + loss2 + loss3) / 3

        for param, param_ema in zip(model1.parameters(), EMA1.parameters()):
            param.data = param.data * args.momentum_1 + param_ema.data * (1 - args.momentum_1)
            param_ema.data.copy_(param.data)

        for param, param_ema in zip(model2.parameters(), EMA2.parameters()):
            param.data = param.data * args.momentum_2 + param_ema.data * (1 - args.momentum_2)
            param_ema.data.copy_(param.data)

        for param, param_ema in zip(model3.parameters(), EMA3.parameters()):
            param.data = param.data * args.momentum_3 + param_ema.data * (1 - args.momentum_3)
            param_ema.data.copy_(param.data)


        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f, Lamda: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data, lam))


    train_acc=float(train_correct)/float(train_total)
    return train_acc


# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, _, _, _, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


# Evaluate the Model
def evaluate_ensemble(test_loader, model1, model2, model3):
    model1.eval()    # Change model to 'eval' mode.
    model2.eval()    # Change model to 'eval' mode.
    model3.eval()    # Change model to 'eval' mode.

    print('previous_best', best_acc_)
    correct = 0
    total = 0
    for images, _, _, _, labels, _ in test_loader:
        images = Variable(images).cuda()
        logits1 = model1(images)
        logits2 = model2(images)
        logits3 = model3(images)
        logits = (logits1 + logits2 + logits3) / 3
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc



#####################################main code ################################################
args = parser.parse_args()
################################
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[
                        logging.StreamHandler()
                    ])
#########################
# Seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model


print('building model1...')
model1 = ResNet34(num_classes)
print('building model1 done')
optimizer1 = torch.optim.SGD(model1.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

print('building model2...')
model2 = ResNet34(num_classes)
print('building model2 done')
optimizer2 = torch.optim.SGD(model2.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)

print('building model3...')
model3 = ResNet34(num_classes)
print('building model3 done')
optimizer3 = torch.optim.SGD(model3.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)



train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)
alpha_plan = [0.1] * 60 + [0.05]*15 + [0.01] * 15 + [0.001] * 10
model1.cuda()
model2.cuda()
model3.cuda()


epoch=0
train_acc = 0
best_acc_ = 0

# training
noise_prior_cur = noise_prior
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer1, epoch, alpha_plan)
    adjust_learning_rate(optimizer2, epoch, alpha_plan)
    adjust_learning_rate(optimizer3, epoch, alpha_plan)
    model1.train()
    model2.train()
    model3.train()
    train_acc1 = train_ours5(epoch, train_loader, model1, model2, model3, optimizer1, optimizer2, optimizer3,args)

    # evaluate all models
    test_acc_ensemble = evaluate_ensemble(test_loader, model1, model2, model3)
    # print('******test acc on test images is ', test_acc_ensemble, '*******')
    # save results
    logging.info('test acc on test images is {}'.format(test_acc_ensemble))
    best_acc_ = max(best_acc_, test_acc_ensemble)
