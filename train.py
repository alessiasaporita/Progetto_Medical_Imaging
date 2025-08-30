#coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.optim
import sys
from tensorboardX import SummaryWriter
from utils.random_seed import setup_seed
from models.UNet import *
from models.nnFormer import nnFormer
from data.transforms import *
from data.dataset import Brats_loadall_nii, Brats_loadall_test_nii, Brats_loadall_val_nii
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax
import random

M = 2**32 - 1

def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=2, type=int, help='Batch size')
parser.add_argument('--datapath', default=None, type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=3e-5, type=float)
parser.add_argument('--model', default="UNet3D", type=str)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--debug', action='store_true', default=False)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')

#args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomScale3D(scale_range=(0.7, 1.4), anisotropic=False), GaussianBlur(dim=3), RandomIntensityChange((0.1,0.1)), RandomFlip(), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
# rotation, scaling, gaussian blur, brightness adjust, and mirroring

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))
val_check = [20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 850, 900, 910, 920, 930, 940, 950, 955, 960, 965, 970, 975, 980, 985, 990, 995, 1000] 
print(f"Validation checks: {val_check}")

def main():
    ##########setting seed
    setup_seed(args.seed)
    
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ##########init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'BraTS23_{args.model}_epoch{args.num_epochs}_jobid{slurm_job_id}'
    if not args.debug:
        wandb.init(
            project="Progetto_AI_Medical_Imaging",
            name=wandb_name_and_id,
            id=wandb_name_and_id,
            resume="allow",
            config={
                "architecture": args.model,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
            }
        )
    
    ##########setting models
    num_cls = 4
    if args.model == 'UNet3D':
        model = UNet3D(4, num_cls)
    elif args.model == 'nnFormer':
        model = nnFormer(crop_size=(128,128,128), in_channels=4, num_classes=num_cls)
    else:
        raise ValueError(f"Unknown model {args.model}")

    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting learning scheduler and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    #train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    #optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay
    )

    ########## Setting data
    train_file = 'datalist/train.txt'
    test_file = 'datalist/test.txt'
    val_file = 'datalist/val.txt'
    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, val_file=val_file)

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    #test_loader = MultiEpochsDataLoader(
    #    dataset=test_set,
    #    batch_size=1,
    #    shuffle=False,
    #    num_workers=0,
    #    pin_memory=True)
    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = len(train_loader) #number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0

    ##########Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        model.train()
        loss_epoch = 0.0
        cross_loss_epoch = 0.0
        dice_loss_epoch  = 0.0

        ########## training epoch
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target = data[:2] #x=(B, M=4, 128, 128, 128), target = (B, C, 128, 128, 128)

            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits = model(x)

            ###Loss compute
            cross_loss = criterions.softmax_weighted_loss(logits, target, num_cls=num_cls)
            dice_loss = criterions.dice_loss(logits, target, num_cls=num_cls)
            loss = cross_loss + dice_loss

            cross_loss_epoch += cross_loss.item()
            dice_loss_epoch += dice_loss.item()
            loss_epoch += loss.item()

            ### backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'CE:{:.4f}, DiceLoss:{:.4f},'.format(cross_loss.item(), dice_loss.item())
            logging.info(msg)

            if args.debug:
                break
        
        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch / iter_per_epoch,
                "train/CE": cross_loss_epoch / iter_per_epoch,
                "train/DiceLoss": dice_loss_epoch / iter_per_epoch,
                "train/learning_rate": step_lr,
            })

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            },
            file_name)
        
        ########## validation and test
        if ((epoch + 1) % 20 == 0) or (epoch == 0):
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    model_name = args.model)
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score 
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()))
            val_dice = ((val_ET + val_WT + val_TC) / 3).item()
            if not args.debug:
                wandb.log({
                    "val/epoch":epoch,
                    "val/val_ET_Dice": val_ET.item(),
                    "val/val_ETpp_Dice": val_ETpp.item(),
                    "val/val_WT_Dice": val_WT.item(),
                    "val/val_TC_Dice": val_TC.item(),
                    "val/val_Dice": val_dice.item(), 
                    "val/seg_loss": seg_loss.cpu().item(),       
                })
            
            if val_dice > val_Dice_best:
                val_Dice_best = val_dice
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'val_Dice_best': val_Dice_best,
                    },
                    file_name)
            """   
            print('testing ...')
            test_score = AverageMeter()
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    test_loader,
                    model,
                    dataname = args.dataname)
            test_WT, test_TC, test_ET, test_ETpp = dice_score   
            logging.info('Testing epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ET_postpro = {:.2}'.format(epoch, test_WT.item(), test_TC.item(), test_ET.item(), test_ETpp.item()))
            test_dice = (test_ET + test_WT + test_TC)/3
            if not args.debug:
                wandb.log({
                    "test/epoch":epoch,
                    "test/test_WT_Dice": test_WT.item(),
                    "test/test_TC_Dice": test_TC.item(),
                    "test/test_ET_Dice": test_ET.item(),
                    "test/test_ETpp": test_ETpp.item(),
                    "test/test_Dice": test_dice.item(),  
                    "test/seg_loss": seg_loss.cpu().item(),   
                })
            """
            model.train()

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

if __name__ == '__main__':
    main()
