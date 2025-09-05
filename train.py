#coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
import wandb
import torch
import torch.nn as nn 
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.random_seed import setup_seed
from models.UNet import *
from models.nnFormer import nnFormer
from data.data_augmentation import *
from data.dataset import Brats_loadall_nii, Brats_loadall_val_nii
from utils.criterions import dice_loss, softmax_weighted_loss
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader 
from predict import test_softmax
import random
from torch.cuda.amp import GradScaler

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
parser.add_argument('--model', default="UNet3D", type=str, choices=['UNet3D', 'nnFormer'])
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--seed', default=999, type=int)
parser.add_argument('--deep_supervision', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--val_check', default=10, type=int)

## parse arguments
args = parser.parse_args()
setup(args, 'training')


args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomScale3D(scale_range=(0.7, 1.4)), GaussianBlur(dim=3), RandomIntensityChange((0.1,0.1)), RandomFlip(), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
# rotation, scaling, gaussian blur, brightness adjust, and mirroring

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

### tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

def main():
    ########## setting seed
    setup_seed(args.seed)
    os.makedirs(args.savepath, exist_ok=True)
    
    ########## print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ########## init wandb
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    wandb_name_and_id = f'BraTS23_{args.model}_jobid{slurm_job_id}'
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
                "deep_supervision": args.deep_supervision,
            }
        )
    
    ########## setting model
    num_cls = 4
    num_stages = 3
    if args.model == 'UNet3D':
        model = UNet3D(n_channels=4, 
                       n_classes=num_cls, 
                       deep_supervision=args.deep_supervision)
    elif args.model == 'nnFormer':
        model = nnFormer(crop_size=(128,128,128), 
                         embedding_dim=96, 
                         depths=[2, 2, 2, 2], 
                         num_heads=[3, 6, 12, 24], 
                         input_channels=4, 
                         num_classes=num_cls, 
                         window_size=[4, 4, 8, 4], 
                         patch_size=[4, 4, 4], 
                         conv_op=nn.Conv3d,
                         deep_supervision=args.deep_supervision)
    else:
        raise ValueError(f"Unknown model {args.model}")

    print(model)
    model = torch.nn.DataParallel(model).cuda()

    ########## Setting learning scheduler, loss and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, momentum=0.99, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = GradScaler()


    ########## Setting data
    train_file = 'datalist/train.txt'
    val_file = 'datalist/val.txt'
    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    val_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, num_cls=num_cls, val_file=val_file)

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
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

    if args.deep_supervision:
        weights = np.array([1 / (2 ** i) for i in range(num_stages)])
        weights = weights / weights.sum()
    else:
        weights = np.array([1.0], dtype=np.float32)
        print(weights)

    ########## Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume, weights_only=False)
        logging.info('best epoch: {}'.format(checkpoint['epoch']))
        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        model.train()
        loss_epoch = 0.0

        ########## training epoch
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, yo, name = data # x = (B, 4, 128, 128, 128), target = (B, 1, 128, 128, 128), yo = (B, 4, 128, 128, 128)

            x = x.cuda(non_blocking=True)
            yo = yo.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs = model(x)   # (B, 4, 128, 128, 128)
                outs_list = outs if isinstance(outs, (list, tuple)) else [outs]

                total_loss = 0.0
                ### Loss compute
                for j, out in enumerate(outs_list):
                    tgt_j = F.interpolate(yo.float(), size=out.shape[2:], mode="nearest")
                    loss_j = (dice_loss(out, tgt_j, num_cls=num_cls) + softmax_weighted_loss(out, tgt_j, num_cls=num_cls)) * float(weights[j])
                    total_loss = total_loss + loss_j

                loss_epoch += float(total_loss.item())

                ### backpropagation
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            ###log
            writer.add_scalar('loss', total_loss.item(), global_step=step)
            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, total_loss.item())
            logging.info(msg)

            if args.debug:
                break
        
        ########## log current epoch metrics and save current model 
        if not args.debug:
            wandb.log({
                "train/epoch": epoch,
                "train/loss": loss_epoch / iter_per_epoch,
                "train/learning_rate": step_lr,
            })

        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'val_Dice_best': val_Dice_best,
            "scaler": scaler.state_dict(),
            },
            file_name)
        
        ########## validation and test
        if (epoch % args.val_check == 0) or (epoch==args.num_epochs-1) or args.debug: 
            print('validate ...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model)
        
            val_WT, val_TC, val_ET, val_ETpp = dice_score 
            logging.info('Validate epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ETpp = {:.2}, loss = {:.2}'.format(epoch, val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item(), seg_loss.cpu().item()))
            val_dice = ((val_ET + val_WT + val_TC) / 3).item()
            if not args.debug:
                wandb.log({
                    "val/epoch": epoch,
                    "val/val_ET_Dice": val_ET.item(),
                    "val/val_ETpp_Dice": val_ETpp.item(),
                    "val/val_WT_Dice": val_WT.item(),
                    "val/val_TC_Dice": val_TC.item(),
                    "val/val_Dice": val_dice, 
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
                    "scaler": scaler.state_dict(),
                    },
                    file_name)
            model.train()

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

if __name__ == '__main__':
    main()
