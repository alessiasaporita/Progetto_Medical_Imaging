
import torch
from predict import AverageMeter, test_softmax
from data.dataset import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader 
from models.UNet import *
from models.nnFormer import nnFormer
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model', default="UNet3D", type=str, choices=['UNet3D', 'nnFormer'])
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--test_file', default='datalist/test.text', type=str)
parser.add_argument('--datapath', default="/work/grana_neuro/missing_modalities/BRATS2023_Training_npy", type=str)
parser.add_argument('--deep_supervision', action='store_true', default=False)
path = os.path.dirname(__file__)

if __name__ == '__main__':
    args = parser.parse_args()    
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    num_cls = 4

    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=args.datapath, test_file=args.test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

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
    
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    best_epoch = checkpoint['epoch'] + 1
    out_path = args.savepath
    output_path = f"{out_path}_{best_epoch}.txt"

    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        dice_score = test_softmax(
                        test_loader,
                        model,
                        compute_loss=False)
        val_WT, val_TC, val_ET, val_ETpp = dice_score
            
        with open(output_path, 'a') as file:
            file.write('Performance: WT = {:.4f}, TC = {:.4f}, ET = {:.4f}, ETpp = {:.4f}\n'.format(val_WT.item(), val_TC.item(), val_ET.item(), val_ETpp.item()))

        
