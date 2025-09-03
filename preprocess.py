import os
import numpy as np
import medpy.io as medio
join=os.path.join

src_path = '/path/to/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
tar_path = '/path/to/BRATS2023_Training_npy'

name_list = os.listdir(src_path)


def sup_128(xmin, xmax, dim):
    if xmax - xmin < 128:
        print ('#' * 100)
        ecart = (128 - (xmax - xmin)) // 2
        right_ecart = (128 - (xmax - xmin)) - ecart
        
        xmax = min(dim, xmax + right_ecart)
        xmin = max(0, xmin - ecart)

        # If clamping shrank the window, re-extend on the other side
        if xmax - xmin < 128:
            if xmin == 0:
                xmax = min(dim, 128)
            elif xmax == dim:
                xmin = max(0, dim - 128) 
    return xmin, xmax

def Crop_to_Nonzero(vol):

    if len(vol.shape) == 4:
        vol = np.amax(vol, axis=0)

    assert len(vol.shape) == 3
    H, W, D = vol.shape

    x_nonzeros, y_nonzeros, z_nonzeros = np.where(vol != 0)

    x_min, x_max = int(np.amin(x_nonzeros)), int(np.amax(x_nonzeros)) + 1
    y_min, y_max = int(np.amin(y_nonzeros)), int(np.amax(y_nonzeros)) + 1
    z_min, z_max = int(np.amin(z_nonzeros)), int(np.amax(z_nonzeros)) + 1

    x_min, x_max = sup_128(x_min, x_max, H)
    y_min, y_max = sup_128(y_min, y_max, W)
    z_min, z_max = sup_128(z_min, z_max, D)

    return x_min, x_max, y_min, y_max, z_min, z_max

def ZScoreNormalization(vol):
    mask = vol.sum(0) > 0
    for k in range(4):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / (max(y.std(), 1e-8))
        vol[k, ...] = x

    return vol

if not os.path.exists(os.path.join(tar_path, 'vol')):
    os.makedirs(os.path.join(tar_path, 'vol'))

if not os.path.exists(os.path.join(tar_path, 'seg')):
    os.makedirs(os.path.join(tar_path, 'seg'))

for file_name in name_list:
    print (file_name)

    case_id = file_name.split('/')[-1]
    flair, flair_header = medio.load(os.path.join(src_path, file_name, case_id+'-t2f.nii.gz'))
    t1ce, t1ce_header = medio.load(os.path.join(src_path, file_name, case_id+'-t1c.nii.gz'))
    t1, t1_header = medio.load(os.path.join(src_path, file_name, case_id+'-t1n.nii.gz'))
    t2, t2_header = medio.load(os.path.join(src_path, file_name, case_id+'-t2w.nii.gz'))

    vol = np.stack((flair, t1ce, t1, t2), axis=0).astype(np.float32) #(4, 240, 240, 155)

    # Crop to non-zero region
    x_min, x_max, y_min, y_max, z_min, z_max = Crop_to_Nonzero(vol)
    # Z-score normalization
    vol1 = ZScoreNormalization(vol[:, x_min:x_max, y_min:y_max, z_min:z_max])

    vol1 = vol1.transpose(1,2,3,0)
    print(vol1.shape)

    seg, seg_header = medio.load(os.path.join(src_path, file_name, case_id+'-seg.nii.gz')) #(240, 240, 155)
    seg = seg.astype(np.uint8)
    seg1 = seg[x_min:x_max, y_min:y_max, z_min:z_max]

    np.save(os.path.join(tar_path, 'vol', case_id+'_vol.npy'), vol1)
    np.save(os.path.join(tar_path, 'seg', case_id+'_seg.npy'), seg1)
