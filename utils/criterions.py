import torch.nn.functional as F
import torch

def dice_loss(output, target, num_cls, eps=1e-7):
    p = F.softmax(output, 1)
    dice = 0.0; n = 0
    for i in range(1, num_cls):                 
        num = (p[:,i] * target[:,i]).sum()
        den = p[:,i].sum() + target[:,i].sum()
        dice += 2.0 * num / (den + eps)
        n += 1

    return 1.0 - dice / max(n,1)

def softmax_weighted_loss(output, target, num_cls=5):
    output = F.softmax(output, 1)

    B, _, H, W, Z = output.size() #(B, C, 128, 128, 128)
    for i in range(num_cls):
        outputi = output[:, i, :, :, :] #(B, 128, 128, 128)
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

   