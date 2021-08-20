import math
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def make_optimizer(opt, my_model, if_main=True):
    if if_main:
        evaluators = my_model.model.evaluator
        for layer in evaluators:
            for name, value in layer.named_parameters():
                value.requires_grad = False
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    else:
        for name, value in my_model.named_parameters():
            value.requires_grad = True
        trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay
    
    return optimizer_function([{'params': trainable, 'initial_lr': opt.lr}], **kwargs)


def make_scheduler(opt, my_optimizer, epoch):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        int(opt.epochs),
        eta_min=opt.eta_min,
        last_epoch=epoch
    )

    return scheduler

def genERP(i,j,N):
    val = math.pi/N
    w = math.cos((j - (N/2) + 0.5) * val)
    return w

def compute_map_ws(img):
    """calculate weights for the sphere, the function provide weighting map for a given video
        :img    the input original video
    """
    equ = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for j in range(0,equ.shape[0]):
        for k in range(0,equ.shape[1]):
            for i in range(0,equ.shape[2]):
                equ[j, k, i] = genERP(i,j,equ.shape[0])
    return equ

def getGlobalWSMSEValue(mx,my):

    mw = compute_map_ws(mx)
    val = np.sum(np.multiply((mx-my)**2,mw))
    den = val/(np.sum(mw))

    return den

def ws_psnr(image1,image2):
    image1_y = rgb2y(image1.data).cpu().numpy()
    image2_y = rgb2y(image2.data).cpu().numpy()

    ws_mse = getGlobalWSMSEValue(image1_y, image2_y)
    # second estimate the ws_psnr

    try:
        ws_psnr = 20 * math.log10(255.0 / math.sqrt(ws_mse))
    except ZeroDivisionError:
        ws_psnr = np.inf

    return ws_psnr

def rgb2y(input_im):
    im_flat = input_im.contiguous().permute(0,2,3,1).view(-1, 3).float()
    mat = torch.Tensor([[0.257, 0, 0],
                        [0.507, 0, 0],
                        [0.098, 0, 0]])
    bias = torch.Tensor([16.0, 0, 0])
    mat = mat.cuda()
    bias = bias.cuda()
    temp = im_flat.mm(mat) + bias
    out = temp.view(1, input_im.shape[2], input_im.shape[3], 3).permute(0,3,1,2)
    return out[:, 0]



