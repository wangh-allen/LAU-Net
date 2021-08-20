import os
import math
import torch
import torch.nn as nn
import model.lau

def dataparallel(model, gpu_list):
    ngpus = len(gpu_list)
    assert ngpus != 0, "only support gpu mode"
    assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
    assert isinstance(model, list), "Invalid Type of Dual model"
    for i in range(len(model)):
        if ngpus >= 2:
            model[i] = nn.DataParallel(model[i], gpu_list).cuda()
        else:
            model[i] = model[i].cuda()
    return model


class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs

        self.model = lau.make_model(opt).to(self.device)
        self.a = self.model.state_dict()
        
        if not opt.cpu and opt.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))

        self.load(opt.pre_train,  cpu=opt.cpu)

        if not opt.test_only:
            print(self.model, file=ckp.log_file)
        
        # compute parameter
        num_parameter = self.count_parameters(self.model)
        ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    def count_parameters(self, model):
        if self.opt.n_GPUs > 1:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save(self, path, is_best=False):
        target = self.get_model()
        torch.save(
            target.state_dict(), 
            os.path.join(path, 'model', 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', 'model_best.pt')
            )

    def load(self, pre_train='.', pre_train_dual='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )
