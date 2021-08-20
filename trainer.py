import torch
import utility
from tqdm import tqdm


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.cross_loss = torch.nn.MSELoss()
        self.optimizer = utility.make_optimizer(opt, self.model, if_main=True)
        self.eva_opt = [utility.make_optimizer(opt, self.model.model.evaluator[i], if_main=False)for i in range(opt.n_evaluator)]
        self.error_last = 1e8
        self.alpha = 1e-8
        self.epo_state = {"epoch": 0}
        epoch = 0
        self.scheduler = utility.make_scheduler(opt, self.optimizer, epoch)


    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                total_psnr = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])
                    if isinstance(sr, list): sr = sr[-1]
                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:
                        single_psnr = utility.ws_psnr(sr, hr)
                        total_psnr += single_psnr
                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)
                    print('filename:', filename, 'wspsnr', single_psnr)

                self.ckp.log[-1, si] = total_psnr / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tWS-PSNR: {:.2f} '.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si]
                    )
                )

        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(self.ckp.log[-1, si] >= best[0][si]))
            self.epo_state["epoch"] = epoch
            torch.save(obj=self.epo_state, f=self.opt.save + r'epoch.pth')

    def step(self):
        self.scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs