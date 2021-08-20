import utility
import data
import model
from option import args
from checkpoint import Checkpoint
from trainer import Trainer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
utility.set_seed(args.seed)
checkpoint = Checkpoint(args)


if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    t = Trainer(args, loader, model, None, checkpoint)
    t.test()
    checkpoint.done()
    print("testing complete")

