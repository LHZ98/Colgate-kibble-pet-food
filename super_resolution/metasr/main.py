import torch
import utility
import data
import model
import loss
from importlib import import_module
from option import args
from trainer import Trainer
from trainerSingle import TrainerSingle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)     ###setting the log and the train information

if __name__ == '__main__':
    if checkpoint.ok:
        loader = data.Data(args)                ##data loader
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        if args.model == 'metardu' or args.model == 'metamsrn' or args.model == 'metarcan' or args.model == 'metarrdb' or args.model == 'metardn' or args.model == 'msrnsingle' or args.model == 'rcansingle' or args.model == 'rdnsingle' or args.model == 'rdusingle'or args.model == 'rrdbsingle':
            t = Trainer(args, loader, model, loss, checkpoint)
        elif args.model == 'edsr':
            module = import_module('model.edsr')
            model = module.make_model(args).to('cuda')
            t = TrainerSingle(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            #t.test()
        checkpoint.done()
