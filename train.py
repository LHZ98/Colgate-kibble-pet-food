import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

import wandb
from evaluate import evaluate
from unet import UNet,UNet_atten,Unet_cross_scale_transformer,Unet_cross_scale_transformer_nospatial,Unet_cross_scale_transformer_noself
from othermodels import UNet_2Plus, UNet_3Plus, SwinUnet, UNetFormer, AttentionUNet, TransUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from utils.bdyloss import SurfaceLoss
from utils.focal_loss import FocalLoss
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# CUDA_VISIBLE_DEVICES=2

# dir_img = Path('../data for segmentation/highfirst_traindata/f1/train/images/')
# dir_mask = Path('../data for segmentation/highfirst_traindata/f1/train/masks/')

# dir_img = Path('nc/data/cropped/cropped512SR/train/images/')
# dir_mask = Path('nc/data/cropped/cropped512SR/train/masks/')

dir_img = Path('nc/data/hr512_nobg/all_0605/images/')
dir_mask = Path('nc/data/hr512_nobg/all_0605/labels/')

# dir_img2 = Path('../data for segmentation/highfirst_traindata/f1/test/images/')
# dir_mask2 = Path('../data for segmentation/highfirst_traindata/f1/test/masks/')

# dir_img2 = Path('nc/data/cropped/cropped512SR/test/images/')
# dir_mask2 = Path('nc/data/cropped/cropped512SR/test/masks/')

dir_img2 = Path('nc/data/hr512_nobg/test/images/')
dir_mask2 = Path('nc/data/hr512_nobg/test/masks/')
# dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        epochs: int = 100,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    #     dataset = BasicDataset(dir_img, dir_mask, img_scale)

    try:
        train_set = CarvanaDataset(dir_img, dir_mask, img_scale)
        val_set = CarvanaDataset(dir_img2, dir_mask2, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_train = len(train_set)
    n_val = len(val_set)

    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    os.environ["WANDB_API_KEY"] = '6ab4e7dc05536a32bd141b412b981d68257a462f'
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    bdy_loss = SurfaceLoss()
    focal_loss = FocalLoss()

    np_array = np.array([0,1])
    pos_weight = torch.from_numpy(np_array)
    pos_weight = pos_weight.to(device=device)
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    global_step = 0

    # 5. Begin training
    best_dice = 0
    for epoch in range(1, epochs + 1):
        # scheduler.step()
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                # assert images.shape[1] == model.n_channels, \
                #     f'Network has been defined with {model.n_channels} input channels, ' \
                #     f'but loaded images have {images.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:


                        loss = criterion(masks_pred, true_masks)
                        # a=F.softmax(masks_pred, dim=1).float()
                        # b=F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                        # print('1:',a.shape)
                        # print('2:',b.shape)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        # loss += focal_loss(masks_pred, true_masks)

                        # loss =+ loss_dice

                        #######################

                        # fat_masks = true_masks.clone()
                        # fat_masks[true_masks==1] = 0
                        # fat_masks[true_masks==3] = 0
                        # fat_masks[true_masks==2] = 1
                        # # fat_pred=masks_pred[:,2,:,:].clone()
                        # # fat_pred = F.softmax(masks_pred, dim=1).float().clone()                        
                        # fat_pred =  masks_pred.argmax(dim=1).clone()
                        # # print(fat_pred.shape)
                        # # fat_pred[fat_pred==1] = 0
                        # # fat_pred[fat_pred==3] = 0
                        # # fat_pred[fat_pred==2] = 1
                        # # print(fat_masks.shape)

                        # loss_bdy,distance_map,weighted_wrong_res = bdy_loss(fat_masks.float(),
                        #     fat_pred,true_masks
                        #     )

                        # loss += loss_dice

                        # if epoch >=20:
                        # loss += loss_bdy
                        #######################

                        # mask_pred_out = masks_pred.argmax(dim=1)[0]
                        # print(mask_pred_out.unique())
                        
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                # experiment.log({
                #     # 'train bdy loss': loss_bdy.item(),
                #     # 'train dice loss': loss_dice.item(),
                #     'train dice loss': loss.item(),
                #     'step': global_step,
                #     'epoch': epoch
                # })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5/2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is None:
                                print(f"layer:{tag}")
                                print(value.grad)
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        # try:
                        # experiment.log({
                        #     'learning rate': optimizer.param_groups[0]['lr'],
                        #     'validation Dice': val_score,
                        #     'images': wandb.Image(images[0].cpu()),
                        #     'masks': {
                        #         'true': wandb.Image(true_masks[0].float().cpu()),
                        #         'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         # 'dist_map': wandb.Image(distance_map[0].float().cpu()),
                        #         # 'weighted_wrong_res': wandb.Image(weighted_wrong_res[0].float().cpu()),
                        #     },
                        #     'step': global_step,
                        #     'epoch': epoch,
                        #     **histograms
                        # })
                        # except:
                        #     pass

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     state_dict['mask_values'] = dataset.mask_values
        #     torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')
        if save_checkpoint and epoch%20==0 :
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            with open(dir_checkpoint / "res.txt", "a") as file:
                # Write new content to the file
                file.write("epoch: " + str(epoch) + "; validation dice: " + str(val_score) + "\n")


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--dir_save', '-dir', type=str, default='./checkpoints_sam100_ct_phase2_cross_validation/', help='Load model from a .pth file')
    parser.add_argument('--model', '-m', type=str, default='UNet_trans', help='Load model from a .pth file')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dir_checkpoint = Path(args.dir_save)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.model == 'UNet':
        model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'UNet3Plus':
        model = UNet_3Plus(n_channels=1, n_classes=args.classes)
    elif args.model == 'UNet2Plus':
        model = UNet_2Plus(n_channels=1, n_classes=args.classes)
    elif args.model =='UNetFormer':
        model = UNetFormer(n_classes=args.classes)
    elif args.model =='SwinUNet':
        model = SwinUnet(n_channels=1, n_classes=args.classes)
    elif args.model == 'AttentionUNet':
        model = AttentionUNet(img_ch=1, n_classes=4)
    elif args.model == 'TransUNet':
        model = TransUNet(img_dim = 512, n_classes = 4)
    elif args.model == 'UNet_atten':
        model = UNet_atten(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model == 'UNet_trans':
        model = Unet_cross_scale_transformer(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model =='UNet_transformer_nospatial':
        model = Unet_cross_scale_transformer_nospatial(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    elif args.model =='UNet_transformer_noself':
        model = Unet_cross_scale_transformer_noself(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)


    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_classes} output channels (classes)\n')

    # if args.load:
    #     state_dict = torch.load(args.load, map_location=device)
    #     del state_dict['mask_values']
    #     model.load_state_dict(state_dict)
    #     logging.info(f'Model loaded from {args.load}')

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_scale=args.scale,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #     )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
