import os
from pathlib import Path
import time
import copy
import configargparse
from datetime import datetime
import json

import numpy as np
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from functools import partial
from torchvision import models
from loguru import logger

from utils.model_ema import ModelEmaV2
from utils.plot_report import plot_learning_curve
from utils.folder import ImageFolder

import timm
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy

from datetime import datetime

torch.manual_seed(1234)
np.random.seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

history = {}
all_train_loss = []
all_train_accuracy = []
all_val_loss = []
all_val_accuracy = []


def set_config():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--my_config', required=False, is_config_file=True, help='config file path')
    parser.add_argument('--train_dir', type=str, help='Path to train folder')
    parser.add_argument('--val_dir', type=str, help='Path to validation folder')
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes')
    parser.add_argument('--arch', default='resnet101d', type=str,
                        choices=["", 'seresnet152d', "efficientnetv2_rw_m", "efficientnetv2_rw_s", "resnest101e",
                                 "rexnet_100", "resnet101d"])
    parser.add_argument("--PretrainedImageNet", action="store_true", help="Flag for using pretrained model")
    parser.add_argument("--own_weights", type=str, help="If true then use own weights")
    parser.add_argument('--input_size', default=320, type=int)
    parser.add_argument('--batch_size', '--bs', default=64, type=int, help='Batch size per gpu')
    parser.add_argument('--use_model_ema', action='store_true', help='Flag for using EMA')
    parser.add_argument('--model_ema_decay', type=float, default=0.98, help='EMA weight decay')
    parser.add_argument('--optim', default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--frozen_lr', type=float, default=5e-5, help='Frozen learning rate')
    parser.add_argument('--epochs', type=int, default=9, help='Count of epochs')
    parser.add_argument('--freeze_epochs', type=int, default=3, help='Count epochs with freezing weights')
    parser.add_argument('--lr', type=float, default=0.0003, help='Unfrozen Learning rate')
    parser.add_argument('--unfreeze', default='first', choices=['all', 'first', 'nolast'])
    parser.add_argument('--loss', default='cross', choices=['cross', 'smooth'])
    parser.add_argument('--schedule', action='store_true')
    parser.add_argument('--schedule_step', default=3, type=int, help='Change lr every step epochs in gamma times')
    parser.add_argument('--schedule_gamma', default=0.5, type=float, help='new_lr = gamma * old_lr')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    #parser.add_argument('--data_file', type=str, default=None, help='File with selected classes')

    args = parser.parse_args()

    return args


def validation(model, dataloader, criterion, epoch) -> Tuple[float, float]:
    print(f'Validation epoch {epoch + 1}')
    model.eval()  # Set model to evaluate mode

    sum_acc = 0
    count_acc = 0
    sum_loss = 0

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            acc = torch.sum(preds == labels.data) / inputs.size(0)

        sum_acc += acc.item()
        sum_loss += loss.item()
        count_acc += 1

        print(f'{i}/{len(image_datasets["val"]) // args.batch_size} ', end='')
        print(f'| Loss = {(sum_loss / count_acc):.8f} | Accuracy =  {(sum_acc / count_acc):.8f}', end='\r')

        writer.add_scalar("Loss/val", sum_loss / count_acc,
                          i + (epoch + 1) * len(image_datasets["val"]) / args.batch_size)
        writer.add_scalar("Accuracy/val", sum_acc / count_acc,
                          i + (epoch + 1) * len(image_datasets["val"]) / args.batch_size)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    all_val_loss.append(sum_loss / count_acc)
    all_val_accuracy.append(sum_acc / count_acc)

    epoch_loss = running_loss / len(image_datasets['val'])
    epoch_acc = running_corrects.double() / len(image_datasets['val'])

    del (sum_acc, count_acc, sum_loss)

    print()

    return epoch_loss, epoch_acc  # sum_loss / count_acc, sum_acc / count_acc


def train(model, dataloaders, optimizer, criterion, num_epochs, start_epoch=0,
          model_ema: ModelEmaV2 = None, scheduler=None, is_inception=False):
    
    # Creates a GradScaler once at the beginning of training.
    scaler = GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict()) if model_ema is None else \
        copy.deepcopy(model_ema.module.state_dict())
    best_acc = 0.0
    best_loss = 0.0

    # Measurment time of batch
    start = time.time()

    for epoch in range(start_epoch, num_epochs):
        
        # Each epoch has a training and validation phase
        # =========== Train ============
        print(f'\nTrain epoch {epoch + 1}/{num_epochs}')
        model.train()  # Set model to training mode

        sum_acc = 0
        count_acc = 0
        sum_loss = 0

        # Iterate over data.
        for i, (inputs, labels) in enumerate(dataloaders['train']):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            # track history if only in train
            with autocast():
                if is_inception:
                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                    # Inception_v3
                    if args.arch == 'inception_v3':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    elif args.arch == 'inception_v4':
                        # inception_v4
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

            # backward + optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            torch.cuda.synchronize()

            acc = torch.sum(preds == labels.data) / inputs.size(0)

            if model_ema:
                model_ema.update(model)

            sum_acc += acc.item()
            sum_loss += loss.item()
            count_acc += 1


            print(f'{i}/{len(image_datasets["train"]) // args.batch_size} ', end='')
            print(f'| Loss = {(sum_loss / count_acc):.8f} | Accuracy = {(sum_acc / count_acc):.8f}', end='\r')

            writer.add_scalar("Loss/train", sum_loss / count_acc,
                              i + (epoch + 1) * len(image_datasets["train"]) / args.batch_size)
            writer.add_scalar("Accuracy/train", sum_acc / count_acc,
                              i + (epoch + 1) * len(image_datasets["train"]) / args.batch_size)

            # end batch ------------------------------

        all_train_loss.append(sum_loss / count_acc)
        all_train_accuracy.append(sum_acc / count_acc)

        del (sum_acc, count_acc, sum_loss)
        print ()

        # ============ Validataion ==============
        epoch_loss, epoch_acc = validation(model_ema.module if model_ema else model,
                                           dataloaders['val'], criterion, epoch)

        # Adjust Learning Rate
        if scheduler:
            scheduler.step()

        # deep copy the model
        if epoch_acc > best_acc:
            print(f'Best accuracy was updated from {best_acc:.4f} to {epoch_acc:.4f}')
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict()) if model_ema is None else \
                copy.deepcopy(model_ema.module.state_dict())

            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': best_model_wts,
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, os.path.join(writer.log_dir, 'chkpt.pt'))
            # history["best_epoch"] = epoch + 1

        history["train_loss"] = all_train_loss
        history["train_accuracy"] = all_train_accuracy
        history["val_loss"] = all_val_loss
        history["val_accuracy"] = all_val_accuracy

        history["batch_size"] = args.batch_size
        history["arch"] = args.arch
        history["epochs"] = num_epochs

        # Save learning curve
        plot_learning_curve(history=history, path_to_save=os.path.join(writer.log_dir, "learning_curve.png"),
                            num_epochs=num_epochs)

        # Save learning curve to json
        with open(os.path.join(writer.log_dir, f"results_{args.arch}.json"), "w") as file:
            json.dump(history, file, indent=4)
        # end epoch ------------------------------

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def main():
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True,
                                                       num_workers=args.num_workers) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Статус pretrained: ", args.PretrainedImageNet)

    # ======== Model architecture ==============

    if args.PretrainedImageNet == False:
        print("Путь до собственных весов: ", args.own_weights)

        start_name = Path(args.own_weights).stem

        if start_name.startswith('resne') or start_name.startswith('seresnet') or start_name.startswith(
                'efficientnet') or start_name.startswith("rexnet"):

            model_ft = torch.load(args.own_weights)

            set_parameter_requires_grad(model_ft, feature_extract)
            if start_name.startswith('efficient'):
                num_ftrs = model_ft.classifier.in_features
                model_ft.classifier = nn.Linear(num_ftrs, args.num_classes)
            elif args.arch.startswith('rexnet'):
                num_ftrs = model_ft.head.fc.in_features
                model_ft.head.fc = nn.Linear(num_ftrs, args.num_classes)
            else:
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
        else:
            raise Exception("Веса модели не относятся к архитектурам resnet или seresnet152d, или efficientnet")
    else:
        if args.arch.startswith('resne') or args.arch == 'seresnet' or args.arch.startswith(
                'efficientnet') or args.arch.startswith('rexnet'):

            if args.arch == "efficientnetv2_rw_s":
                model_ft = timm.create_model('efficientnetv2_rw_s', pretrained=True)
            elif args.arch == "efficientnetv2_rw_m":
                model_ft = timm.create_model("efficientnetv2_rw_m", pretrained=True)
            elif args.arch == "resnet101d":
                model_ft = timm.create_model("resnet101d", pretrained=True)
            elif args.arch == "resnest101e":
                model_ft = timm.create_model("resnest101e", pretrained=True)
            elif args.arch == "seresnet152d":
                model_ft = timm.create_model("seresnet152d", pretrained=True)
            elif args.arch == "rexnet_100":
                model_ft = timm.create_model("rexnet_100", pretrained=True)
            else:
                raise Exception("Wrong architecture name")

            set_parameter_requires_grad(model_ft, feature_extract)
            if args.arch.startswith('efficient'):
                num_ftrs = model_ft.classifier.in_features
                model_ft.classifier = nn.Linear(num_ftrs, args.num_classes)
            elif args.arch.startswith('tresnet_l') or args.arch.startswith('rexnet'):
                num_ftrs = model_ft.head.fc.in_features
                model_ft.head.fc = nn.Linear(num_ftrs, args.num_classes)
            else:
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
            # for param in model_ft.fc.parameters():
            #     param.requires_grad = True
        elif args.arch.startswith('inception'):
            """ Inception v3
                Be careful, expects (299,299) sized images and has auxiliary output
                """
            if args.arch == 'inception_v4':
                model_ft = timm.create_model('inception_v4', pretrained=True)
                set_parameter_requires_grad(model_ft, feature_extract)
                # Inception_v4
                num_ftrs = model_ft.last_linear.in_features
                model_ft.last_linear = nn.Linear(num_ftrs, args.num_classes)
            elif args.arch == 'inception_v3':
                # Inception_v3
                model_ft = models.inception_v3(pretrained=use_pretrained)
                set_parameter_requires_grad(model_ft, feature_extract)
                # Handle the auxilary net
                num_ftrs = model_ft.AuxLogits.fc.in_features
                model_ft.AuxLogits.fc = nn.Linear(num_ftrs, args.num_classes)
                # Handle the primary net
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, args.num_classes)
            else:
                raise Exception('Wrong architecture name')
    # else:
    #     model_ft = torch.load()
    # ==========================================

    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                print("\t", name)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.use_model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model_ft, decay=args.model_ema_decay)

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(params_to_update, lr=args.frozen_lr, amsgrad=True)
    # optimizer_ft = torch.optim.SGD(params_to_update, lr=0.0003, momentum=0.9)

    scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.5)

    # Setup the loss fxn
    if args.loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'smooth':
        criterion = LabelSmoothingCrossEntropy()

    # Start freeze training
    model_ft = train(
        model_ft, dataloaders_dict, optimizer_ft, criterion,
        args.freeze_epochs, model_ema=model_ema,
        scheduler=scheduler, is_inception=is_inception)

    if is_inception:
        # for name, param in model_ft.named_parameters():
        #     if name.startswith('AuxLogits'):
        #         param.requires_grad = True
        for i, param in enumerate(model_ft.parameters()):
            if i < 13:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:

        # for name, param in model_ft.named_parameters():
        #     if not name.startswith('layer1'):
        #         param.requires_grad = True
        for i, param in enumerate(model_ft.parameters()):
            if i < 15:
                param.requires_grad = False
            else:
                param.requires_grad = True

        for name, param in model_ft.named_parameters():
            param.requires_grad = True

    params_to_update = model_ft.parameters()
    print("Params to learn:")
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)


    params_to_update = list(params_to_update)
    optimizer_ft = torch.optim.SGD(  # params_to_update, lr=0.0001, momentum=0.9)
        [{'params': params_to_update[:-2]},
         {'params': params_to_update[-2:], 'lr': args.lr * 10}],
        lr=args.lr, momentum=0.9)
    if args.schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=args.schedule_step,
                                                    gamma=args.schedule_gamma)
    else:
        scheduler = None

    print('Start unfreeze training')
    # Return best model
    model_ft = train(
        model_ft, dataloaders_dict, optimizer_ft, criterion,
        args.epochs, start_epoch=args.freeze_epochs,
        model_ema=model_ema, scheduler=scheduler, is_inception=is_inception)
    # load best model weights
    # model.load_state_dict(best_model_wts)

    cpu_model = model_ft.eval().cpu()
    # sample_input_cpu = torch.zeros(1, 3, args.input_size, args.input_size, dtype=torch.float).cpu()
    # traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)
    suffix = '_ema' if args.use_model_ema else ''

    if args.PretrainedImageNet == False:
        torch.save(cpu_model, os.path.join(writer.log_dir, start_name + suffix + '.pt'))
    else:
        #torch.jit.save(traced_cpu, os.path.join(writer.log_dir, args.arch + suffix + '.pt'))  # "Inception_v4_ema.pt"
        torch.save(cpu_model, os.path.join(writer.log_dir, args.arch + suffix + '.pt'))

if __name__ == "__main__":
    args = set_config()
    writer = SummaryWriter(os.path.join("results", datetime.now().strftime('%b-%d-%Y_%H:%M')))

    data_transforms = {
        'train': transforms.Compose([
            partial(transforms.functional.adjust_hue, hue_factor=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(20, ),  # translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(args.input_size, ),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {}
    image_datasets['train'] = ImageFolder(args.train_dir, data_transforms['train'])
                                          # limits_classes=limits_classes,
                                          # valid_classes=valid_classes)
    print('Training Datasets', image_datasets['train'])
    image_datasets['val'] = ImageFolder(args.val_dir, data_transforms['val'],
                                        external_id=image_datasets['train'].class_to_idx)
                                        # valid_classes=valid_classes)

    yaml.dump(vars(args), open(writer.log_dir + "/config.yaml", "w"), default_flow_style=False)
    yaml.dump(image_datasets['train'].class_to_idx, open(writer.log_dir + "/mapping.yaml", "w"),
              default_flow_style=False)
    is_inception = args.arch.startswith('inception')
    feature_extract = True
    use_pretrained = True

    start_train = datetime.now()
    main()
    ent_train = datetime.now()
    logger.success(f"Model training time: {ent_train - start_train}")

    writer.flush()
    writer.close()
