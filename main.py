import json
import math
import os
import sys
from datetime import datetime
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing
import torch.multiprocessing as mp
from absl import flags
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from config import get_default_config
from models.siamese_baseline import SiameseBaselineModelv1,SiameseLocalandMotionModelBIG
from utils import TqdmToLogger, get_logger,AverageMeter,accuracy,ProgressMeter
from datasets import CityFlowNLDataset
from datasets import CityFlowNLInferenceDataset
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import time
import torch.nn.functional as F
from transformers import BertTokenizer,RobertaTokenizer, RobertaModel
from collections import OrderedDict


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)

best_top1_eval = 0.
def evaluate(model,valloader,epoch,cfg,index=2):
    global best_top1_eval
    print("Test::::")
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_acc = AverageMeter('Acc@1', ':6.2f')
    top5_acc = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(valloader),
        [batch_time, data_time, losses, top1_acc, top5_acc],
        prefix="Test Epoch: [{}]".format(epoch))
    end = time.time()
    with torch.no_grad():
        for batch_idx,batch in enumerate(valloader):
            if cfg.DATA.USE_MOTION:
                image,text,bk,id_car = batch
            else:
                image,text,id_car = batch
            tokens = tokenizer.batch_encode_plus(text, padding='longest',
                                                   return_tensors='pt')
            data_time.update(time.time() - end)
            if cfg.DATA.USE_MOTION:
                pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda(),bk.cuda())
            else:
                pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda())
            logit_scale = logit_scale.mean().exp()
            loss =0 

            # for visual_embeds,lang_embeds in pairs:
            visual_embeds,lang_embeds = pairs[index]
            sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
            sim_t_2_i = sim_i_2_t.t()
            loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
            loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
            loss += (loss_t_2_i+loss_i_2_t)/2

            
            acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1_acc.update(acc1[0], image.size(0))
            top5_acc.update(acc5[0], image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(batch_idx)
    if top1_acc.avg > best_top1_eval:
        best_top1_eval = top1_acc.avg
        checkpoint_file = args.name+"/checkpoint_best_eval.pth"
        torch.save(
            {"epoch": epoch, 
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)


parser = argparse.ArgumentParser(description='AICT5 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--config', default="configs/baseline.yaml", type=str,
                    help='config_file')
parser.add_argument('--name', default="baseline", type=str,
                    help='experiments')
args = parser.parse_args()

cfg = get_default_config()
cfg.merge_from_file(args.config)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(cfg.DATA.SIZE, scale=(0.8, 1.)),
    torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation(10)],p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])



use_cuda = True
train_data=CityFlowNLDataset(cfg.DATA, json_path = cfg.DATA.TRAIN_JSON_PATH, transform=transform_test)
trainloader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
val_data=CityFlowNLDataset(cfg.DATA,json_path = cfg.DATA.EVAL_JSON_PATH, transform=transform_test,Random = False)
valloader = DataLoader(dataset=val_data, batch_size=cfg.TRAIN.BATCH_SIZE*20, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)
os.makedirs(args.name,exist_ok = True)

if cfg.MODEL.NAME == "base":
    model = SiameseBaselineModelv1(cfg.MODEL)
elif cfg.MODEL.NAME == "dual-stream":
    model = SiameseLocalandMotionModelBIG(cfg.MODEL)
else:
    assert cfg.MODEL.NAME in ["base","dual-stream"] , "unsupported model"
if args.resume:
    checkpoint = torch.load(cfg.EVAL.RESTORE_FROM)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # cudnn.benchmark = True


optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.LR.BASE_LR)
step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT*cfg.TRAIN.LR.DELAY , gamma=0.1)
scheduler = WarmUpLR(lr_scheduler = step_scheduler , warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))

if cfg.MODEL.BERT_TYPE == "BERT":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
elif cfg.MODEL.BERT_TYPE == "ROBERTA":
    tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)


model.train()
global_step = 0
best_top1 = 0.
for epoch in range(cfg.TRAIN.EPOCH):
    evaluate(model,valloader,epoch,cfg,0)
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_acc = AverageMeter('Acc@1', ':6.2f')
    top5_acc = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT,
        [batch_time, data_time, losses, top1_acc, top5_acc],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        for batch_idx,batch in enumerate(trainloader):
            if cfg.DATA.USE_MOTION:
                image,text,bk,id_car = batch
            else:
                image,text,id_car = batch
            tokens = tokenizer.batch_encode_plus(text, padding='longest',
                                                   return_tensors='pt')
            data_time.update(time.time() - end)
            global_step+=1
            optimizer.zero_grad()
            if cfg.DATA.USE_MOTION:
                pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda(),bk.cuda())
            else:
                pairs,logit_scale,cls_logits = model(tokens['input_ids'].cuda(),tokens['attention_mask'].cuda(),image.cuda())
            logit_scale = logit_scale.mean().exp()
            loss =0 
 
            for visual_embeds,lang_embeds in pairs:
                sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
                sim_t_2_i = sim_i_2_t.t()
                loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(image.size(0)).cuda())
                loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(image.size(0)).cuda())
                loss += (loss_t_2_i+loss_i_2_t)/2
            for cls_logit in cls_logits:
                loss+= 0.5*F.cross_entropy(cls_logit, id_car.long().cuda())

            acc1, acc5 = accuracy(sim_t_2_i, torch.arange(image.size(0)).cuda(), topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1_acc.update(acc1[0], image.size(0))
            top5_acc.update(acc5[0], image.size(0))

            loss.backward()
            optimizer.step()
          
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step%(len(trainloader)*30))
    
    if epoch%8==1:
        checkpoint_file = args.name+"/checkpoint_%d.pth"%epoch
        torch.save(
            {"epoch": epoch, "global_step": global_step,
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)
    if top1_acc.avg > best_top1:
        best_top1 = top1_acc.avg
        checkpoint_file = args.name+"/checkpoint_best.pth"
        torch.save(
            {"epoch": epoch, "global_step": global_step,
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()}, checkpoint_file)
