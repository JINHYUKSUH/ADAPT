import argparse
import math
import random
from train_util.train_utils import adjust_learning_rate, AverageMeter
import torch
import os
import torch.backends.cudnn as cudnn
from train_util.loaders.data_loader import get_loader_in
from train_util.loaders.model_loader import set_model
from train_util.train_utils import get_optimizer
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import torch
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples in neural networks')

    parser.add_argument('--in-dataset', default="CIFAR-100",
                        type=str, help='CIFAR-10 imagenet')
    parser.add_argument('--out-datasets', default=['SVHN','iSUN', 'LSUN', 'LSUN_resize',  'Imagenet_resize','texture'], nargs="*", type=str,
                        help="['SVHN', 'LSUN', 'iSUN', 'dtd', 'places365']  ['inat', 'sun50', 'places50', 'dtd', ]")
    parser.add_argument('--backbone', default='resnet34',
                        type=str, help='model backbone')
    parser.add_argument('--method', default='ADAPT',
                        type=str, help='method used for training')
    parser.add_argument('--seed', default=1, type=int, help='seed')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device')

    # Optimization options
    parser.add_argument('--epochs', default=500, type=int,
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=512,
                        type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=0.000001, type=float,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--print_every', default=50, type=int,
                        help='print model status')
    parser.add_argument('--fine_tune', action='store_true', default=False,
                        help='fine_tuning')
    parser.add_argument('--temp', default=0.1, type=float,
                        help='temperature')
    parser.add_argument('--cosine', action='store_true', default=True,
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # backbone options
    parser.add_argument('--layers', default=100, type=int,
                        help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int,
                        help='depth of resnet')
    parser.add_argument('--width', default=4, type=int, help='width of resnet')
    parser.add_argument('--growth', default=12, type=int,
                        help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--save-path', default="ADAPT.pt",
                        type=str, help="the path to save the trained model")
    parser.add_argument('--reduce', default=0.5, type=float,
                        help='compression rate in transition stage (default: 0.5)')
    parser.add_argument('--score', default="mahalanobis", type=str,
                        help='the scoring function for evaluation')
    parser.add_argument('--threshold', default=1.0,
                        type=float, help='sparsity level')
    parser.set_defaults(argument=True)
    
    # prototypes arguments
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help="SGD momentum")
    parser.add_argument('--proto_m', default=0.999, type=float, help="prototypes update momentum")
    parser.add_argument('--cache-size', default=6, type=int)
    parser.add_argument('--nviews', default=2, type=int)
    parser.add_argument('--beta', default=0.15, type=float)
    parser.add_argument('--anneal', default=None, type=str, help='Beta annealing')

    # loss config
    parser.add_argument('--lambda_pcon', default=1., type=float)
    parser.add_argument('--epsilon', default=0.05, type=float)
    
    args = parser.parse_args()
    
    if "noaug" in args.method:
        args.nviews = 1

    if args.batch_size > 256:
        args.warm = True
    if args.warm:
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr

    return args


class ADAPT(nn.Module):
    def __init__(self, args, num_classes=10, n_protos=1000, proto_m=0.99, temp=0.1, lambda_pcon=1, k=0,feat_dim=128, epsilon=0.05):
        super(ADAPT, self).__init__()
        self.num_classes = num_classes
        self.temp = temp  # temperature scaling
        self.nviews = args.nviews
        self.cache_size = args.cache_size
        
        self.lambda_pcon = lambda_pcon
        
        self.feat_dim = feat_dim
        
        self.epsilon = epsilon
        self.sinkhorn_iterations = 3
        self.k = min(k, self.cache_size)
        
        self.n_protos = n_protos
        self.proto_m = proto_m
        self.register_buffer("protos", torch.rand(self.n_protos,feat_dim))
        self.protos = F.normalize(self.protos, dim=-1)
        
    def sinkhorn(self, features):
        out = torch.matmul(features, self.protos.detach().T)
            
        Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if torch.isinf(sum_Q):
            self.protos = F.normalize(self.protos, dim=1, p=2)
            out = torch.matmul(features, self.protos.detach().T)
            Q = torch.exp(out.detach() / self.epsilon).t()# Q is K-by-B for consistency with notations from our paper
            sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            Q = F.normalize(Q, dim=1, p=1)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q = F.normalize(Q, dim=0, p=1)
            Q /= B

        Q *= B
        return Q.t()
        
    def mle_loss(self, features, targets, beta):
        # update prototypes by EMA
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        anchor_labels = targets.contiguous().repeat(self.nviews).view(-1, 1)
        contrast_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(anchor_labels, contrast_labels.T).float().cuda()

        Q = self.sinkhorn(features)
        # topk
        if self.k > 0:
            update_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            update_mask = F.normalize(topk_mask * update_mask, dim=1, p=1)
        else:
            update_mask = F.normalize(mask * Q, dim=1, p=1)
        update_features = torch.matmul(update_mask.T, features)
        
        
        protos = self.protos
        protos = self.proto_m * protos + (1-self.proto_m) * update_features
        
        with torch.no_grad():
            normalized_update = F.normalize(update_features, dim=1)
            normalized_protos = F.normalize(self.protos, dim=1)
            sim = F.cosine_similarity(normalized_update, normalized_protos, dim=1)
            sim_mean = sim.mean().item()

        self.protos = F.normalize(protos, dim=1, p=2)
        
        Q = self.sinkhorn(features)
        
        proto_dis = torch.matmul(features, self.protos.detach().T)
        anchor_dot_contrast = torch.div(proto_dis, self.temp)
        logits = F.softplus(anchor_dot_contrast)

        if self.k > 0:
            loss_mask = mask*Q
            _, topk_idx = torch.topk(update_mask, self.k, dim=1)
            topk_mask = torch.scatter(
                torch.zeros_like(update_mask),
                1,
                topk_idx,
                1
            ).cuda()
            loss_mask = F.normalize(topk_mask*loss_mask, dim=1, p=1)
            masked_logits = loss_mask * logits 
        else:  
            masked_logits = F.normalize(Q*mask, dim=1, p=1) * logits
        
        positive_logits = mask * logits
        negative_mask = 1 - mask
        negative_logits = logits * negative_mask
        
        epsilon = 1e-10
        imp = (beta * (negative_logits + epsilon).log()).exp().detach()
        weighted_negative_logits = imp*negative_logits
        total_logits = positive_logits + weighted_negative_logits
    
        pos=torch.sum(masked_logits, dim=1)
        neg=torch.log(torch.sum(torch.exp(total_logits), dim=1, keepdim=True))
        log_prob=pos-neg
        
        loss = -torch.mean(log_prob)

        return loss, sim_mean
    
    def proto_contra(self, sim_mean):
        
        protos = F.normalize(self.protos, dim=1)
        batch_size = self.num_classes
        
        proto_labels = torch.arange(self.num_classes).repeat(self.cache_size).view(-1,1).cuda()
        mask = torch.eq(proto_labels, proto_labels.T).float().cuda()    

        contrast_count = self.cache_size
        contrast_feature = protos

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        with torch.no_grad():
            standard = 0.5
            tau_0 = 0.4
            alignment = sim_mean
            adaptive_tau = (1 + 1.0 * (standard-alignment)) * tau_0

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            adaptive_tau)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to('cuda'),
            0
        )
        mask = mask*logits_mask
        
        pos = torch.sum(F.normalize(mask, dim=1, p=1)*logits, dim=1)
        neg = torch.log(torch.sum(logits_mask * torch.exp(logits), dim=1))
        log_prob=pos-neg

        # loss
        loss = - torch.mean(log_prob)
        return loss
    
           
    def forward(self, features, targets, beta):
        loss = 0
        loss_dict = {}

        g_con, sim_mean = self.mle_loss(features, targets, beta)
        loss += g_con
        loss_dict['mle'] = g_con.cpu().item()
        loss_dict['sim_mean'] = sim_mean
                    
        if self.lambda_pcon > 0:            
            loss_proto = self.proto_contra(sim_mean)
            g_dis = self.lambda_pcon * loss_proto
            loss += g_dis
            loss_dict['proto_contra'] = g_dis.cpu().item()
                                
        self.protos = self.protos.detach()
                
        return loss, loss_dict
    
def train_ADAPT(args, train_loader, model, criterion, optimizer, epoch, beta, scaler=None):
    model.train()

    losses = AverageMeter()
    sub_loss = {}

    for step, (images, labels) in enumerate((train_loader), start=epoch * len(train_loader)):
        if (len(images)) == 2:
            twocrop = True
            images = torch.cat([images[0], images[1]], dim=0)
        else:
            twocrop = False
            
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                if args.fine_tune:
                    features = model.fine_tune_forward(images)
                else:
                    features = model(images)
                if twocrop:
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                else:
                    features = features.unsqueeze(1)
                loss, l_dict = criterion(features, labels, beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()    
            new_scale = scaler.get_scale()   

        losses.update(loss.item(), bsz)
        
        if new_scale >= old_scale:
            adjust_learning_rate(args, optimizer, train_loader, step)
            
        if step%len(train_loader) == 0:
            for k in l_dict.keys():
                sub_loss[k] = []
                
        for k in l_dict.keys():
            sub_loss[k].append(l_dict[k])
        
            
    for k in sub_loss.keys():
        sub_loss[k] = np.mean(sub_loss[k])
 

    return losses.avg, sub_loss

def main():

    train_loader, num_classes = get_loader_in(args, split='train')

    model, _ = set_model(args, num_classes, load_ckpt=False)
    criterion = ADAPT(args, temp=args.temp, num_classes=num_classes, proto_m=args.proto_m, n_protos=num_classes*args.cache_size,  k=args.k, lambda_pcon=args.lambda_pcon)
    model.to(device)
    model.encoder.to(device)
    criterion.to(device)
    
    beta = args.beta

    # build optimizer
    optimizer = get_optimizer(args, model, criterion)
    loss_min = np.Inf

    # tensorboard
    t = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")
    logger = SummaryWriter(log_dir=f"runs/{args.backbone}-{args.method}/{t}")

    # get trainer and scaler
    trainer = train_ADAPT
    scaler = torch.cuda.amp.GradScaler()
                
    for epoch in tqdm(range(args.epochs)):
        loss = trainer(args, train_loader, model, criterion, optimizer, epoch, beta, scaler=scaler)
        if type(loss)==tuple:
            loss, l_dict = loss
            logger.add_scalar('Loss/train', loss, epoch)
            for k in l_dict.keys():
                logger.add_scalar(f'Loss/{k}', l_dict[k], epoch)
                print(f'{k} Loss: {l_dict[k]:.4f}')
        else:
            logger.add_scalar('Loss/train', loss, epoch)
        logger.add_scalar('Lr/train', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {loss:.4f}')
        
        if loss < loss_min:
            loss_min = loss
            torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":

    FORCE_RUN = True
    # FORCE_RUN=False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    args.save_epoch = 50

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # check if the model is trained
    if os.path.exists(args.save_path) and not FORCE_RUN:
        exit()

    main()