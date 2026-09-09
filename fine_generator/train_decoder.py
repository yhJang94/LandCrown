import argparse
from datetime import datetime
from torch import optim
from torch.utils.data import DataLoader
from utils.dataset import *
from utils.logger import *
from model.DiffusionPretrain import *
from model.Encoder import *
import datetime as dt
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
from metrics.evaluation_metrics import chamfer_distance_l1, chamfer_distance_l2
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
from model.Encoder import *
from model.Encoder_Component import *
import torch.nn.functional as F

wand = False


def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

parser = argparse.ArgumentParser()
# Experiment setting
if wand == True:
    parser.add_argument('--batch_size', type=int, default=128)
else:
    parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')  # mps for mac
parser.add_argument('--log', type=bool, default=True)
parser.add_argument('--save_dir', type=str, default='./results')

# Grouping setting
parser.add_argument('--group_size', type=int, default=32)
parser.add_argument('--num_group', type=int, default=512)
parser.add_argument('--num_points', type=int, default=4096)
parser.add_argument('--num_output', type=int, default=4096)
parser.add_argument('--diffusion_output_size', default=16384)

# Transformer setting
parser.add_argument('--trans_dim', type=int, default=384)
parser.add_argument('--depth', type=int, default=12)
parser.add_argument('--drop_path_rate', type=float, default=0.1)
parser.add_argument('--num_heads', type=int, default=6)

# Encoder setting
parser.add_argument('--encoder_depth', type=int, default=12)
parser.add_argument('--encoder_num_heads', type=int, default=6)
parser.add_argument('--loss', type=str, default='cdl2')
parser.add_argument('--encoder_dims', type=int, default=384)

# Decoder setting
parser.add_argument('--decoder_depth', type=int, default=4)
parser.add_argument('--decoder_num_heads', type=int, default=8)
# diffusion
parser.add_argument('--num_steps', type=int, default=1000)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')

# sche / optim
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--eta_min', type=float, default=0.00001)
parser.add_argument('--t_max', type=float, default=10000)

args = parser.parse_args()

torch.autograd.set_detect_anomaly(True)

print('loading dataset')

train_dset = CrownDataset(
    input_path = './DiffPMAE/dataset/data_re_12288',
    cr_path = './DiffPMAE/dataset/data_re_448',
    subset = 'train'
)

val_dset = CrownDataset(
    input_path = './DiffPMAE/dataset/data_re_12288',
    cr_path = './DiffPMAE/dataset/data_re_448',
    subset = 'test'
)

if wand == True:
    trn_loader = DataLoader(train_dset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, pin_memory=True, num_workers=4)
else:
    trn_loader = DataLoader(train_dset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, pin_memory=True, num_workers=0)

print('dataset loaded')

print('loading model')

encoder = Crown_Encoder_Module(args).to(args.device)
model = Diff_Point_MAE(args).to(args.device)

encoder = nn.DataParallel(encoder, device_ids=[0])
model = nn.DataParallel(model, device_ids=[0])

ChamferDisL2 = chamfer_distance_l2

optimizer = optim.AdamW(
    list(encoder.parameters()) + list(model.parameters()),
    lr=args.learning_rate,
    weight_decay=args.weight_decay
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.eta_min, T_max=args.t_max)

date = dt.datetime.now()
accumulation_steps = 1


def train(i, batch, epoch):
    
    encoder.eval()
    model.train()
    
    optimizer.zero_grad()
    
    part = batch['partial'].to(args.device)
    gt = batch['gt'].to(args.device)
    bbox = batch['bbox'].to(args.device)

    cond_idx = batch['cond_idx'].to(args.device)
    pred_coarse = batch['coarse_pred'].to(args.device)
    
    x_vis, pos_vis, pred, ae_loss  = encoder(part, cond_idx)
    t = torch.randint(1, args.num_steps, (gt.size(0),))
    fine_loss = model(x_vis, pos_vis, gt, t, pred_coarse, bbox)
    
    loss = ae_loss + fine_loss

    loss.backward()
    optimizer.step()
        
    return loss

def validate(epoch):
    all_pred = []
    all_gt = []
    total_val_loss = 0
    
    encoder.eval()
    model.eval()
    
    for i, batch in enumerate(tqdm(val_loader, desc=f"Validation epoch {epoch + 1}", leave=False)):

        part = batch['partial'].to(args.device)
        gt = batch['gt'].to(args.device)
        cond_idx = batch['cond_idx'].to(args.device)
        pred_coarse = batch['coarse_pred'].to(args.device)

        B, C, N = gt.shape

        with torch.no_grad():

            loss, x_vis, pos_vis = encoder.module.encode(part, cond_idx)
            pred = model.module.sampling(x_vis, pos_vis, pred_coarse)
        
        loss = ChamferDisL2(pred, gt)
        loss = loss.mean()
        total_val_loss += loss
        
        all_pred.append(pred)
        all_gt.append(gt)
        
    total_val_loss = total_val_loss / len(val_loader)
    
    print(f"Epoch :: {epoch}  Val loss :: {total_val_loss}")
    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    
    np.save(f'./npy_result/test_pred_{epoch}.npy', all_pred.cpu().numpy())
    np.save(f'./npy_result/test_gt_{epoch}.npy', all_gt.cpu().numpy())
    
    if wand == True:
        wandb.log({
            "epoch": epoch,
            "valid_loss": total_val_loss,
        })
    
    return total_val_loss


# ---------------------------------------------------------------------------------


# try:
train_optim_loss = float("inf")
optim_loss = float("inf")
n_it = 3000
epoch = 0
best_epoch = -1

while epoch < n_it:
    start = time.time()
    train_loss = 0
    train_classes_loss = 0
    train_binary_loss = 0
    for i, pc in enumerate(tqdm(trn_loader, desc=f"Epoch {epoch + 1}/{n_it}", leave=False)):
        loss = train(i, pc, epoch)
        train_loss += loss

    train_loss = train_loss / len(trn_loader)
    
    if train_loss < train_optim_loss:
        train_optim_loss = train_loss
    
    if wand == True:
        wandb.log({"epoch": epoch, "train_loss": train_loss})
        
    print(f"Epoch :: {epoch + 1}  Trn loss :: {train_loss}")
    
    if epoch % 50 == 0 :
    
        val_loss = validate(epoch)
        
        if val_loss < optim_loss:
            
            optim_loss = val_loss
            best_epoch = epoch
            
            saved_file = {'args': args, 'model': model.state_dict()}
            torch.save(saved_file, f'./pretrain_model/model_best.pt')
            print(f"save the optim_model to ./pretrain_model/model_best.pt")
            
        saved_file = {'args': args, 'model': model.state_dict()}
        torch.save(saved_file, f'./pretrain_model/model_{epoch}.pt')
        print(f"save the optim_model to ./pretrain_model/model_{epoch}.pt")
        
        saved_file = {'args': args, 'optimizer': optimizer.state_dict()}
        torch.save(saved_file, f'./pretrain_model/optimizer_{epoch}.pt')
        
        saved_file = {'args': args, 'scheduler': scheduler.state_dict()}
        torch.save(saved_file, f'./pretrain_model/scheduler_{epoch}.pt')
            
    print(f"train minimum loss : {train_optim_loss}")
    print(f"val minimum loss : {optim_loss}")
    print(f'best epoch : {best_epoch}')
    
    # if epoch + 1 % 2 == 0:
    scheduler.step()
    epoch += 1
    end = time.time()
    print(f"{end - start:.5f} sec")



