
import numpy as np
import torch
from sklearn.metrics import f1_score
import os
import logging
import yaml
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
)
from graphmae.datasets.data_util import load_inductive_dataset
from graphmae.models import build_model, build_model2
import wandb


is_wandb = False
folder_name = "fully_arch_gat"
loss1_weight = 0.001
eval_interval = 10


def evaluete(model, loader, device, epoch, model2):
    model.eval()
    model2.train()
    total_val_loss = 0
    mse_list, seed_loss_list, seed_mse_list, cd_list, seed_cd_list, rep_list, ext_list = [], [], [], [], [], [], []
    all_pred = []
    all_gt = []
    lm_preds = []
    lm_gts = []

    for subgraph, subgraph2, idx, name, coarse_gt in loader:
        with torch.no_grad():
            subgraph = subgraph.to(device)
            subgraph2 = subgraph2.to(device)
            coarse_gt = coarse_gt.to(device)

            lm_pred, lm_gt, loss1, loss_dict, hidden, lm_init, pred_centroid = model(subgraph, subgraph.ndata["feat"], subgraph2, idx, coarse_gt, is_train=False)
            center = pred_centroid
            cr_pred, cr_gt, loss2, loss_dict2 = model2(subgraph2, subgraph2.ndata["feat"], idx, hidden, subgraph.batch_num_nodes().tolist(), subgraph, lm_init, coarse_gt, center, is_train=False, epoch=epoch)

            loss = (loss1_weight * loss1) + loss2
        total_val_loss += loss
        mse_list.append(loss_dict["mse"])
        seed_loss_list.append(loss_dict["seed_loss"])
        seed_mse_list.append(loss_dict["seed_mse_loss"])
        cd_list.append(loss_dict2["cd_loss"])
        seed_cd_list.append(loss_dict2["seed_cd_loss"])
        rep_list.append(loss_dict2["rep_loss"])
        ext_list.append(loss_dict2["ext_loss"])
        all_pred.append(cr_pred)
        all_gt.append(cr_gt)
        lm_preds.append(lm_pred)
        lm_gts.append(lm_gt)

    total_val_loss = total_val_loss / len(loader)
    val_loss_dict = {
        "mse": np.mean(mse_list),
        "seed_loss": np.mean(seed_loss_list),
        "seed_mse_loss": np.mean(seed_mse_list),
        "cd_loss": np.mean(cd_list),
        "seed_cd_loss": np.mean(seed_cd_list),
        "rep_loss": np.mean(rep_list),
        "ext_loss": np.mean(ext_list),
    }
    print(f"Epoch :: {epoch}  Val loss :: {total_val_loss}  {val_loss_dict}")

    all_pred = torch.cat(all_pred, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    lm_preds = torch.cat(lm_preds, dim=0)
    lm_gts = torch.cat(lm_gts, dim=0)

    np.save(f'./debug/{folder_name}/test_pred_{epoch}.npy', all_pred.cpu().numpy())
    np.save(f'./debug/{folder_name}/test_gt_{epoch}.npy', all_gt.cpu().numpy())
    np.save(f'./debug/{folder_name}/test_lm_pred_{epoch}.npy', lm_preds.cpu().numpy())
    np.save(f'./debug/{folder_name}/test_lm_gt_{epoch}.npy', lm_gts.cpu().numpy())

    return total_val_loss, val_loss_dict


def mutli_graph_linear_evaluation(model, feat, labels, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_val_epoch = 0
    best_val_test_acc = 0

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        for x, y in zip(feat["train"], labels["train"]):
            out = model(None, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_out = []
            test_out = []
            for x, y in zip(feat["val"], labels["val"]):
                val_pred = model(None, x)
                val_out.append(val_pred)
            val_out = torch.cat(val_out, dim=0).cpu().numpy()
            val_label = torch.cat(labels["val"], dim=0).cpu().numpy()
            val_out = np.where(val_out >= 0, 1, 0)

            for x, y in zip(feat["test"], labels["test"]):
                test_pred = model(None, x)# 
                test_out.append(test_pred)
            test_out = torch.cat(test_out, dim=0).cpu().numpy()
            test_label = torch.cat(labels["test"], dim=0).cpu().numpy()
            test_out = np.where(test_out >= 0, 1, 0)

            val_acc = f1_score(val_label, val_out, average="micro")
            test_acc = f1_score(test_label, test_out, average="micro")
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_val_test_acc = test_acc

        if not mute:
            epoch_iter.set_description(f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_acc:{val_acc}, test_acc:{test_acc: .4f}")

    if mute:
        print(f"# IGNORE: --- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f},  Final-TestAcc: {test_acc:.4f}--- ")
    else:
        print(f"--- Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch}, Early-stopping-TestAcc: {best_val_test_acc:.4f}, Final-TestAcc: {test_acc:.4f} --- ")

    return test_acc, best_val_test_acc


def pretrain(model, dataloaders, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger=None, model2=None):
    
    optim_val_loss = float("inf")
    best_epoch = -1
    logging.info("start training..")
    train_loader, val_loader, test_loader, eval_train_loader = dataloaders

    epoch_iter = tqdm(range(max_epoch))

    if isinstance(train_loader, list) and len(train_loader) ==1:
        train_loader = [train_loader[0].to(device)]
        eval_train_loader = train_loader
    if isinstance(val_loader, list) and len(val_loader) == 1:
        val_loader = [val_loader[0].to(device)]
        test_loader = val_loader

    for epoch in epoch_iter:
        model.train()
        model2.train()
        loss_list = []
        mse_list = []
        seed_loss_list = []
        seed_mse_list = []
        cd_list = []
        seed_cd_list = []
        rep_list = []
        ext_list = []

        for subgraph, subgraph2, idx, name, coarse_gt in train_loader:
            subgraph = subgraph.to(device)
            subgraph2 = subgraph2.to(device)
            idx = idx.to(device)
            coarse_gt = coarse_gt.to(device)

            lm_pred, lm_gt, loss1, loss_dict, hidden, lm_init, pred_centroid = model(subgraph, subgraph.ndata["feat"], subgraph2, idx, coarse_gt, is_train=True)
            center = pred_centroid.detach()
            hidden_detached = [h.detach() for h in hidden]
            cr_pred, cr_gt, loss2, loss_dict2 = model2(subgraph2, subgraph2.ndata["feat"], idx, hidden_detached, subgraph.batch_num_nodes().tolist(), subgraph, lm_init, coarse_gt, center, is_train=True, epoch=epoch)

            total_loss = (loss1_weight * loss1) + loss2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())
            mse_list.append(loss_dict["mse"])
            seed_loss_list.append(loss_dict["seed_loss"])
            seed_mse_list.append(loss_dict["seed_mse_loss"])
            cd_list.append(loss_dict2["cd_loss"])
            seed_cd_list.append(loss_dict2["seed_cd_loss"])
            rep_list.append(loss_dict2["rep_loss"])
            ext_list.append(loss_dict2["ext_loss"])


        train_loss = np.mean(loss_list)
        train_mse_loss = np.mean(mse_list)
        train_seed_loss = np.mean(seed_loss_list)
        train_seed_mse_loss = np.mean(seed_mse_list)
        train_cd_loss = np.mean(cd_list)
        train_seed_cd_loss = np.mean(seed_cd_list)
        train_rep_loss = np.mean(rep_list)
        train_ext_loss = np.mean(ext_list)
        cr_rep_weight = model2.module.current_rep_weight(epoch)
        lm_disp_scale = model.module.disp_scale.item()
        cr_disp_scale = model2.module.disp_scale.item()
        epoch_iter.set_description(f"# Epoch {epoch} | train_loss: {train_loss:.4f} | mse_loss: {train_mse_loss:.4f} | seed_loss: {train_seed_loss:.4f} | seed_mse_loss: {train_seed_mse_loss:.4f} | cd_loss: {train_cd_loss:.4f} | seed_cd_loss: {train_seed_cd_loss:.4f} | rep_loss: {train_rep_loss:.4f} | ext_loss: {train_ext_loss:.4f} | rep_w: {cr_rep_weight:.2f} | lm_disp_scale: {lm_disp_scale:.4f} | cr_disp_scale: {cr_disp_scale:.4f}")

        if is_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mse_loss": train_mse_loss,
                "train_seed_loss": train_seed_loss,
                "train_seed_mse_loss": train_seed_mse_loss,
                "train_cd_loss": train_cd_loss,
                "train_seed_cd_loss": train_seed_cd_loss,
                "train_rep_loss": train_rep_loss,
                "train_ext_loss": train_ext_loss,
                "cr_rep_weight": cr_rep_weight,
                "lm_disp_scale": lm_disp_scale,
                "cr_disp_scale": cr_disp_scale,
            })
        
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)
            
        if epoch % eval_interval == 0 and epoch != 0:
            np.save(f"./debug/{folder_name}/pred_{epoch}.npy", cr_pred.detach().cpu().numpy())
            np.save(f"./debug/{folder_name}/gt_{epoch}.npy", cr_gt.detach().cpu().numpy())
            val_loss, val_loss_dict = evaluete(model, val_loader, device, epoch, model2)
            if val_loss < optim_val_loss:
                optim_val_loss = val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), f"./pretrain_model/{folder_name}/model_best.pt")
                torch.save(model2.state_dict(), f"./pretrain_model/{folder_name}/model2_best.pt")
                
            torch.save(model.state_dict(), f"./pretrain_model/{folder_name}/model_epoch_{epoch}.pt")
            torch.save(model2.state_dict(), f"./pretrain_model/{folder_name}/model2_epoch_{epoch}.pt")
                
            if is_wandb:
                wandb.log({
                    "epoch": epoch,
                    "valid_loss": val_loss,
                    "valid_mse_loss": val_loss_dict["mse"],
                    "valid_seed_loss": val_loss_dict["seed_loss"],
                    "valid_seed_mse_loss": val_loss_dict["seed_mse_loss"],
                    "valid_cd_loss": val_loss_dict["cd_loss"],
                    "valid_seed_cd_loss": val_loss_dict["seed_cd_loss"],
                    "valid_rep_loss": val_loss_dict["rep_loss"],
                    "valid_ext_loss": val_loss_dict["ext_loss"],
                })
        
            if scheduler is not None:
                scheduler.step(val_loss)
        
        print(f"best_epoch : {best_epoch} / best_val_loss : {optim_val_loss}")
            
            
    return model


def main(args):
    global is_wandb, folder_name, loss1_weight, eval_interval
    is_wandb = args.use_wandb
    folder_name = args.folder_name
    loss1_weight = args.loss1_weight
    eval_interval = args.eval_interval

    os.makedirs(f"./pretrain_model/{folder_name}", exist_ok=True)
    os.makedirs(f"./debug/{folder_name}", exist_ok=True)

    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer

    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    (
        train_dataloader,
        valid_dataloader, 
        test_dataloader, 
        eval_train_dataloader, 
        num_features, 
        num_classes
    ) = load_inductive_dataset(dataset_name, args.batch_size)
    args.num_features = num_features
    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        run_name = f"{dataset_name}_{encoder_type}_{decoder_type}_nh{num_hidden}_nl{num_layers}_seed{seed}_{datetime.now().strftime('%y%m%d_%H%M')}"
        if is_wandb:
            wandb.init(
                project="landmark prediction",
                name=run_name,
                config={
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "epochs": max_epoch,
                    "batch_size": args.batch_size,
                    "optimizer": optim_type,
                    "encoder": encoder_type,
                    "decoder": decoder_type,
                    "num_hidden": num_hidden,
                    "num_layers": num_layers,
                    "loss_fn": loss_fn,
                    "seed": seed,
                    "dataset": dataset_name,
                },
                reinit="finish_previous",
            )

        model = build_model(args)
        model2 = build_model2(args)
        
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        
        model.to(device)
        model2.to(device)
        
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        model2 = DDP(model2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
        params = list(model.parameters()) + list(model2.parameters())
        if optim_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optim_type.lower() == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"unsupported optimizer: {optim_type}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.scheduler_patience, factor=args.scheduler_factor)

        if not load_model:
            model = pretrain(model, (train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader), optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f, max_epoch_f, linear_prob, logger, model2)
        model = model.cpu()
        model2 = model2.cpu()

        model = model.to(device)
        model2 = model2.to(device)
        model.eval()
        model2.eval()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        if is_wandb:
            wandb.finish()


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    return args


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
