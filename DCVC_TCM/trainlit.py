import sys
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
sys.path.append("../../")

import json, os, math

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from src.models.video_net_dmc import DMC

from util.dataset.Vimeo90K import Vimeo90K


class DCVC_TCM_Lit(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        
        self.stage = 0
        self.cfg = cfg
        self._parse_cfg()

        self.model = DMC()
        self.model.apply(self._init_weights)
        self.model.optic_flow._load_Spynet(self.flow_pretrain_dir)

        self.mv_modules: list[nn.Module] = [
            self.model.optic_flow,
            self.model.mv_encoder,
            self.model.mv_prior_encoder,

            self.model.bit_estimator_z_mv,

            self.model.mv_prior_decoder,
            self.model.mv_decoder
        ]

        self.sum_count = 0
        self.sum_out = {
            "bpp_mv_y": 0,
            "bpp_mv_z": 0,
            "bpp_y": 0,
            "bpp_z": 0,
            "bpp": 0,

            "ME_MSE": 0,
            "ME_PSNR": 0,

            "MSE": 0,
            "PSNR": 0,

            "loss": 0,
        }

    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        loss = 0
        feature = None
        ref_frame = batch[:, 0, ...].to(self.device)

        for i in range(1, T):
            input_frame = batch[:, i,...].to(self.device)
            out = self.model(input_frame, ref_frame, feature)

            loss += self._get_loss(input_frame, out, self.train_lambda)

            # take recon image as ref image
            ref_frame = out["recon_image"]
            feature = out["feature"]


            # log
            self.sum_count += 1
            self.sum_out["bpp_mv_y"] += out["bpp_mv_y"].item()
            self.sum_out["bpp_mv_z"] += out["bpp_mv_z"].item()
            self.sum_out["bpp_y"]    += out["bpp_y"].item()
            self.sum_out["bpp_z"]    += out["bpp_z"].item()
            self.sum_out["bpp"]      += out["bpp"].item()

            self.sum_out["MSE"]      += out["MSE"]
            self.sum_out["PSNR"]     += out["PSNR"]

            self.sum_out["ME_MSE"]   += out["ME_MSE"]
            self.sum_out["ME_PSNR"]  += out["ME_PSNR"]


        # average loss
        loss = loss / (T - 1)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        
        # log
        if self.global_step % 50 == 0:
            for key in self.sum_out.keys():
                self.sum_out[key] /= self.sum_count

            self.sum_out["stage"] = float(self.stage)
            self.sum_out["lr"]    = self.optimizers().optimizer.state_dict()['param_groups'][0]['lr']
            self.sum_out["loss"]  = loss.item()

            self.log_dict(self.sum_out)

            for key in self.sum_out.keys():
                self.sum_out[key] = 0

            self.sum_count = 0


    def _get_loss(self, input, output, frame_lambda):
        dist_me = F.mse_loss(input, output["warpped_image"])
        dist_recon = F.mse_loss(input, output["recon_image"])

        output["MSE"] = dist_recon.item()
        output["PSNR"] = mse2psnr(output["MSE"])

        output["ME_MSE"] = dist_me.item()
        output["ME_PSNR"] = mse2psnr(output["ME_MSE"])

        if self.stage == 0:
            dist = dist_me
            rate = output["bpp_mv_y"] + output["bpp_mv_z"]
        
        elif self.stage == 1:
            dist = dist_recon
            rate = 0

        elif self.stage == 2:
            dist = dist_recon
            rate = output["bpp_y"] + output["bpp_z"]

        else:
            dist = dist_recon
            rate = output["bpp"]

        return frame_lambda * dist + rate


    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr = self.base_lr)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = opt,
            milestones = [self.stage_milestones[-1] + 10, self.stage_milestones[-1] + 20]
        )

        return [opt], [scheduler]


    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()


    def on_train_epoch_start(self):
        self._training_stage()

        # save last epcoh
        if self.current_epoch in self.stage_milestones:
            self._save_model(name = f"model_milestone_st{self.stage}_ep{self.current_epoch - 1}.pth")

        if self.stage == 3:
            self._save_model(name = f"model_epoch{self.current_epoch - 1}_step{self.global_step}", folder = "log/model_ckpt")


    def _training_stage(self):
        self.stage = 0
        for step in self.stage_milestones:
            if self.current_epoch < step:
                break
            else:
                self.stage += 1

        if self.stage == 0:
            self._train_mv()

        elif self.stage == 1:
            self._freeze_mv()

        elif self.stage == 2:
            self._freeze_mv()

        elif self.stage == 3:
            self._train_all()


    def _parse_cfg(self):
        print(self.cfg)

        self.stage_milestones = self.cfg["training"]["stage_milestones"]
        self.base_lr = self.cfg["training"]["base_lr"]
        self.aux_lr = self.cfg["training"]["aux_lr"]
        self.flow_pretrain_dir = self.cfg["training"]["flow_pretrain_dir"]
        self.train_lambda = self.cfg["training"]["train_lambda"]
        self.multi_frame_training = self.cfg["training"]["multi_frame_training"]


    def _freeze_mv(self):
        self._train_all()

        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = False

    def _train_mv(self):
        for p in self.model.parameters():
            p.requires_grad = False

        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = True

    def _train_all(self):
        for p in self.model.parameters():
            p.requires_grad = True


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _save_model(self, folder = "log/model_ckpt", name = None):
        if name == None:
            name = "model_ep{}_st{}.pth".format(self.current_epoch, self.global_step)

        torch.save(
            self.model.state_dict(), 
            os.path.join(folder, name)
        )


def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))
    

if __name__ == "__main__":
    L.seed_everything(3407)

    with open("config.json") as f:
        config = json.load(f)

    model_module = DCVC_TCM_Lit(config)
    # model_module = DCVCLit.load_from_checkpoint("lightning_logs/version_8/checkpoints/epoch=76-step=1243781.ckpt", cfg = config)

    if config["training"]["multi_frame_training"]:
        frame_num = 4
        interval = 1
        batch_size = config["training"]["batch_size"] // 2
    
    else:
        frame_num = 2
        interval = 2
        batch_size = config["training"]["batch_size"]


    train_dataset = Vimeo90K(
        root = "D:/vimeo_septuplet/", split_file="sep_trainlist.txt",
        frame_num = frame_num, interval = interval, rnd_frame_group = True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, persistent_workers=True, pin_memory = True
    )


    trainer = L.Trainer(
        max_epochs = 1000,
        # fast_dev_run = True
    )

    trainer.fit(
        model = model_module,
        train_dataloaders = train_dataloader,
        ckpt_path = "lightning_logs/version_0/checkpoints/epoch=4-step=80765.ckpt"
    )
    
