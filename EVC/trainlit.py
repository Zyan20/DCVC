import sys
sys.path.append("../../")

import math, yaml, os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.image_model import EVC
from util.dataset.Vimeo90K import Vimeo90K

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EVCLit(L.LightningModule):
    LAMBDA = [0.0075, 0.015, 0.03, 0.045]

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

        self._parse_cfg(global_config)

        self.model = EVC()
        self.model.apply(self._init_weights)


        self.sum_count = 0
        self.sum_out = {
            "loss": 0,
            "bit": 0,
            "bpp": 0,
            "bpp_y": 0,
            "bpp_z": 0,

            "MSE": 0,
            "PSNR": 0
        }

        self.train_q_index = 0


    def training_step(self, batch, idx):
        image = batch[:, 0, ...]

        out = self.model(image, self.model.q_scale[self.train_q_index])

        dist = F.mse_loss(out["x_hat"], image)
        rate = out["bpp"]

        loss = self.LAMBDA[self.train_q_index] * 255**2 * dist + rate

        self.sum_out["loss"] += loss.item()
        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()



        # log
        self.sum_count += 1
        self.sum_out["bpp_y"]    += out["bpp_y"].item()
        self.sum_out["bpp_z"]    += out["bpp_z"].item()
        self.sum_out["bpp"]      += out["bpp"].item()

        self.sum_out["MSE"]      += dist.item()


        if self.global_step % 50 == 0:
            for key in self.sum_out.keys():
                self.sum_out[key] /= self.sum_count

            self.sum_out["lr"]    = self.optimizers().optimizer.state_dict()['param_groups'][0]['lr']
            self.sum_out["PSNR"]  = mse2psnr(self.sum_out["MSE"])
            self.sum_out[f"q_scale_{self.q_index}"]  = self.model.q_scale[self.q_index].item()

            self.log_dict(self.sum_out)

            for key in self.sum_out.keys():
                self.sum_out[key] = 0

            self.sum_count = 0


            self.train_q_index = np.random.randint(0, 4)    # random q


    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr = self.base_lr)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = opt,
            milestones = [20],
            gamma = 0.1
        )

        return [opt], [scheduler]
    

    def on_train_epoch_end(self) -> None:
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

        self._save_model()


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

        
    def _parse_cfg(self, config):
        self.base_lr = config["training"]["base_lr"]
        self.train_lambda = config["training"]["train_lambda"]
        self.q_index = config["training"]["q_index"]

        self.name = config["name"]


    def _save_model(self, folder = "lightning_logs/model_ckpt", name = None):
        if name == None:
            name = "model_ep{}_st{}.pth".format(self.current_epoch, self.global_step)

        torch.save(
            self.model.state_dict(), 
            os.path.join(folder, name)
        )
        
        

def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))


global_config = None
if __name__ == "__main__":
    L.seed_everything(3407)

    with open("config.yaml") as f:
        global_config = yaml.safe_load(f)


    model_module = EVCLit()

    train_dataset = Vimeo90K(
        root = global_config["datasets"]["vimeo90k"]["root"], 
        split_file= global_config["datasets"]["vimeo90k"]["split_file"],
        frame_num = 1, interval = 1, rnd_frame_group = True
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = global_config["training"]["batch_size"], 
        shuffle = True, 
        num_workers = 1, persistent_workers=True, pin_memory = True
    )
    
    trainer = L.Trainer(
        max_epochs = 60,
        fast_dev_run = True,
    )

    trainer.fit(model_module, train_dataloader)

    

