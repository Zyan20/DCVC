import sys, yaml, os, math, json
sys.path.append("../../")
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src.models.video_model import DMC

from util.dataset.Vimeo90K import Vimeo90K


class DCVC_DC_Lit(L.LightningModule):
    LAMBDA = [256, 512, 1024, 2048]
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        
        self.stage = 0
        self.cfg = cfg
        self._parse_cfg()

        self.model = DMC()
        self.model.apply(self._init_weights)
        self._load_Spynet(self.flow_pretrain_dir)

        self.mv_modules: list[nn.Module] = [
            self.model.optic_flow,

            self.model.mv_encoder,
            self.model.mv_hyper_prior_encoder,

            self.model.bit_estimator_z_mv,

            self.model.mv_hyper_prior_decoder,
            self.model.mv_decoder,
        ]
        

        self.param_q: list[nn.Module] = [
            self.model.y_q_basic,
            self.model.y_q_scale
        ]

        self.param_mv_q: list[nn.Module] = [
            self.model.mv_y_q_basic,
            self.model.mv_y_q_scale
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
        ref_frame = batch[:, 0, ...].to(self.device)

        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }

        for i in range(1, T):
            # start from 1
            input_frame = batch[:, i,...].to(self.device)
            out = self.model(
                input_frame, 
                dpb, 
                self.model.mv_y_q_scale[self.q_index],
                self.model.y_q_scale[self.q_index]
            )

            dpb = out["dpb"]

            loss += self._get_loss(input_frame, out, self.LAMBDA[self.q_index])

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

        self.sum_out["loss"]     += loss.item()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()

        # random
        self.q_index = np.random.randint(0, 4)
        
        # log
        if self.global_step % 50 == 0:
            for key in self.sum_out.keys():
                self.sum_out[key] /= self.sum_count

            self.sum_out["stage"] = float(self.stage)
            self.sum_out["lr"]    = self.optimizers().optimizer.state_dict()['param_groups'][0]['lr']

            for i in range(4):
                self.sum_out[f"y_q_{i}"]    = self.model.y_q_scale[i].item()
                self.sum_out[f"mv_y_q_{i}"] = self.model.mv_y_q_scale[i].item()

            self.log_dict(self.sum_out)

            for key in self.sum_out.keys():
                self.sum_out[key] = 0

            self.sum_count = 0


    def validation_step(self, batch, idx):
        B, T, C, H, W = batch.shape

        loss = 0
        bpp = 0
        psnr = 0
        me_psnr = 0
        mv_bpp = 0

        ref_frame = batch[:, 0, ...].to(self.device)

        dpb = {
            "ref_frame": ref_frame,
            "ref_feature": None,
            "ref_mv_feature": None,
            "ref_y": None,
            "ref_mv_y": None,
        }

        for i in range(1, T):
            # start from 1
            input_frame = batch[:, i,...].to(self.device)
            out = self.model(
                input_frame, 
                dpb, 
                self.model.mv_y_q_scale[self.q_index],
                self.model.y_q_scale[self.q_index]
            )

            dpb = out["dpb"]

            loss += self._get_loss(input_frame, out, self.train_lambda)

            bpp  += out["bpp"]
            psnr += out["PSNR"]

            me_psnr += out["ME_PSNR"]
            mv_bpp  += out["bpp_mv_y"] + out["bpp_mv_z"]


        loss = loss / (T - 1)
        bpp = bpp / (T - 1)
        psnr = psnr / (T - 1)
        me_psnr = me_psnr / (T - 1)
        mv_bpp = mv_bpp / (T - 1)

        self.log_dict({
            "val_ME_PSNR": me_psnr,
            "val_MV_BPP": mv_bpp,

            "val_BPP": bpp,
            "val_PSNR": psnr,

            "val_loss": loss,

        }, on_epoch = True)


    def _get_loss(self, input, output, frame_lambda):
        dist_me = F.mse_loss(input, output["warpped_image"])
        dist_recon = F.mse_loss(input, output["dpb"]["ref_frame"])

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

        if self.multi_frame_training:
            milestones = self.lr_milestones_multi
        
        else:
            milestones = self.lr_milestones


        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = opt,
            milestones = milestones,
            gamma = self.lr_gamma
        )

        return [opt], [scheduler]

    def on_train_start(self) -> None:
        # lr_scheduler = self.lr_schedulers()
        # for _ in range(8):
        #     lr_scheduler.step()
        
        print("Hack lr", self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])

    
    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

        self._log_q("log/q_info.txt")

    def on_train_epoch_start(self):
        # self._training_stage()
        self._train_all()
        self.stage = 3

        # save last epcoh
        name = "multi" if self.multi_frame_training else "single"
        if self.current_epoch in self.stage_milestones:
            self._save_model(
                name = f"{name}_milestone_stage{self.stage}_epoch{self.current_epoch - 1}.pth",
                folder = "log/model_ckpt"
            )

        if self.stage == 3:
            self._save_model(
                name = f"{name}_ep{self.current_epoch - 1}_st{self.global_step}.pth", 
                folder = "log/model_ckpt"
            )

    def _training_stage(self):
        # multi-frame training
        if self.multi_frame_training:
            self._train_all()
            self.stage = 3

            return
        
        # single-frame training
        self.stage = 0
        for step in self.stage_milestones:
            if self.current_epoch < step:
                break
            else:
                self.stage += 1
        
        if self.stage == 0:
            self._train_mv()

        elif self.stage == 1:
            self._train_all()
            self._freeze_mv()
            self._set_y_q_param(False)

        elif self.stage == 2:
            self._train_all()
            self._freeze_mv()

        elif self.stage == 3:
            self._train_all()


    def _parse_cfg(self):
        self.stage_milestones = self.cfg["training"]["stage_milestones"]
        self.base_lr = self.cfg["training"]["base_lr"]
        self.flow_pretrain_dir = self.cfg["training"]["flow_pretrain_dir"]
        self.train_lambda = self.cfg["training"]["train_lambda"]
        self.multi_frame_training = self.cfg["training"]["multi_frame_training"]
        
        self.lr_milestones_multi = self.cfg["training"]["lr_milestones_multi"]
        self.lr_milestones = self.cfg["training"]["lr_milestones"]
        self.lr_gamma = self.cfg["training"]["lr_gamma"]

        self.q_index = self.cfg["training"]["q_index"]


    def _freeze_mv(self):      
        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = False
        
        self._set_mv_y_q_param(False)

    def _train_mv(self):
        for p in self.model.parameters():
            p.requires_grad = False

        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = True

        self._set_mv_y_q_param(True)

    def _train_all(self):
        for p in self.model.parameters():
            p.requires_grad = True

    def _set_y_q_param(self, requires_grad):
        self.model.y_q_scale.requires_grad = requires_grad

    def _set_mv_y_q_param(self, requires_grad):
        self.model.mv_y_q_scale.requires_grad = requires_grad


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _save_model(self, folder = "log/model_ckpt", name = None):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        if name == None:
            name = "model_ep{}_st{}.pth".format(self.current_epoch, self.global_step)

        torch.save(
            self.model.state_dict(), 
            os.path.join(folder, name)
        )

    def _log_q(self, file):
        info = {}
        with open(file, "a+") as f:
            info["mv_q"] = self.model.mv_y_q_scale.detach().cpu().reshape(4).numpy().tolist()
            info["q"] = self.model.y_q_scale.detach().cpu().reshape(4).numpy().tolist()
            info["epoch"] = self.current_epoch

            json.dump(info, f, indent = 2)

    def _load_Spynet(self, model_dir):
        for i, layer in enumerate(self.model.optic_flow.moduleBasic):
            layer_name = f"modelL{i + 1}"

            layer.conv1.weight.data, layer.conv1.bias.data = load_flow_weight_form_np(model_dir, layer_name + '_F-1')
            layer.conv2.weight.data, layer.conv2.bias.data = load_flow_weight_form_np(model_dir, layer_name + '_F-2')
            layer.conv3.weight.data, layer.conv3.bias.data = load_flow_weight_form_np(model_dir, layer_name + '_F-3')
            layer.conv4.weight.data, layer.conv4.bias.data = load_flow_weight_form_np(model_dir, layer_name + '_F-4')
            layer.conv5.weight.data, layer.conv5.bias.data = load_flow_weight_form_np(model_dir, layer_name + '_F-5')

def load_flow_weight_form_np(me_model_dir, layername):
    index = layername.find('modelL')
    if index == -1:
        print('load models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = me_model_dir + name + '-weight.npy'
        modelbias = me_model_dir + name + '-bias.npy'
        weightnp = np.load(modelweight)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)
        


def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))


    

if __name__ == "__main__":
    L.seed_everything(3407)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        print(config)
    
    # model_module = DCVC_DC_Lit(config)
    model_module = DCVC_DC_Lit.load_from_checkpoint("log/HEM_main_randn/version_0/checkpoints/epoch=36-step=199245.ckpt", cfg = config)

    if config["training"]["multi_frame_training"]:
        frame_num = 4
        interval = 1
        batch_size = config["training"]["batch_size"] // 2
    
    else:
        frame_num = 2
        interval = 2
        batch_size = config["training"]["batch_size"]


    train_dataset = Vimeo90K(
        root = config["datasets"]["vimeo90k"]["root"], 
        split_file= config["datasets"]["vimeo90k"]["split_file"],
        frame_num = frame_num, interval = interval, rnd_frame_group = True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, 
        num_workers = 8, persistent_workers=True, pin_memory = True
    )

    logger = TensorBoardLogger(save_dir = "log", name = config["name"])
    trainer = L.Trainer(
        max_epochs = 80,
        # fast_dev_run = True,
        logger = logger,
        # strategy = "ddp_find_unused_parameters_true"
    )



    if config["training"]["resume"]:
        trainer.fit(
            model = model_module,
            train_dataloaders = train_dataloader,
            ckpt_path = config["training"]["ckpt"]
        )
    
    else:
        trainer.fit(
            model = model_module,
            train_dataloaders = train_dataloader,
        )
