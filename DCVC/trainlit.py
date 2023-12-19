import json, os

import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning as L

from src.models.DCVC_net import DCVC_net


class DCVCLit(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False

        self.stage = 0
        self.cfg = cfg
        self._parse_cfg()

        self.model = DCVC_net()
        self.model.opticFlow._load_Spynet(self.flow_pretrain_dir)
        # todo load pretrain MV


        self.mv_modules: list[nn.Module] = [
            self.model.opticFlow,
            self.model.mvEncoder,
            self.model.mvpriorEncoder,
            self.model.bitEstimator_z_mv,

            self.model.mvpriorDecoder,
            self.model.auto_regressive_mv,
            self.model.entropy_parameters_mv,

            self.model.mvDecoder_part1,
            self.model.mvDecoder_part2,
        ]

    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        ref_frame = batch[:, 0, ...].to(self.device)
        input_frame = batch[:, 1, ...].to(self.device)  # todo check device

        if self.stage == 0:
            out = self.model.forward_mv_generation(ref_frame, input_frame)

        else:
            out = self.model(ref_frame, input_frame)


        loss = self._get_loss(input_frame, out)
 
        for opt in self.optimizers():
            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt.step()
        

    def on_train_epoch_end(self):
        # lr scheduler
        for sch in self.lr_schedulers():
            sch.step()


        if self.current_epoch in self.stage_milestones:
            self._save_model(name = f"model_milestone_{self.stage}_{self.current_epoch}")

        if self.stage == 3:
            self._save_model()

        # change training stage
        self._training_stage()



    def _get_loss(self, input_frame, output):
        if self.stage == 0:
            distortion = F.mse_loss(input_frame, output["recon_image"])
            rate = output["bpp_mv_y"] + output["bpp_mv_z"]

        elif self.stage == 1:
            distortion = F.mse_loss(input_frame, output["recon_image"])
            rate = 0

        elif self.stage == 2:
            distortion = F.mse_loss(input_frame, output["recon_image"])
            rate = output["bpp_y"] + output["bpp_z"]

        elif self.stage == 3:
            distortion = F.mse_loss(input_frame, output["recon_image"])
            rate = output["bpp"]

        return distortion + rate


    def configure_optimizers(self):
        params_dict = dict(self.model.named_parameters())
        aux_params_key = {
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "bitEstimator" in name
        }

        all_params_key = {
            name
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        net_params_key = all_params_key - aux_params_key


        net_params = (params_dict[key] for key in sorted(net_params_key))
        net_opt = optim.AdamW(net_params, lr = self.base_lr)

        aux_params = (params_dict[key] for key in sorted(aux_params_key))
        aux_opt = optim.AdamW(aux_params, lr = self.aux_lr)

        net_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = net_opt,
            milestones = self.stage_milestones[:-1] + 10
        )
        aux_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = aux_params, 
            milestones = self.stage_milestones[:-1] + 10
        )

        return [net_opt, aux_opt], [net_scheduler, aux_scheduler]
        


    def _training_stage(self):
        self.stage = 0
        for step in self.stage_milestones:
            if self.global_step < step:
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


    def _parse_cfg(self):
        print(self.cfg)

        self.stage_milestones = self.cfg["training"]["stage_milestones"]
        self.base_lr = self.cfg["training"]["base_lr"]
        self.aux_lr = self.cfg["training"]["aux_lr"]
        self.flow_pretrain_dir = self.cfg["training"]["flow_pretrain_dir"]


    def _save_model(self, folder = "model_ckpt", name = None):
        if name == None:
            name = "model_{}_{}.pth".format(self.current_epoch, self.global_step)

        torch.save(
            self.model.state_dict(), 
            os.path.join(folder, name)
        )




if __name__ == "__main__":
    # with open("config.json") as f:
    #     config = json.load(f)

    # traning_cfg = config["training"]


    model = DCVC_net()
    

    

