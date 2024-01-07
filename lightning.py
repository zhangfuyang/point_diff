import torch
import pytorch_lightning as pl
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

class MyLightningModule(pl.LightningModule):
    def __init__(self, model, config) -> None:
        super().__init__()
        self.config = config
        self.model = model

        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000, clip_sample=True, 
            prediction_type="epsilon", beta_schedule='squaredcos_cap_v2')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate)
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer, T_max=self.config.num_epochs, eta_min=1e-7),
            'interval': 'epoch',  # update the learning rate after each epoch
            'name': 'cosine_annealing_lr',
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def training_step(self, batch, batch_idx):
        x = batch['x'] # bs 16 64 64
        noise = torch.randn(x.shape, device=x.device)
        bs = x.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (bs,), device=x.device,
            dtype=torch.int64
        )

        noise_x = self.scheduler.add_noise(x, noise, timesteps)

        noise_pred = self.model(noise_x, timesteps)

        loss = F.mse_loss(noise_pred, noise)

        self.log('train_loss', loss, prog_bar=True, rank_zero_only=True)
        self.log('lr', 
                 self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"],
                 prog_bar=True, rank_zero_only=True)
            
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()

        self.scheduler.set_timesteps(100)
        timesteps = self.scheduler.timesteps

        z = torch.randn_like(batch['x']) # bs 3 512
        bs = z.shape[0]

        for i, t in enumerate(timesteps):
            model_output = self.model(z, t)

            z = self.scheduler.step(model_output, t, z).prev_sample
            #print(z.min(), z.max())
        
        latents = z
        if self.trainer.global_rank != 0:
            return

        # visualize & save latent
        image_base_path = os.path.join(self.logger.log_dir, 'image')
        os.makedirs(image_base_path, exist_ok=True)

        self.visualize_latent(latents, image_base_path, batch_idx)
        self.visualize_latent(batch['x'], image_base_path, batch_idx, surfix='gt')

    def visualize_latent(self, latents, image_base_path, batch_idx, surfix=''):
        os.makedirs(image_base_path, exist_ok=True)
        for i in range(latents.shape[0]):
            latent = latents[i].detach().cpu().numpy() 
            # latent: 3x512
            flag = latent[2] > 0
            latent = latent[:2, flag]
            latent = (latent + 1) / 2 * 64
            plt.figure()
            plt.scatter(latent[0], latent[1], s=0.5)
            plt.xlim(0, 64)
            plt.ylim(0, 64)
            plt.axis('off')
            plt.savefig(
                os.path.join(image_base_path, 
                             f'{self.global_step}_{batch_idx}_{i}_{surfix}.png'),
                             dpi=200)
            plt.close()



