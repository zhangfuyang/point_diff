import torch
from config import TrainingConfig
from dataset import PointCloudDataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from model import myModel
from lightning import MyLightningModule

train_config = TrainingConfig()

seed_everything(train_config.seed, workers=True)

train_dataset = PointCloudDataset(train_config.data_path)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_config.train_batch_size, shuffle=True,
    num_workers=8)

val_dataset = PointCloudDataset(train_config.data_path)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=train_config.eval_batch_size, shuffle=True,
    num_workers=8
)

model = myModel(train_config)

lightning_module = MyLightningModule(model, train_config)
#lightning_module.load_state_dict(
#    torch.load('image-butterflies/lightning_logs/version_2/checkpoints/last.ckpt', 
#               map_location='cpu')['state_dict'], strict=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=2, monitor='train_loss', mode='min', save_last=True, 
    filename='{epoch}-loss={train_loss:.4f}')

trainer = pl.Trainer(
    default_root_dir=train_config.output_dir,
    accelerator='gpu', strategy='ddp', devices=8, num_nodes=1,
    precision=16, max_epochs=train_config.num_epochs,
    val_check_interval=1., check_val_every_n_epoch=8,
    num_sanity_val_steps=1, limit_val_batches=1,
    accumulate_grad_batches=1, gradient_clip_val=1,
    detect_anomaly=False,
    callbacks=[checkpoint_callback])
trainer.fit(lightning_module, train_dataloader, val_dataloader) 
