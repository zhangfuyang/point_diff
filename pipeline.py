from typing import List, Optional, Tuple, Union
import torch
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor


class MyPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        room_num: int = 1,
        num_inference_steps: int = 1000,
    ) -> Tuple:
        # Sample gaussian noise to begin loop
        if isinstance(self.model.train_config.image_size, int):
            image_shape = (
                batch_size,
                self.model.train_config.in_channel,
                self.model.train_config.image_size,
                self.model.train_config.image_size,
            )
        else:
            image_shape = (batch_size, self.model.train_config.in_channel, 
                           *self.model.train_config.image_size)
        
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.model(image, t, room_num)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        return (image,)

