from __future__ import annotations

import torch
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from einops import pack, unpack

from accelerate import Accelerator

# ema - apparently greatly helped with results
from ema_pytorch import EMA

from tiny_recursive_model.trm import TinyRecursiveModel

# Optional: optimized optimizers
try:
    from adam_atan2_pytorch import MuonAdamAtan2
except ImportError:
    MuonAdamAtan2 = None

from x_transformers import Encoder, Decoder

# helpers

def exists(v):
    return v is not None

def range_from_one(n):
    return range(1, n + 1)

def is_empty(t):
    return t.numel() == 0

# trainer

class Trainer(Module):
    def __init__(
        self,
        model: TinyRecursiveModel | Module,
        dataset: Dataset,
        optim_klass = AdamW,
        optim: Optimizer | None = None,
        learning_rate = 1e-4,
        muon_learning_rate = 1e-3,
        weight_decay = 1.,
        batch_size = 16,
        epochs = 2,
        halt_prob_thres = 0.5,
        max_recurrent_steps = 12,
        warmup_steps = 2000,
        ema_decay_rate = 0.999,
        switch_ema_every = 10000,
        accelerate_kwargs: dict = dict(),
        cpu = False,
        compile_model = True 
    ):
        super().__init__()

        self.accelerator = Accelerator(**accelerate_kwargs, cpu=cpu)

        self.batch_size = batch_size
        self.epochs = epochs

        # data
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = 4 if not cpu else 0,
            pin_memory = True if not cpu else False,
            prefetch_factor = 2 if not cpu else None
        )

        # optim

        if not exists(optim):
            if isinstance(model.network, (Encoder, Decoder)) and MuonAdamAtan2 is not None:
                optim = MuonAdamAtan2(
                    model.network.muon_parameters(),
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    muon_lr = muon_learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )
            else:
                optim = optim_klass(
                    model.parameters(),
                    lr = learning_rate / (batch_size * max_recurrent_steps),
                    weight_decay = weight_decay
                )

        self.optim = optim

        # scheduler

        self.scheduler = LambdaLR(self.optim, lambda step: min((step + 1) / warmup_steps, 1.0))

        # model

        self.model = model

        # ema model

        self.ema_model = None

        if self.accelerator.is_main_process:
            self.ema_model = EMA(
                model,
                beta = ema_decay_rate,
                update_model_with_ema_every = switch_ema_every,
                forward_method_names = ('predict',)
            )

        # recurrent and act related variables

        self.halt_prob_thres = halt_prob_thres
        self.max_recurrent_steps = max_recurrent_steps

        # UPGRADE: Torch Compile logic fixed
        if compile_model and not cpu and hasattr(torch, "compile"):
            # FIX: Switched from "reduce-overhead" to "default".
            # "reduce-overhead" uses CUDAGraphs which crash on recursive loops 
            # where outputs are fed back as inputs (memory overwrite).
            # "default" still provides kernel fusion (Triton) speedups without the crash.
            self.accelerator.print("Compiling model with torch.compile (mode='default')...")
            self.model = torch.compile(self.model, mode="default")

        # prepare maybe distributed
        self.model, self.optim, self.dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optim, self.dataloader, self.scheduler
        )

    def forward(self):

        for epoch in range_from_one(self.epochs):

            for dataset_input, dataset_output in self.dataloader:
                
                self.optim.zero_grad()

                outputs, latents = self.model.get_initial()
                
                # We need to tell the compiler that a new step is beginning if we were using CUDAGraphs,
                # but with mode="default", this loop is safe.
                for recurrent_step in range_from_one(self.max_recurrent_steps):
                    
                    loss, (main_loss, halt_loss), outputs, latents, pred, halt = self.model(
                        dataset_input, outputs, latents, labels = dataset_output
                    )

                    scaled_loss = loss / self.max_recurrent_steps

                    self.accelerator.backward(scaled_loss)
                    
                    if recurrent_step % 4 == 0 or recurrent_step == self.max_recurrent_steps:
                        self.accelerator.print(f'[{epoch} ({recurrent_step} / {self.max_recurrent_steps})] loss: {main_loss.mean().item():.3f} | halt loss: {halt_loss.mean().item():.3f}')

                    halt_mask = halt >= self.halt_prob_thres

                    if not halt_mask.any():
                        continue
                    
                    outputs = outputs[~halt_mask]
                    latents = latents[~halt_mask]
                    dataset_input = dataset_input[~halt_mask]
                    dataset_output = dataset_output[~halt_mask]

                    if is_empty(outputs):
                        break
                
                self.optim.step()
                self.scheduler.step()

                if self.accelerator.is_main_process:
                    self.ema_model.update()

        self.accelerator.print('complete')

        if self.accelerator.is_main_process:
            self.ema_model.copy_params_from_ema_to_model()