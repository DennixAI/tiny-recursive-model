from __future__ import annotations
from contextlib import nullcontext

import torch
from torch import nn, cat, arange, tensor
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, reduce, pack, unpack

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_empty(t):
    return t.numel() == 0

def range_from_one(n):
    return range(1, n + 1)

# classes

class TinyRecursiveModel(Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        network: Module,
        num_refinement_blocks = 3,   # T in paper
        num_latent_refinements = 6,  # n in paper
        halt_loss_weight = 1.,
        num_register_tokens = 0
    ):
        super().__init__()
        assert num_refinement_blocks > 1

        self.input_embed = nn.Embedding(num_tokens, dim)
        self.output_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(dim) * 1e-2)

        self.network = network

        self.num_latent_refinements = num_latent_refinements
        self.num_refinement_blocks = num_refinement_blocks

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim) * 1e-2)

        self.to_pred = nn.Linear(dim, num_tokens, bias = False)

        self.to_halt_pred = nn.Sequential(
            nn.Linear(dim, 1, bias = False),
            nn.Sigmoid()
        )

        self.halt_loss_weight = halt_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    def get_initial(self):
        outputs = self.output_init_embed
        latents = self.latent_init_embed
        return outputs, latents

    def embed_inputs_with_registers(self, seq):
        batch = seq.shape[0]
        inputs = self.input_embed(seq)
        registers = repeat(self.register_tokens, 'n d -> b n d', b = batch)
        inputs, packed_shape = pack([registers, inputs], 'b * d')
        return inputs, packed_shape

    def refine_latent_then_output_once(self, inputs, outputs, latents):
        # inputs, outputs, latents are all (b, n, d)
        
        # Latent loop
        for _ in range(self.num_latent_refinements):
            # Simple addition fusion
            combined = outputs.add(latents).add(inputs)
            latents = self.network(combined)

        # Output refinement
        combined_out = outputs.add(latents)
        outputs = self.network(combined_out)

        return outputs, latents

    def deep_refinement(self, inputs, outputs, latents):
        for step in range_from_one(self.num_refinement_blocks):
            is_last = step == self.num_refinement_blocks
            context = torch.no_grad if not is_last else nullcontext

            with context():
                outputs, latents = self.refine_latent_then_output_once(inputs, outputs, latents)

        return outputs, latents

    @torch.no_grad()
    def predict(
        self,
        seq,
        halt_prob_thres = 0.5,
        max_deep_refinement_steps = 12
    ):
        batch = seq.shape[0]
        inputs, packed_shape = self.embed_inputs_with_registers(seq)
        outputs, latents = self.get_initial()

        active_batch_indices = arange(batch, device = self.device, dtype = torch.float32)

        preds = []
        exited_step_indices = []
        exited_batch_indices = []

        for step in range_from_one(max_deep_refinement_steps):
            is_last = step == max_deep_refinement_steps

            outputs, latents = self.deep_refinement(inputs, outputs, latents)

            # Native mean reduction for speed (b n d -> b d) -> (b 1) -> (b)
            halt_prob = self.to_halt_pred(outputs.mean(dim=1)).squeeze(-1)

            should_halt = (halt_prob >= halt_prob_thres) | is_last

            if not should_halt.any():
                continue

            registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')
            
            # Save predictions
            pred = self.to_pred(outputs_for_pred[should_halt])
            preds.append(pred)

            count = should_halt.sum().item()
            exited_step_indices.extend([step] * count)
            exited_batch_indices.append(active_batch_indices[should_halt])

            if is_last:
                continue

            # Filter for next round
            not_halt = ~should_halt
            inputs = inputs[not_halt]
            outputs = outputs[not_halt]
            latents = latents[not_halt]
            active_batch_indices = active_batch_indices[not_halt]

            if is_empty(outputs):
                break

        preds = cat(preds).argmax(dim = -1)
        
        # FIX: Ensure tensor is created on the correct device
        exited_step_indices = tensor(exited_step_indices, device=self.device)
        
        exited_batch_indices = cat(exited_batch_indices)
        
        sort_indices = exited_batch_indices.argsort(dim = -1)

        return preds[sort_indices], exited_step_indices[sort_indices]

    def forward(
        self,
        seq,
        outputs,
        latents,
        labels = None
    ):
        inputs, packed_shape = self.embed_inputs_with_registers(seq)

        outputs, latents = self.deep_refinement(inputs, outputs, latents)

        registers, outputs_for_pred = unpack(outputs, packed_shape, 'b * d')

        pred = self.to_pred(outputs_for_pred)

        # Optimization: use native mean
        halt_prob = self.to_halt_pred(outputs.mean(dim=1)).squeeze(-1)

        # Detach state to allow truncated BPTT via the Trainer loop
        outputs, latents = outputs.detach(), latents.detach()
        return_package = (outputs, latents, pred, halt_prob)

        if not exists(labels):
            return return_package

        # Calculate losses
        # Reshape for cross entropy: (b, n, tokens) -> (b, tokens, n)
        loss = F.cross_entropy(pred.transpose(1, 2), labels, reduction = 'none')
        loss = loss.mean(dim=-1) # Reduce sequence

        is_all_correct = (pred.argmax(dim = -1) == labels).all(dim = -1)
        halt_loss = F.binary_cross_entropy(halt_prob, is_all_correct.float(), reduction = 'none')

        total_loss = (loss + halt_loss * self.halt_loss_weight)

        losses = (loss, halt_loss)

        return (total_loss.sum(), losses, *return_package)