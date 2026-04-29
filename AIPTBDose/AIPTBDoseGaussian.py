import torch
import torch.nn as nn
from .AIPTBDose import AIPTBDose

class AIPTBDoseGaussian(AIPTBDose):
    def __init__(self, initial_temperature: float = 1.0, debug: bool = False):
        super().__init__(debug)
        # train sqrt value of temperature in order to avoid negaite values
        self.sqr_temperature = nn.Parameter(torch.tensor([initial_temperature], dtype=torch.float32))

    def compute_kernel(self, query: torch.Tensor) -> torch.Tensor:
        key = self.key_dataset["feature"]
        q_norm = torch.sum(query ** 2, dim=-1, keepdim=True)
        k_norm = torch.sum(key ** 2, dim=-1, keepdim=True)
        dist = q_norm + k_norm.transpose(-2, -1) - 2 * torch.matmul(query, key.transpose(-2, -1))

        kernel_logits = -dist / (self.sqr_temperature ** 2 + 1e-8)
        return kernel_logits