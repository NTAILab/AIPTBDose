import torch
import torch.nn as nn
from .AIPTBDose import AIPTBDose

class AIPTBDoseDotProduct(AIPTBDose):

    def __init__(self, dim: int, min_A_weight: float = 0.3, debug: bool = False):
        super().__init__(debug)
        self.sq_dim = 1/torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        self.W_k = nn.Parameter(torch.rand((dim, dim*2+1), dtype=torch.float32))
        self.W_q = nn.Parameter(torch.rand((dim, dim*2+1), dtype=torch.float32))
        self.min_A_weight = 1/(dim*2+1) # doage feature weight. Important for datasets with many features
        if self.min_A_weight < min_A_weight:
            self.min_A_weight = min_A_weight
        self.W_f = torch.eye(dim*2+1)
        self.W_f[0,0] = self.min_A_weight
        other_weights = (1 - self.min_A_weight)/(dim*2)
        for i in range(1, dim*2+1):
            self.W_f[i, i] = other_weights
        self.W_f = self.W_f.sqrt()

    def compute_kernel(self, query: torch.Tensor) -> torch.Tensor:
        key = self.key_dataset["feature"]

        k = self.W_k @ self.W_f @ key.T
        if len(query.shape) == 1:
            query = query.unsqueeze(0)
        q = self.W_q @ self.W_f @ query.T

        kernel_logits = self.sq_dim * q.T @ k
        return kernel_logits