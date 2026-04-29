import torch
import torch.nn as nn
import torch.nn.functional as F
from .AttentionDataset import AttentionDataset
from typing import Optional, Dict, Callable
from tqdm import tqdm

class AIPTBDose(nn.Module):
    """Attention based model class for IPTB classification with dosage of treatment"""
    def __init__(self, debug: bool = False):
        super().__init__()
        self.key_dataset = None # keys vectors
        self.debug = debug # show additional information during training

    def compute_kernel(self, query: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Implement compute_kernel in the subclass")

    def forward(
            self,
            query: torch.Tensor,
            query_indicies: list = None, # use mask if keys tensor contains queries
            mask_self_attention: bool = False) -> torch.Tensor:
        value = self.key_dataset["value"]

        kernel_logits = self.compute_kernel(query)

        if mask_self_attention:
            N = query.shape[0]
            M = value.shape[0]
            diag_mask = None
            if not query_indicies is None:
                diag_mask = torch.zeros((N,M), dtype=torch.bool, device=kernel_logits.device)
                for i in range(len(query_indicies)):
                    diag_mask[i, query_indicies[i]] = True
            kernel_logits = kernel_logits.masked_fill(diag_mask, float('-inf'))

        attn_weights = F.softmax(kernel_logits, dim=-1) 
        output = torch.matmul(attn_weights, value).clamp(0, 1.0) 

        return output

    def fit(
            self,
            datasets: Dict[str, AttentionDataset],
            optimizer: torch.optim.Optimizer,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            num_epochs: int = 10,
            batch_size: int = 10,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            metric_fns: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
    ) -> tuple[Dict[str, list], Dict[str, Optional[dict]]]:

        self.to(device)
        if metric_fns is None:
            metric_fns = {}

        train_history: Dict[str, list] = {split: [] for split in datasets}
        best_states: Dict[str, Optional[dict]] = {}
        best_loss: Dict[str, float] = {}
        for split in datasets:
            best_states[split] = None
            best_loss[split] = float("inf")

        self.key_dataset = datasets["train"]

        def _evaluate(dataset: AttentionDataset, mask_self_attention: bool = False) -> Dict[str, float]:
            self.eval()
            total_loss = 0.0
            metric_sums = {name: 0.0 for name in metric_fns}
            num_samples = dataset["feature"].shape[0]

            with torch.no_grad():
                indices_all = torch.arange(dataset["feature"].shape[0])

                for i in range(0, len(indices_all), batch_size):
                    limiter = i + batch_size
                    if limiter >= len(indices_all):
                        limiter = len(indices_all)
                    indices = indices_all[i:limiter]
                    query = dataset["feature"][indices, :]
                    values_q = dataset["value"][indices, :]

                    idx_key = None
                    idx_query = None

                    if mask_self_attention:
                        mask = torch.ones(len(indices_all), dtype=torch.bool)
                        mask[i:limiter] = False
                        idx_key = indices_all[mask]
                        idx_query = indices

                    output = self(
                        query=query,
                        query_indicies=idx_query,
                        mask_self_attention=mask_self_attention
                    ).squeeze(0)

                    b_size = output.shape[0]
                    Loss = loss_fn(output, values_q)
                    total_loss += Loss.item() * (b_size/num_samples)

                    for name, metric_fn in metric_fns.items():
                        y_pred_class = (output > 0.5).int()
                        metric_sums[name] += metric_fn(y_pred_class, values_q) * (b_size/num_samples)

            results = {"Loss": total_loss}
            for name in metric_fns:
                results[name] = metric_sums[name]

            return results

        if self.debug:
            pbar = range(num_epochs)
        else:
            pbar = tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            if self.debug:
                print(f"Epoch {epoch + 1}/{num_epochs}")

            self.train(True)  
            train_loss_running = 0.0
            num_train_samples = 0

            indices_all = torch.randperm(self.key_dataset["feature"].shape[0])

            for i in range(0,len(indices_all),batch_size):
                limiter = i + batch_size
                if limiter >= len(indices_all):
                    limiter = len(indices_all)
                indices = indices_all[i:limiter]
                half = indices.shape[0] // 2
                idx_query = indices[:half]
                idx_key = indices[half:]
                query = self.key_dataset["feature"][idx_query, :]
                values_q = self.key_dataset["value"][idx_query, :]

                optimizer.zero_grad()
                output = self(  
                    query=query,
                )
                Loss = loss_fn(output, values_q)
                Loss.backward()
                optimizer.step()

                b_size = query.shape[0]
                train_loss_running += Loss.item() * b_size
                num_train_samples += b_size

            eval_results = {}
            for split, dataset in datasets.items():
                eval_results[split] = _evaluate(dataset, split == "train")

            for split in datasets:
                epoch_log = {"epoch": epoch + 1, **eval_results[split]}
                train_history[split].append(epoch_log)

            pass
            loss_str = ""
            for split in list(best_states.keys()):
                loss_str += f"{split}={eval_results[split]["Loss"]:.4f} "

                current_loss = eval_results[split]["Loss"]
                if current_loss < best_loss[split]:
                    best_loss[split] = current_loss
                    best_states[split] = {
                        k: v.cpu().clone() for k, v in self.state_dict().items()
                    }
                    if self.debug:
                        print(f"New best result for {split} (Loss = {current_loss:.6f})")
            
            if not self.debug:
                pbar.set_postfix(loss=loss_str)
        if self.debug:
            print("Training finished!")
        return train_history, best_states

