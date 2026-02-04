# scdiva/trainer.py
import os
from typing import Optional, List
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from transformers import Trainer

class ScDiVaMonitorTrainer(Trainer):
    def __init__(self, ema_smoothing_factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.ema_smoothing_factor = ema_smoothing_factor # 指数移动平均（EMA）的平滑因子，用于平滑训练损失的曲线
        self.ema_loss: Optional[float] = None
        self._recent_losses: List[float] = []

    def compute_loss(self, model, inputs, return_outputs=False):
        gene_ids = inputs["gene_ids"] # id
        target_values = inputs["target_values"] # 目标基因表达值
        attention_mask = inputs["attention_mask"] # 注意力掩码
        device = gene_ids.device

        outputs = model(
            gene_ids=gene_ids,
            target_values=target_values,
            attention_mask=attention_mask,
        )

        loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device, requires_grad=True)

        if model.training:
            if torch.isfinite(loss):
                cur_val = float(loss.detach())
                self._recent_losses.append(cur_val)
                if len(self._recent_losses) > self.args.logging_steps:
                    self._recent_losses.pop(0)
                if self.ema_loss is None:
                    self.ema_loss = cur_val
                else:
                    self.ema_loss = (
                        (1 - self.ema_smoothing_factor) * self.ema_loss
                        + self.ema_smoothing_factor * cur_val
                    )

            # 间隔特定步数记录训练过程中的各种指标
            is_logging_step = (
                self.state.global_step > 0
                and self.state.global_step % self.args.logging_steps == 0
                and self.state.is_world_process_zero
            ) 

            if is_logging_step:
                masked_indices = outputs.masked_indices
                if masked_indices is not None and masked_indices.any():
                    with torch.no_grad():
                        m_preds = outputs.pred_values[masked_indices]
                        m_targets = target_values[masked_indices]
                        m_logits = outputs.gene_logits[masked_indices]
                        m_gene_ids = gene_ids[masked_indices]

                        # 计算各种指标
                        mse = F.mse_loss(m_preds, m_targets)
                        mae = F.l1_loss(m_preds, m_targets)
                        rmse = torch.sqrt(mse)

                        pred_ids = torch.argmax(m_logits, dim=-1)
                        acc = (pred_ids == m_gene_ids).float().mean()

                        pearson_val, spearman_val, r2_val = 0.0, 0.0, 0.0
                        if m_preds.numel() > 10:
                            try:
                                mp_np = m_preds.detach().float().cpu().numpy()
                                mt_np = m_targets.detach().float().cpu().numpy()

                                pv, _ = pearsonr(mp_np, mt_np)
                                sv, _ = spearmanr(mp_np, mt_np)

                                pearson_val = float(pv) if not np.isnan(pv) else 0.0
                                spearman_val = float(sv) if not np.isnan(sv) else 0.0

                                ss_res = np.sum((mt_np - mp_np) ** 2)
                                ss_tot = np.sum((mt_np - np.mean(mt_np)) ** 2)
                                r2_val = 1 - (ss_res / (ss_tot + 1e-8))
                            except:
                                pass

                        log_payload = {
                            "loss/total": cur_val,
                            "loss/ema": self.ema_loss,
                            "train/mse": float(mse),
                            "train/rmse": float(rmse),
                            "train/mae": float(mae),
                            "train/accuracy": float(acc),
                            "train/pearson": float(pearson_val),
                            "train/spearman": float(spearman_val),
                            "train/r2": float(r2_val),
                        }
                        if outputs.loss_ce is not None:
                            log_payload["train/loss_ce"] = float(outputs.loss_ce)
                        if outputs.loss_mse is not None:
                            log_payload["train/loss_mse"] = float(outputs.loss_mse)

                        self.log(log_payload)

        return (loss, outputs) if return_outputs else loss

    def save_embeddings(self):
        if self.is_world_process_zero():
            print("Saving trained gene embeddings...")
            emb_layer = self.model.gene_embedding
            trained_weight = emb_layer.weight.detach().cpu().numpy()
            save_path = os.path.join(self.args.output_dir, "trained_gene2vec.npy")
            np.save(save_path, trained_weight)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            target_dir = output_dir if output_dir else self.args.output_dir
            emb_layer = self.model.gene_embedding
            trained_weight = emb_layer.weight.detach().cpu().numpy()
            save_path = os.path.join(target_dir, "trained_gene2vec.npy")
            np.save(save_path, trained_weight)
            print(f"✅ Embeddings saved to {save_path}")
