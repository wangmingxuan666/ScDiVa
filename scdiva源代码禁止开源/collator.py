# scdiva/collator.py
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
from .config import Config

@dataclass
class DataCollatorForCell:
    pad_token_id: int
    max_len: int = 1200
    _first_batch_logged: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not self._first_batch_logged:
            print(f"✅ [DataCollator] 成功读取到第一批数据！Batch size: {len(features)}")
            self._first_batch_logged = True

        batch_gene_ids = []
        batch_target_values = []
        batch_masks = []

        max_batch_len = 0
        for f in features:
            g = f["gene_ids"]
            max_batch_len = max(max_batch_len, min(len(g), self.max_len))

        cfg = Config()

        for f in features:
            g_ids = f["gene_ids"]
            t_vals = f["target_values"]

            if hasattr(g_ids, "tolist"):
                g_ids = g_ids.tolist()
            if hasattr(t_vals, "tolist"):
                t_vals = t_vals.tolist()

            seq_len = min(len(g_ids), self.max_len)
            g_ids = g_ids[:seq_len]
            t_vals = t_vals[:seq_len]

            pad_len = max_batch_len - seq_len
            padded_g_ids = g_ids + [self.pad_token_id] * pad_len
            padded_t_vals = t_vals + [0.0] * pad_len

            # attention_mask：只把 41815(-2 padding) 当 padding=0；41816/41817 可见=1
            att_mask_core = []
            for gid, val in zip(g_ids, t_vals):
                gid_i = int(gid)
                val_f = float(val)
                if gid_i == cfg.PAD_FILL_GENE_ID or val_f == -2.0:
                    att_mask_core.append(0)
                else:
                    att_mask_core.append(1)
            att_mask = att_mask_core + [0] * pad_len

            batch_gene_ids.append(padded_g_ids)
            batch_target_values.append(padded_t_vals)
            batch_masks.append(att_mask)

        return {
            "gene_ids": torch.tensor(batch_gene_ids, dtype=torch.long),
            "target_values": torch.tensor(batch_target_values, dtype=torch.float),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
            # label_names 会指定 target_values 和 gene_ids
            "labels": torch.tensor(batch_target_values, dtype=torch.float),
        }
