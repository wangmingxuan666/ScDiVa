"""
ScDiVa Inference SDK
High-level wrappers for single-cell analysis tasks.
"""
import torch
import numpy as np
from modeling_scdiva import ScDiVaModel

class ScDiVaInference:
    def __init__(self, model_name: str = "warming666/ScDiVa", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing ScDiVa on {self.device}...")
        self.model = ScDiVaModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _preprocess(self, adata) -> torch.Tensor:
        # Placeholder for preprocessing (normalization, etc.)
        # In real usage, this aligns genes and converts to tensor
        if hasattr(adata.X, "toarray"):
            expr = adata.X.toarray()
        else:
            expr = adata.X
        return torch.tensor(expr, dtype=torch.float32).to(self.device)

    def annotate(self, adata):
        data = self._preprocess(adata)
        with torch.no_grad():
            logits = self.model.predict(data, task="annotation")
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def integrate_batches(self, adata_list):
        # Placeholder for integration logic
        results = []
        for adata in adata_list:
            data = self._preprocess(adata)
            with torch.no_grad():
                emb = self.model.encode(data)["latent"]
                results.append(emb.cpu().numpy())
        return np.concatenate(results, axis=0)
