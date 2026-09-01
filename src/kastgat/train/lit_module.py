from __future__ import annotations

import torch
from torch.nn import functional as F
import lightning as L

from kastgat.data.schema import INTENT_CLASSES
from kastgat.models.kastgat import KASTGAT, encode_delta_traj
from kastgat.models.losses import FocalLoss, physics_penalty, trajectory_mse
from kastgat.utils.device import amp_enabled, resolve_device


class KASTGATModule(L.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("train", {})
        self.net = KASTGAT(
            in_dim=len(cfg.get("features", {}).get("node", [0] * 7)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            gat_heads=int(model_cfg.get("gat_heads", 4)),
            gat_layers=int(model_cfg.get("gat_layers", 2)),
            tcn_channels=list(model_cfg.get("tcn_channels", [64, 64])),
            tcn_kernel=int(model_cfg.get("tcn_kernel", 3)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            horizon=int(cfg.get("horizon_steps", 6)),
            n_intent=len(INTENT_CLASSES),
        )
        self.focal = FocalLoss(gamma=float(train_cfg.get("focal_gamma", 2.0)))
        self.l_traj = float(train_cfg.get("lambda_traj", 1.0))
        self.l_conf = float(train_cfg.get("lambda_conflict", 0.5))
        self.l_int = float(train_cfg.get("lambda_intent", 0.3))
        self.l_phy = float(train_cfg.get("lambda_physics", 0.1))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.wd = float(train_cfg.get("weight_decay", 1e-4))

    def forward(self, batch):
        return self.net(batch["x"], batch["node_mask"], batch["edge_index"])

    def _step(self, batch, stage: str):
        out = self.forward(batch)
        mask = batch["node_mask"][:, -1]
        last_pos = batch["x"][:, -1, :, :3]
        target_delta = encode_delta_traj(batch["y_traj"], last_pos)
        loss_t = trajectory_mse(out["traj_delta"], target_delta, mask)
        loss_c = self.focal(out["conflict_logits"], batch["y_conflict"], mask)
        loss_i = F.cross_entropy(
            out["intent_logits"].reshape(-1, out["intent_logits"].size(-1)),
            batch["y_intent"].reshape(-1),
            reduction="none",
        )
        loss_i = (loss_i * mask.reshape(-1)).sum() / mask.sum().clamp_min(1.0)
        loss_p = physics_penalty(out["traj"], mask)
        loss = self.l_traj * loss_t + self.l_conf * loss_c + self.l_int * loss_i + self.l_phy * loss_p
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/mse": loss_t,
                f"{stage}/conflict": loss_c,
                f"{stage}/intent": loss_i,
                f"{stage}/physics": loss_p,
            },
            prog_bar=stage == "val",
            on_step=False,
            on_epoch=True,
            batch_size=batch["x"].size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)


def precision_from_cfg(cfg: dict) -> str:
    device = resolve_device(cfg.get("device", "auto"))
    if amp_enabled(device, bool(cfg.get("train", {}).get("amp", True))):
        return "16-mixed"
    return "32-true"
