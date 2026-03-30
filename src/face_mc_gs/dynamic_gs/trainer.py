"""Training loop for canonical Gaussians + deformation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from face_mc_gs.dynamic_gs.dataset_manifest import ManifestDataset
from face_mc_gs.dynamic_gs.deformation import DeformationNetwork
from face_mc_gs.dynamic_gs.gaussian_model import GaussianModel
from face_mc_gs.dynamic_gs.losses import l1_rgb, neighbor_coherence_pairwise, scale_opacity_reg
from face_mc_gs.dynamic_gs.neighbor_graph import build_neighbor_indices
from face_mc_gs.dynamic_gs.renderer_simple import render_simple


class Trainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        ds = ManifestDataset(
            cfg["dataset_manifest"],
            root=Path(cfg["dataset_manifest"]).parent,
            image_size=tuple(cfg.get("image_size", [256, 256])),
            t_normalize=bool(cfg.get("time_normalize", True)),
        )
        self.loader = DataLoader(
            ds,
            batch_size=int(cfg["training"]["batch_size"]),
            shuffle=True,
            num_workers=0,
        )
        self.t_min = ds.t_min
        self.t_max = ds.t_max

        n = int(cfg["num_gaussians"])
        self.gauss = GaussianModel(n, sh_degree=int(cfg.get("sh_degree", 0))).to(self.device)
        dcfg = cfg.get("deformation", {})
        self.deform = DeformationNetwork(
            hidden_dim=int(dcfg.get("hidden_dim", 128)),
            num_layers=int(dcfg.get("num_layers", 4)),
            posenc_L=int(dcfg.get("posenc_L", 6)),
        ).to(self.device)

        self.opt = torch.optim.Adam(
            list(self.gauss.parameters()) + list(self.deform.parameters()),
            lr=float(cfg["training"]["lr"]),
            weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        )

        k = int(cfg.get("neighbor_k", 8))
        with torch.no_grad():
            idx = build_neighbor_indices(self.gauss.get_xyz().detach(), k)
        self.neigh_idx = idx.to(self.device)

        w = cfg.get("loss", {})
        self.w_rgb = float(w.get("w_rgb", 1.0))
        self.w_neighbor = float(w.get("w_neighbor", 0.0))
        self.w_scale_reg = float(w.get("w_scale_reg", 0.001))
        self.w_opacity_reg = float(w.get("w_opacity_reg", 0.01))

    def deform_positions(self, t_scalar: torch.Tensor) -> torch.Tensor:
        xyz = self.gauss.get_xyz()
        N = xyz.shape[0]
        t = t_scalar.expand(N, 1)
        dxyz, _, _ = self.deform(xyz, t)
        return xyz + dxyz

    def train(self) -> None:
        epochs = int(self.cfg["training"]["num_epochs"])
        out = Path(self.cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        H, W = tuple(self.cfg.get("image_size", [256, 256]))

        for ep in range(epochs):
            total = 0.0
            n_batch = 0
            for batch in tqdm(self.loader, desc=f"epoch {ep}"):
                img = batch["image"].to(self.device)
                t = batch["t"].to(self.device)
                K = batch["K"].to(self.device)
                R = batch["R_cw"].to(self.device)
                tvec = batch["t_cw"].to(self.device)

                B = img.shape[0]
                loss_acc = torch.tensor(0.0, device=self.device)
                for bi in range(B):
                    ti = t[bi]
                    means = self.deform_positions(ti)
                    col = self.gauss.get_color()
                    op = self.gauss.get_opacity()
                    sc = self.gauss.get_scale()
                    pred = render_simple(
                        means,
                        col,
                        op,
                        sc,
                        R[bi],
                        tvec[bi],
                        K[bi],
                        (H, W),
                    )
                    tgt = img[bi]
                    loss_acc = loss_acc + self.w_rgb * l1_rgb(pred, tgt)

                    if self.w_neighbor > 0:
                        means_j = self.deform_positions(ti + 1e-3)
                        loss_acc = loss_acc + self.w_neighbor * neighbor_coherence_pairwise(
                            means, means_j, self.neigh_idx
                        )

                reg = scale_opacity_reg(self.gauss.get_scale(), self.gauss.get_opacity())
                loss = loss_acc / B + self.w_scale_reg * reg + self.w_opacity_reg * reg

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total += float(loss.item())
                n_batch += 1

            print(f"epoch {ep} loss {total / max(n_batch, 1):.6f}")
            if (ep + 1) % int(self.cfg.get("checkpoint_every", 10)) == 0:
                self.save(out / f"checkpoint_ep{ep}.pt")

        self.save(out / "checkpoint.pt")

    def save(self, path: Path) -> None:
        torch.save(
            {
                "gauss": self.gauss.state_dict(),
                "deform": self.deform.state_dict(),
                "cfg": self.cfg,
                "t_min": self.t_min,
                "t_max": self.t_max,
            },
            path,
        )
