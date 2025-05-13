from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Tuple, List, Callable

import matplotlib
matplotlib.use("agg")  
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#  Utilities
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
                    datefmt="%H:%M:%S")


def get_device(gpu_index: int) -> torch.device:
    """Return CUDA device if available (and index >=0) else CPU."""
    cuda_available = torch.cuda.is_available() and gpu_index >= 0
    dev = torch.device(f"cuda:{gpu_index}" if cuda_available else "cpu")
    LOG.info("Using device: %s", dev)
    return dev



#  Data generation
def build_spiral_dataset(
    *,
    batch_size: int = 1000,
    n_total: int = 1000,
    window: int = 100,
    subsample: int | None = None,
    theta_start: float = 0.0,
    theta_stop: float = 6 * np.pi,
    noise_std: float = 0.1,
    a: float = 0.0,
    b: float = 0.3,
    plot_example: bool = True,
    clustered_sampling: bool = False,
    num_clusters: int = 2,
    cluster_std_fraction: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Archimedean spirals with optional clustered irregular sampling."""
    # 1) Full θ‐grid and initial obs_t
    full_t = np.linspace(theta_start, theta_stop, num=n_total)
    obs_t  = full_t[:window]

    # 2) Build CW and CCW spirals
    ts = np.linspace(theta_start, theta_stop, num=n_total)
    r = a + b * ts
    xy_cw = np.stack((r * np.cos(ts), r * np.sin(ts)), axis=1)
    ts_ccw = ts[::-1]
    r_ccw = a + b * ts_ccw
    xy_ccw = np.stack((r_ccw * np.cos(ts_ccw), r_ccw * np.sin(ts_ccw)), axis=1)

    if plot_example:
        plt.figure()
        plt.plot(xy_cw[:,0], xy_cw[:,1], color="tab:green",  label="GT CW")
        plt.plot(xy_ccw[:,0],xy_ccw[:,1], color="tab:orange",label="GT CCW")
        plt.legend()
        plt.title("Ground‑Truth Spirals")
        plt.tight_layout()
        plt.savefig("ground_truth.png", dpi=200)
        plt.close()

    # 3) Collect clean + noisy full windows
    gt_list, obs_list = [], []
    valid_offset = n_total - 2 * window
    probs = [1.0 / valid_offset] * valid_offset
    for _ in range(batch_size):
        t0   = int(np.argmax(npr.multinomial(1, probs))) + window
        base = xy_cw if (npr.rand() > 0.5) else xy_ccw
        gt_list.append(base)
        win = base[t0 : t0 + window].copy()
        win += npr.randn(*win.shape) * noise_std
        obs_list.append(win)

    gt_traj  = np.stack(gt_list, axis=0)   # (B, n_total, 2)
    obs_traj = np.stack(obs_list, axis=0)  # (B, window, 2)

    # 4) Subsample into K points: uniform or clustered
    if subsample is not None and subsample < window:
        if clustered_sampling:
            # Evenly‑spaced cluster centers
            raw_centers = np.linspace(
                window / (num_clusters + 1),
                num_clusters * window / (num_clusters + 1),
                num_clusters,
            )
            centers = raw_centers.astype(int)

            std_idx = max(1, int(cluster_std_fraction * window))
            per_cl   = subsample // num_clusters
            rem      = subsample % num_clusters

            # draw until we have ≥ subsample unique indices
            idx_set = set()
            while len(idx_set) < subsample:
                draws = []
                for c in centers:
                    pts = npr.normal(loc=c, scale=std_idx, size=per_cl).astype(int)
                    pts = np.clip(pts, 0, window - 1)
                    draws.extend(pts.tolist())
                for i in range(rem):
                    v = int(np.clip(
                        npr.normal(loc=centers[i % num_clusters], scale=std_idx),
                        0, window - 1
                    ))
                    draws.append(v)
                idx_set.update(draws)

            # **FIX**: randomly pick exactly `subsample` unique indices
            uniques = np.array(list(idx_set), dtype=int)
            chosen  = npr.choice(uniques, size=subsample, replace=False)
            idx_sub = np.sort(chosen)

        else:
            # uniform random subsample
            idx_sub = np.sort(npr.choice(window, subsample, replace=False))

        # apply to both obs_traj and obs_t
        obs_traj = obs_traj[:, idx_sub, :]
        obs_t    = obs_t[idx_sub]

    return gt_traj, obs_traj, full_t, obs_t






def build_changing_freq_sine_dataset(
    *,
    batch_size: int = 1000,
    n_total: int = 1000,
    window: int = 100,
    subsample: int | None = None,
    t_start: float = 0.0,
    t_stop: float = 10.0,
    noise_std: float = 0.1,
    f0: float = 1.0,     # start frequency (cycles per unit t)
    f1: float = 5.0,     # end frequency
    plot_example: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of 2D chirp trajectories (time vs sin) with
    linearly changing frequency, plus Gaussian noise on the y-axis.

    Returns:
      gt_traj:  (B, n_total, 2) clean full trajectories (t, sin)
      obs_traj: (B, K, 2)      noisy observed window
      full_t:   (n_total,)     global time grid
      obs_t:    (K,)           timestamps for the observations
    """
    # 1) full time grid
    full_t = np.linspace(t_start, t_stop, num=n_total)

    # 2) linearly ramp frequency and integrate to get phase
    Δt = t_stop - t_start
    f_t = f0 + (f1 - f0) * (full_t - t_start) / Δt
    # φ(t) = 2π ∫₀ᵗ f(τ) dτ = 2π [ f0·(t−t_start) + ½(f1−f0)((t−t_start)² / Δt) ]
    rel = full_t - t_start
    phase = 2 * np.pi * (f0 * rel + 0.5 * (f1 - f0) * rel**2 / Δt)

    # 3) build 2D curve: x = time, y = sin(phase)
    y = np.sin(phase)
    xy = np.stack((full_t, y), axis=1)  # shape (n_total, 2)

    # optional example plot
    if plot_example:
        plt.figure()
        plt.plot(xy[:, 0], xy[:, 1], label="Chirp (clean)")
        plt.xlabel("t")
        plt.ylabel("sin φ(t)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("sine_chirp_ground_truth.png", dpi=300)
        LOG.info("Saved ./sine_chirp_ground_truth.png")
        plt.close()

    # 4) sample windows + add noise
    gt_list, obs_list = [], []
    valid_offset = n_total - 2 * window
    probs = [1.0 / valid_offset] * valid_offset

    for _ in range(batch_size):
        # random window start (avoid edges)
        t0 = int(np.argmax(npr.multinomial(1, probs))) + window
        gt_list.append(xy.copy())

        # slice out window and add noise on y only
        win = xy[t0 : t0 + window].copy()
        win[:, 1] += npr.randn(window) * noise_std
        obs_list.append(win)

    gt_traj = np.stack(gt_list, axis=0)   # (B, n_total, 2)
    obs_traj = np.stack(obs_list, axis=0) # (B, window, 2)
    obs_t = full_t[:window]

    # 5) optional subsample inside the window
    if subsample is not None and subsample < window:
        idx = np.sort(npr.choice(window, subsample, replace=False))
        obs_traj = obs_traj[:, idx, :]
        obs_t = obs_t[idx]

    return gt_traj, obs_traj, full_t, obs_t


#  Model definitions
class ODEDynamics(nn.Module):
    """Latent ODE function  ẋ = f_theta(z, t)."""

    def __init__(self, z_dim: int = 4, hidden: int = 20):
        super().__init__()
        self.nfe = 0  # number of function evaluations (for monitoring)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        self.nfe += 1
        return self.net(z)


class EncoderRNN(nn.Module):
    """RNN that parameterizes q(z0 | x_{1:K})."""

    def __init__(self, z_dim: int = 4, x_dim: int = 2, hidden: int = 25, batch: int = 1):
        super().__init__()
        self.hid_size = hidden
        self.batch = batch
        self.i2h = nn.Linear(x_dim + hidden, hidden)
        self.h2o = nn.Linear(hidden, 2 * z_dim)  # mean & logvar

    def init_h(self) -> torch.Tensor:
        return torch.zeros(self.batch, self.hid_size)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat((x_t, h_t), dim=1)
        h_next = torch.tanh(self.i2h(concat))
        out = self.h2o(h_next)
        return out, h_next


class ObservationDecoder(nn.Module):
    """Maps latent state to observation space."""

    def __init__(self, z_dim: int = 4, x_dim: int = 2, hidden: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(z)


class EMA:
    """Exponential moving average tracker."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.avg = 0.0
        self.initialized = False

    def update(self, value: float) -> None:
        if not self.initialized:
            self.avg = value
            self.initialized = True
        else:
            self.avg = self.avg * self.decay + value * (1.0 - self.decay)



# Loss helpers
def log_normal_pdf(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    const = torch.log(torch.tensor(2.0 * np.pi, device=x.device))
    return -0.5 * (const + logvar + (x - mean) ** 2 / torch.exp(logvar))


def kl_divergence(mu_q: torch.Tensor, lv_q: torch.Tensor) -> torch.Tensor:
    """KL(q || p) where p is N(0, I)."""
    var_q = torch.exp(lv_q)
    return 0.5 * (var_q + mu_q ** 2 - 1.0 - lv_q)





# Training / evaluation
def train_epoch(
    *,
    func: ODEDynamics,
    encoder: EncoderRNN,
    decoder: ObservationDecoder,
    optimizer: optim.Optimizer,
    obs_traj: torch.Tensor,
    obs_time: torch.Tensor,
    noise_std: float,
) -> float:
    """One optimization step over the full batch (ELBO)."""
    optimizer.zero_grad()

    # q(z0 | x_{1:K}) 
    h = encoder.init_h().to(obs_traj.device)
    for t in reversed(range(obs_traj.size(1))):
        out_t, h = encoder(obs_traj[:, t], h)
    q_mu, q_logvar = out_t.chunk(2, dim=1)
    eps = torch.randn_like(q_mu)
    z0 = q_mu + eps * torch.exp(0.5 * q_logvar)

    # latent dynamics 
    from torchdiffeq import odeint  # imported here for reduced clutter

    z_pred = odeint(func, z0, obs_time).permute(1, 0, 2)  # (B, K, z_dim)
    x_pred = decoder(z_pred)  # (B, K, x_dim)

    # ELBO 
    noise_logvar = 2.0 * torch.log(torch.tensor(noise_std, device=obs_traj.device))
    log_px = log_normal_pdf(obs_traj, x_pred, noise_logvar).sum((-1, -2))
    kl = kl_divergence(q_mu, q_logvar).sum(-1)
    neg_elbo = -(log_px - kl).mean()
    neg_elbo.backward()
    optimizer.step()
    return neg_elbo.item()


# visualization helper
def plot_latent_vector_field(
    func: ODEDynamics,
    device: torch.device,
    dim1: int = 0,
    dim2: int = 1,
    grid_min: float = -3.0,
    grid_max: float =  3.0,
    grid_n: int   = 21,
    out_path: str = "latent_vecfield.png",
):
    """
    Samples a grid in (z_dim dim1 × z_dim dim2), evaluates dot{z} = f(z),
    and streamplots the normalized vector field.
    """
    import matplotlib.pyplot as plt

    # 1) build meshgrid in two latent dims
    zs = np.linspace(grid_min, grid_max, grid_n)
    X, Y = np.meshgrid(zs, zs)                   # shape (N,N)
    # flatten to (N*N, 2), then pad zeros for other dims
    Zflat = np.zeros((grid_n * grid_n, func.net[0].in_features), dtype=np.float32)
    Zflat[:, dim1] = X.ravel()
    Zflat[:, dim2] = Y.ravel()

    # 2) eval f at t=0
    with torch.no_grad():
        Zt = torch.from_numpy(Zflat).to(device)
        dZ = func(torch.tensor(0.0, device=device), Zt).cpu().numpy()

    # 3) extract the two dims and normalize for nice arrows
    U = dZ[:, dim1].reshape(grid_n, grid_n)
    V = dZ[:, dim2].reshape(grid_n, grid_n)
    M = np.sqrt(U**2 + V**2) + 1e-8
    U, V = U / M, V / M

    # 4) plot
    plt.figure(figsize=(5,5))
    plt.streamplot(X, Y, U, V, density=1.0, color="black")
    plt.title(f"Latent vector field on dims {dim1},{dim2}")
    plt.xlabel(f"z[{dim1}]")
    plt.ylabel(f"z[{dim2}]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    LOG.info("Saved %s", out_path)






def run_visualization(
    *,
    func: ODEDynamics,
    encoder: EncoderRNN,
    decoder: ObservationDecoder,
    full_traj: torch.Tensor,
    obs_traj: torch.Tensor,
    obs_time: torch.Tensor,
    theta_start: float,
    device: torch.device,
    out_path: str = "vis.png",
):
    """Generate forward/backward reconstructions from inferred z0 and plot."""
    func.eval()
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        h = encoder.init_h().to(device)
        for t in reversed(range(obs_traj.size(1))):
            out_t, h = encoder(obs_traj[:, t], h)
        q_mu, q_logvar = out_t.chunk(2, dim=1)
        z0 = q_mu  # use mean for visualization stability

        from torchdiffeq import odeint

        # forward (future) and backward (past) integration grids
        t_fwd = torch.linspace(theta_start, 2 * np.pi, 2000, device=device)
        t_bwd = torch.linspace(theta_start, -np.pi, 2000, device=device)

        z_fwd = odeint(func, z0[0], t_fwd)
        z_bwd = odeint(func, z0[0], t_bwd)
        x_fwd = decoder(z_fwd).cpu().numpy()
        x_bwd = decoder(z_bwd).cpu().numpy()

    true_xy = full_traj[0].cpu().numpy()
    obs_xy = obs_traj[0].cpu().numpy()

    # plot boundaries padded by 1 unit
    x_min, x_max = true_xy[:, 0].min() - 1, true_xy[:, 0].max() + 1
    y_min, y_max = true_xy[:, 1].min() - 1, true_xy[:, 1].max() + 1

    plt.figure(figsize=(6, 6))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot(true_xy[:, 0], true_xy[:, 1], "g", label="Ground Truth")
    plt.plot(x_fwd[:, 0], x_fwd[:, 1], "r--", label="Forward Reconstruction")
    plt.plot(x_bwd[:, 0], x_bwd[:, 1], "c", label="Backward Reconstruction")
    plt.scatter(obs_xy[:, 0], obs_xy[:, 1], s=5, label="Observations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    LOG.info("Saved %s", out_path)
    plt.close()

    # plot_latent_vector_field(
    #     func=func,
    #     device=device,
    #     dim1=0,
    #     dim2=1,
    #     grid_min=-3,
    #     grid_max=3,
    #     grid_n=25,
    #     out_path="latent_vecfield.png",
    # )
    



#  Main

def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latent ODE on synthetic spirals")
    p.add_argument("--adjoint", type=eval, default=False,
                   help="use adjoint ODE solver (True/False)")
    p.add_argument("--visualize", type=eval, default=False,
                   help="generate reconstruction plot (True/False)")
    p.add_argument("--niters", type=int, default=2000,
                   help="training iterations")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    p.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    p.add_argument("--train_dir", type=str, default=None,
                   help="checkpoint directory")
    p.add_argument("--nsample", type=int, default=100,
                   help="observed points per window")
    p.add_argument("--subsample", type=int, default=100,
                   help="optional subsample inside the window")
    return p.parse_args()


def main() -> None:
    args = parse_arguments()
    device = get_device(args.gpu)

    # dataset
    # Learning spiral distribution
    gt_np, obs_np, full_t_np, obs_t_np = build_spiral_dataset(
        num_clusters=3,
        batch_size=1000,
        n_total=1000,
        window=args.nsample,
        subsample=args.subsample,
        clustered_sampling=True,
        theta_start=0.0,
        theta_stop=6 * np.pi,
        noise_std=0.1,
        a=0.0,
        b=0.3,
        plot_example=True,
    )

    # Learning sinusoid
    # gt_np, obs_np, full_t_np, obs_t_np = build_changing_freq_sine_dataset(
    #     batch_size=1000,
    #     n_total=1000,
    #     window=args.nsample,
    #     subsample=args.subsample,
    #     t_start=0.0,
    #     t_stop=10.0,
    #     noise_std=0.1,
    #     f0=0.5,
    #     f1=1.0,
    #     plot_example=True,
    # )


    gt = torch.from_numpy(gt_np).float().to(device)
    obs = torch.from_numpy(obs_np).float().to(device)
    obs_t = torch.from_numpy(obs_t_np).float().to(device)

    # models
    z_dim = 4
    hidden = 20
    rnn_hidden = 25
    func = ODEDynamics(z_dim, hidden).to(device)
    encoder = EncoderRNN(z_dim, 2, rnn_hidden, batch=gt.size(0)).to(device)
    decoder = ObservationDecoder(z_dim, 2, hidden).to(device)

    optimizer = optim.Adam(list(func.parameters()) +
                           list(encoder.parameters()) +
                           list(decoder.parameters()), lr=args.lr)

    # checkpoint reload if any
    if args.train_dir:
        os.makedirs(args.train_dir, exist_ok=True)
        ckpt_path = os.path.join(args.train_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            func.load_state_dict(state["func"])
            encoder.load_state_dict(state["enc"])
            decoder.load_state_dict(state["dec"])
            optimizer.load_state_dict(state["opt"])
            LOG.info("Loaded checkpoint %s", ckpt_path)

    # training loop
    ema = EMA()
    try:
        for it in range(1, args.niters + 1):
            loss_val = train_epoch(
                func=func,
                encoder=encoder,
                decoder=decoder,
                optimizer=optimizer,
                obs_traj=obs,
                obs_time=obs_t,
                noise_std=0.1,
            )
            ema.update(loss_val)
            if it % 50 == 0 or it == 1:
                LOG.info("Iter %d/%d - avg ELBO: %.4f", it, args.niters, -ema.avg)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user - saving checkpoint…")
    finally:
        if args.train_dir:
            ckpt_path = os.path.join(args.train_dir, "ckpt.pth")
            torch.save({
                "func": func.state_dict(),
                "enc": encoder.state_dict(),
                "dec": decoder.state_dict(),
                "opt": optimizer.state_dict(),
            }, ckpt_path)
            LOG.info("Checkpoint saved to %s", ckpt_path)

    LOG.info("Training complete")

    if args.visualize:
        run_visualization(
            func=func,
            encoder=encoder,
            decoder=decoder,
            full_traj=gt,
            obs_traj=obs,
            obs_time=obs_t,
            theta_start=0.0,
            device=device,
            out_path="vis.png",
        )


if __name__ == "__main__":
    main()
