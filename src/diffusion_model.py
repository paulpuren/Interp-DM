"""
A diffusion model with a FLEX backbone for temporal interpolation.
"""

from __future__ import annotations  # enables |‐based type unions on Python <3.10
import copy  # NOTE: not used at present, but kept in case it is required downstream
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def logsnr_schedule_cosine(
        t: torch.Tensor,
        logsnr_min: float = -20.0,
        logsnr_max: float = 20.0,
        shift: float = 1.0,
    ) -> torch.Tensor:
    """Cosine log‑SNR schedule from Nichol & Dhariwal (2021) with optional shift.

    This schedule maps a normalized continuous time‐step t ∈ [0,1] to the log‑signal‑
    to‑noise‑ratio (log‑SNR) used in diffusion models.  The closed‑form expression below
    is derived from the original paper;

    Args:
        t: Normalised time in [0, 1]; arbitrary leading dimensions are allowed, but
           shape (batch,) is most common.
        logsnr_min / logsnr_max: Lower/upper bounds of the log‑SNR range.
        shift: Scalar multiplier that uniformly shifts the curve along the vertical axis.

    Returns:
        logsnr: Tensor with the same leading dimensions as t containing log‑SNR values.
    """
    # The transformation below is numerically stable for the specified default range.
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b

    return -2.0 * torch.log(torch.tan(a * t + b) * shift)


# Wrapper that additionally provides α and σ — the square‑rooted signal/noise weights
# used by most modern parameterisations (x₀, ε, or v) 
def get_logsnr_alpha_sigma(
        time: torch.Tensor,
        shift: float = 16.0,  
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(logsnr, α, σ)` broadcastable to (B, 1, 1, 1).

    The helper expands the 1‑D time tensor into per‑sample log‑SNR plus its derived
    coefficients α and σ such that subsequent point‑wise arithmetic broadcasts cleanly
    over spatial dimensions.

    Note: 16.0 follows the "imagen" implementation; set to 1.0 to match
    """
    logsnr = logsnr_schedule_cosine(time, shift=shift)[:, None, None, None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))       # α = (SNR / (1+SNR))^½
    sigma = torch.sqrt(torch.sigmoid(-logsnr))      # σ = ( 1  / (1+SNR))^½
    return logsnr, alpha, sigma


class DiffusionModel(nn.Module):
    """Conditional diffusion model with a FLEX backbone for interpolation.

    The network is conceptually split into three parts:

    1. Task encoder: 
        - extract features from the start and end snapshots
        - estimate the intermittent snapshot using linear interpolation
        - provides skip connections (refinement) to the main encoder.
    2. Main encoder: processes the noisy residual together with the
       conditioning features and the diffusion timestep t.
    3. Decoder: merges both streams and predicts either x, ε, or v
       depending on *prediction_type*.

    Notes
    -----
    * The "velocity" parameterisation (v) generally yields better‑behaved gradients, but
      we preserve backwards‑compatibility with the other two for checkpoint reuse.
    """
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            task_encoder: nn.Module,
            diff_steps: int,
            prediction_type: str,
            criterion: nn.Module | None = None,
            logsnr_shift: float = 1.0,
        ) -> None:
        super().__init__()
        assert prediction_type in {"v", "eps", "x"}, (
            "Prediction_type must be one of 'v', 'eps', 'x'"
        )
        self.prediction_type = prediction_type

        # Sub‑modules 
        self.encoder = encoder # u-net encoder
        self.decoder = decoder # u-net decoder
        self.task_encoder = task_encoder # task encoder

        # total number of diffusion steps during *sampling*
        self.diff_steps = diff_steps  

        # loss function
        self.criterion = criterion or nn.L2Loss(reduction="none")

         # shift passed to get_logsnr_alpha_sigma
        self.logsnr_shift = logsnr_shift 

    def forward(
            self,
            target_snapshot: torch.Tensor, 
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,  
            fluid_condition: torch.Tensor,
            target_interp_step = torch.Tensor,
            total_interp_steps = torch.Tensor
        ) -> torch.Tensor:
        """
        Compute per‑pixel loss for a *single random* diffusion timestep.

        The routine implements the standard diffusion training recipe:

        1. Sample a random timestep t.
        2. Diffuse the target residual x₀ → xₜ using Gaussian noise ε.
        3. Predict a target (x₀/ε/v) with the network.
        4. Return the per‑pixel loss so callers can decide on the reduction.

        The target and condition snapshots shape: (B,C,H,W)
        """

        # get the target interpolation step
        target_interp_step = target_interp_step.float()
        target_interp_step_scalar = target_interp_step.view(-1)[0]

        # get the total number of interpolation steps
        total_interp_steps = total_interp_steps.float()
        total_interp_steps_scalar = total_interp_steps.view(-1)[0] + 1.0

        # estimate the intermittent conditioning frame
        delta = target_interp_step_scalar / total_interp_steps_scalar
        est_snapshot = (1 - delta) * cond_snapshot_start + delta * cond_snapshot_end

        # the refinement to be learned 
        refinement = target_snapshot - est_snapshot  # (B,C,H,W)

        # 1. random timestep t ∼ 𝕌(0,1) and corresponding schedule coefficients 
        t = torch.rand(refinement.shape[0], device = refinement.device)
        logsnr, alpha, sigma = get_logsnr_alpha_sigma(t, shift = self.logsnr_shift)

        # 2. forward diffusion (add Gaussian noise)
        eps = torch.randn_like(refinement, device=refinement.device) # ε ∼ 𝒩(0, I)
        residual_t = alpha * refinement + sigma * eps # xₜ

        # 3. forward pass
        # 3.1 conditioning start and end snapshots (provides skip connections)
        # (B,C,H,W)
        cond_input_snapshots = torch.cat(
            (cond_snapshot_start, cond_snapshot_end), 
            dim = 1
        )
        
        head_sr, skips_sr = self.task_encoder(
            cond_input_snapshots,
            fluid_condition = fluid_condition
        )

        # 3.2 diffusion path (main U‑Net)
        h, skips = self.encoder(
            residual_t, 
            t, 
            fluid_condition = fluid_condition, 
            cond_skips = skips_sr,
            target_interp_step = target_interp_step,
            total_interp_steps = total_interp_steps
        )

        # 3.3 Decoder merges streams + timestep embedding
        pred = self.decoder(
            h, 
            skips, 
            head_sr, 
            skips_sr, 
            t, 
            fluid_condition = fluid_condition,
            target_interp_step = target_interp_step,
            total_interp_steps = total_interp_steps
        ) 

        # 4. Convert *pred* to the correct target space
        if self.prediction_type == "x":
            # x₀ (direct regression)
            target = refinement                           

        elif self.prediction_type == "eps":
            # Network is trained as *v* but we supervise with ε.
            pred = alpha * pred + sigma * residual_t # ε̂ (predicted)
            target = eps                             # ε  (ground‑truth)

        elif self.prediction_type == "v":
            # Velocity parameterisation: v = α ε − σ x₀
            target = alpha * eps - sigma * refinement

        # The criterion is reduction="none" by default, so the caller retains control
        # (e.g. they can add importance weighting or reduce to *mean* later).
        return self.criterion(pred, target)
    
    @torch.no_grad()
    def sample(
            self,
            n_sample: int,
            size: Tuple[int, int, int],  # (C, H, W)
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
            device: str = "cuda",
            snapshots_i: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """Iterative reverse diffusion from t = 1 → 0 ala DDPM/DDIM.

        Args:
            n_sample: Number of samples to generate.
            size: Output spatial size as (C, H, W).
            cond_snapshot_start: the first conditioning frames.
            cond_snapshot_end: the last conditioning frames.
            fluid_condition: Auxiliary physical fields (e.g. velocity, vorticity).
            device: Device to run on (default: "cuda").
            snapshots_i: Optional initial noise tensor; if None a fresh x_T is drawn.

        Returns:
            A tensor of shape (n_sample, C, H, W) containing the generated snapshots.
        """

        # 0. Initialize with Gaussian noise (or user‑supplied *x_T*)
        if snapshots_i is None:
            snapshots_i = torch.randn(n_sample, *size, device=device)

        # TODO: check the device
        cond_snapshot_start = cond_snapshot_start.to(device)
        cond_snapshot_end = cond_snapshot_end.to(device)

        # concatenate the conditioning snapshots with shape of (B,C,H,W)
        cond_input_snapshots = torch.cat(
            (cond_snapshot_start, cond_snapshot_end), 
            dim = 1
        ) 

        # alias for brevity
        model_head = self.task_encoder  

        # 1. Reverse diffusion loop t = 1 → 0
        for time_step in range(self.diff_steps, 0, -1):
            # Current and previous (t‑1) timesteps normalised to [0,1]
            t = torch.full(
                (n_sample,),  
                time_step / self.diff_steps,  
                device = device
            )
            
            t_ = torch.full(
                (n_sample,), 
                (time_step - 1) / self.diff_steps, 
                device = device
            )

            _, alpha, sigma = get_logsnr_alpha_sigma(
                t,  
                shift = self.logsnr_shift
            )
            
            _, alpha_, sigma_ = get_logsnr_alpha_sigma(
                t_, 
                shift = self.logsnr_shift
            )

            # 1.1 task encoder
            pred_head, skip_head = model_head(
                cond_input_snapshots, 
                fluid_condition = fluid_condition
            )

            # 1.2 Main U‑Net forward ----
            h, skip = self.encoder(
                snapshots_i, 
                t, 
                fluid_condition = fluid_condition, 
                cond_skips = skip_head,
                target_interp_step = target_interp_step,
                total_interp_steps = total_interp_steps
            )

            pred = self.decoder(
                h, 
                skip, 
                pred_head, 
                skip_head, 
                t, 
                fluid_condition = fluid_condition,
                target_interp_step = target_interp_step,
                total_interp_steps = total_interp_steps
            )

            # 1.3 Convert network output to (mean, eps) pair
            if self.prediction_type == "v":
                mean = alpha * snapshots_i - sigma * pred
                eps  = alpha * pred        + sigma * snapshots_i

            elif self.prediction_type == "x":
                mean = pred  # x₀ (direct prediction)
                eps  = (alpha * pred - snapshots_i) / sigma

            elif self.prediction_type == "eps":
                mean = alpha * snapshots_i - sigma * pred  # identical to 'v'
                eps  = alpha * pred        + sigma * snapshots_i

            # 1.4 DDIM update (deterministic if η = 0)
            eta = 0.0  # 0 → DDIM   |   1 → DDPM (full noise)
            noise = torch.randn_like(snapshots_i, device=device) if eta > 0 else 0.0
            snapshots_i = alpha_ * mean + sigma_ * eps + eta * sigma_ * noise

        # 2. Final prediction uses *mean* (deterministic)
        snapshots_i = mean  # last mean corresponds to t=0

        # 3. Correct the final prediction with the refinement
        # Estimate the intermittent snapshot using linear interpolation
        target_interp_step = target_interp_step.float()
        target_interp_step_scalar = target_interp_step.view(-1)[0]
        total_interp_steps = total_interp_steps.float()
        total_interp_steps_scalar = total_interp_steps.view(-1)[0] + 1.0

        # estimate the intermittent conditioning frame
        delta = target_interp_step_scalar / total_interp_steps_scalar
        est_snapshot = (1 - delta) * cond_snapshot_start + delta * cond_snapshot_end

        # the final prediction is the learned refinement    
        snapshots_i += est_snapshot

        return snapshots_i