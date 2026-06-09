from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ESMFold2StageConfig:
    stage: int
    steps: int
    max_tokens: int
    max_atoms: int
    diffusion_batch_size: int
    full_training: bool
    bond_loss: bool
    smooth_lddt: bool
    distogram: bool
    confidence: bool
    fixed_lr: float | None = None


@dataclass(frozen=True)
class ESMFold2FrozenFeatureConfig:
    sigma_data: float = 16.0
    train_noise_log_mean: float = -1.2
    train_noise_log_std: float = 1.5
    alpha_struct: float = 4.0
    alpha_dist: float = 3e-2
    alpha_conf: float = 1e-4
    alpha_bond: float = 1.0
    alpha_slddt: float = 1.0
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.95
    optimizer_eps: float = 1e-8
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_lr: float = 1.8e-3
    step_decay_factor: float = 0.95
    step_decay_steps: int = 50_000
    confidence_resolution_cutoff: float = 4.0
    confidence_mini_rollout_steps: int = 20
    confidence_pair_drop_prob: float = 0.2
    recurrent_poisson_mean: float = 3.0
    recurrent_min_loops: int = 1
    recurrent_max_loops: int = 6
    recurrent_grad_loops: int = 2
    distogram_min_bin: float = 2.0
    distogram_max_bin: float = 22.0
    distogram_bins: int = 64
    plddt_bins: int = 50
    pde_bins: int = 64
    pae_bins: int = 64
    pde_max_bin: float = 32.0
    pae_max_bin: float = 32.0


STAGE_CONFIGS: dict[int, ESMFold2StageConfig] = {
    1: ESMFold2StageConfig(
        stage=1,
        steps=50_000,
        max_tokens=384,
        max_atoms=9_216,
        diffusion_batch_size=48,
        full_training=True,
        bond_loss=False,
        smooth_lddt=True,
        distogram=True,
        confidence=True,
    ),
    2: ESMFold2StageConfig(
        stage=2,
        steps=5_000,
        max_tokens=640,
        max_atoms=15_360,
        diffusion_batch_size=32,
        full_training=True,
        bond_loss=True,
        smooth_lddt=False,
        distogram=True,
        confidence=True,
    ),
    3: ESMFold2StageConfig(
        stage=3,
        steps=11_000,
        max_tokens=768,
        max_atoms=18_432,
        diffusion_batch_size=32,
        full_training=False,
        bond_loss=True,
        smooth_lddt=False,
        distogram=False,
        confidence=True,
        fixed_lr=1.0e-4,
    ),
}


def get_stage_config(stage: int) -> ESMFold2StageConfig:
    try:
        return STAGE_CONFIGS[int(stage)]
    except KeyError as exc:
        raise ValueError(f"stage must be one of {sorted(STAGE_CONFIGS)}, got {stage}") from exc
