from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op, _validate_scale_range

class curriculum_randomize_rigid_body_material(ManagerTermBase):
    """Curriculum version of randomize_rigid_body_material.

    - Stage 1 (no randoms): Uses MEAN of the provided ranges (nominal behavior)
    - Stage 2 and later: Full randomization (precomputed buckets)
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # === Exact same parsing as the original Isaac Lab class ===
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # Shape parsing (identical to original)
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term failed to parse the number of shapes per body."
                )
        else:
            self.num_shapes_per_body = None

        # === Read curriculum threshold from cfg.params (no env check in __init__) ===
        self.stage_start = 1 * cfg.params.get("n_iter_per_stage", 1500)   # you will pass this

        # === Precompute buckets exactly like original Isaac Lab ===
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 250))

        self.static_friction_range = static_friction_range
        self.dynamic_friction_range = dynamic_friction_range
        self.restitution_range = restitution_range
        self.num_buckets = num_buckets
        self.make_consistent = cfg.params.get("make_consistent", False)

        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu"
        )

        if self.make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

        # Nominal material for Stage 1
        mean_static = (static_friction_range[0] + static_friction_range[1]) / 2.0
        mean_dynamic = (dynamic_friction_range[0] + dynamic_friction_range[1]) / 2.0
        mean_restitution = (restitution_range[0] + restitution_range[1]) / 2.0
        self.nominal_material = torch.tensor(
            [[mean_static, mean_dynamic, mean_restitution]], device="cpu", dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        n_iter_per_stage: int = 1500,
        make_consistent: bool = False,
    ):
        """Called on every reset (including the very first one)."""
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        current_iter = env.learning_iter

        if current_iter < self.stage_start:
            self._apply_materials(nominal=True, env_ids=env_ids)
        else:
            self._apply_materials(nominal=False, env_ids=env_ids)

    def _apply_materials(self, nominal: bool = False, env_ids: torch.Tensor | None = None):
        """Apply either nominal or randomized materials."""
        
        materials = self.asset.root_physx_view.get_material_properties()

        if nominal:
            # Stage 1: mean values
            if self.num_shapes_per_body is not None:
                for body_id in self.asset_cfg.body_ids:
                    start_idx = sum(self.num_shapes_per_body[:body_id])
                    end_idx = start_idx + self.num_shapes_per_body[body_id]
                    materials[env_ids, start_idx:end_idx] = self.nominal_material
            else:
                materials[env_ids] = self.nominal_material
        else:
            # Stage 2+: randomized from precomputed buckets
            total_num_shapes = self.asset.root_physx_view.max_shapes
            bucket_ids = torch.randint(0, self.num_buckets, (len(env_ids), total_num_shapes), device="cpu")
            material_samples = self.material_buckets[bucket_ids]

            if self.num_shapes_per_body is not None:
                for body_id in self.asset_cfg.body_ids:
                    start_idx = sum(self.num_shapes_per_body[:body_id])
                    end_idx = start_idx + self.num_shapes_per_body[body_id]
                    materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
            else:
                materials[env_ids] = material_samples[:]

        self.asset.root_physx_view.set_material_properties(materials, env_ids)

class curriculum_randomize_rigid_body_mass(ManagerTermBase):
    """Curriculum version of randomize_rigid_body_mass.

    - Stage 1 (no randoms): Uses default mass (nominal behavior)
    - Stage 2 and later: Full randomization (exactly like original)
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # === Same parsing as original Isaac Lab class ===
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # === Curriculum threshold from cfg.params (no env check in __init__) ===
        self.stage_start = 1 * cfg.params.get("n_iter_per_stage", 1500.0)

        # Store all randomization parameters
        self.mass_distribution_params = cfg.params["mass_distribution_params"]
        self.operation = cfg.params.get("operation", "scale")
        self.distribution = cfg.params.get("distribution", "uniform")
        self.recompute_inertia = cfg.params.get("recompute_inertia", True)

        # Optional validation (same as original)
        if self.operation == "scale":
            if "mass_distribution_params" in cfg.params:
                _validate_scale_range(
                    cfg.params["mass_distribution_params"], "mass_distribution_params", allow_zero=False
                )
        elif self.operation not in ("abs", "add"):
            raise ValueError(
                f"Randomization term 'curriculum_randomize_rigid_body_mass' does not support operation: '{self.operation}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"],
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        n_iter_per_stage: int = 1500,
        recompute_inertia: bool = True,
    ):
        """Called on every reset (including the very first one)."""
        current_iter = env.learning_iter

        if current_iter < self.stage_start:
            """Stage 1: Nominal behavior → reset to default mass (no randomization)."""
            if env_ids is None:
                env_ids = torch.arange(env.scene.num_envs, device="cpu")
            else:
                env_ids = env_ids.cpu()

            # resolve body indices
            if self.asset_cfg.body_ids == slice(None):
                body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
            else:
                body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

            masses = self.asset.root_physx_view.get_masses() #!
            # reset to default (nominal)
            masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

            self.asset.root_physx_view.set_masses(masses, env_ids)

            # Note: we skip inertia recompute in nominal case (no change)
        else:
            """Stage 2: Linear ramp from nominal → full range over the stage."""
            if env_ids is None:
                env_ids = torch.arange(env.scene.num_envs, device="cpu")
            else:
                env_ids = env_ids.cpu()

            # Compute progress within Stage 2 (0.0 → 1.0)
            progress = min(1.0, max(0.0, (env.learning_iter - self.stage_start) / float(n_iter_per_stage)))

            # Interpolate range (for "scale" the nominal is always (1.0, 1.0))
            if self.operation == "scale":
                nominal_low, nominal_high = 1.0, 1.0
            else:
                nominal_low, nominal_high = 0.0, 0.0  # fallback for add/abs

            target_low, target_high = self.mass_distribution_params

            current_low = nominal_low + progress * (target_low - nominal_low)
            current_high = nominal_high + progress * (target_high - nominal_high)
            current_range = (current_low, current_high)

            # resolve body indices
            if self.asset_cfg.body_ids == slice(None):
                body_ids = torch.arange(self.asset.num_bodies, dtype=torch.int, device="cpu")
            else:
                body_ids = torch.tensor(self.asset_cfg.body_ids, dtype=torch.int, device="cpu")

            masses = self.asset.root_physx_view.get_masses()
            masses[env_ids[:, None], body_ids] = self.asset.data.default_mass[env_ids[:, None], body_ids].clone()

            # print(self.asset.body_names)
            # print("default ", self.asset.data.default_mass)
            # Apply randomization with interpolated range
            masses = _randomize_prop_by_op(
                masses,
                current_range,
                env_ids,
                body_ids,
                operation=self.operation,
                distribution=self.distribution,
            )
            # print("new masses", masses)
            # print()

            self.asset.root_physx_view.set_masses(masses, env_ids)

            # recompute inertia if needed
            if self.recompute_inertia:
                ratios = masses[env_ids[:, None], body_ids] / self.asset.data.default_mass[env_ids[:, None], body_ids]
                inertias = self.asset.root_physx_view.get_inertias()

                if isinstance(self.asset, Articulation):
                    inertias[env_ids[:, None], body_ids] = (
                        self.asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
                    )
                else:
                    inertias[env_ids] = self.asset.data.default_inertia[env_ids] * ratios

                self.asset.root_physx_view.set_inertias(inertias, env_ids)

class curriculum_randomize_joint_velocity_limit(ManagerTermBase):
    """Curriculum version of joint velocity limit randomization.
    - Stage 1 (first n_iter_per_stage iterations): Uses default/initial velocity limits (nominal behavior)
    - Stage 2 and later: Linear ramp from nominal (scale=1.0) → full randomization range

    NEW LOGIC:
    - First apply a homogeneous scale (same multiplier for ALL joints of one hand/env)
      sampled from homogeneous_distribution_parameters
    - Then apply per-joint independent scale on top of the homogeneous one
      (from the original velocity_limit_distribution_params)

    IMPORTANT: initial limits are read ONCE in __init__ and NEVER read again from sim.
    This eliminates the catastrophic fallback bug you had.
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Initial values (read once)
        self.initial_joint_vel_limits = self.asset.data.joint_vel_limits.clone()
        print(f"[curriculum_randomize_joint_velocity_limit __init__] Using asset.data.joint_vel_limits.clone() as initial (shape: {self.initial_joint_vel_limits.shape})")

        # Curriculum settings
        self.stage_start = cfg.params.get("n_iter_per_stage", 1500)
        self.velocity_limit_distribution_params = cfg.params["velocity_limit_distribution_params"]
        self.homogeneous_distribution_params = cfg.params["homogeneous_distribution_parameters"]

        # Validation
        for name, params in [("velocity_limit_distribution_params", self.velocity_limit_distribution_params),
                             ("homogeneous_distribution_parameters", self.homogeneous_distribution_params)]:
            low, high = params
            if not (low <= high):
                raise ValueError(f"{name} {params} must satisfy low <= high.")

        print(f"[DEBUG curriculum_randomize_joint_velocity_limit __init__] Initialized with stage_start={self.stage_start}, "
              f"per_joint={self.velocity_limit_distribution_params}, homogeneous={self.homogeneous_distribution_params}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        velocity_limit_distribution_params: tuple[float, float],
        homogeneous_distribution_parameters: tuple[float, float],
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        n_iter_per_stage: int = 1500,
    ):
        """Called every reset by the EventManager."""
        current_iter = env.learning_iter

        # Resolve env_ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(self.asset.device)

        # Resolve joint indices (slice or specific)
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # Stage 1: Nominal (default limits) - no randomization
        if current_iter < self.stage_start:
            nominal_limits = self.initial_joint_vel_limits[env_ids]
            if joint_ids != slice(None):
                nominal_limits = nominal_limits[:, joint_ids]
            self.asset.write_joint_velocity_limit_to_sim(nominal_limits, joint_ids=joint_ids, env_ids=env_ids)
            return

        # Stage 2: Linear ramp from 1.0 → full randomization range
        progress = min(1.0, max(0.0, (current_iter - self.stage_start) / float(n_iter_per_stage)))

        # === 1. HOMOGENEOUS SCALE (same for all joints of one hand/env) ===
        low_h, high_h = self.homogeneous_distribution_params
        current_low_h = 1.0 + progress * (low_h - 1.0)
        current_high_h = 1.0 + progress * (high_h - 1.0)

        homogeneous_scales = torch.rand(len(env_ids), device=self.asset.device) * (current_high_h - current_low_h) + current_low_h
        homogeneous_scales = homogeneous_scales.unsqueeze(1)   # [num_envs, 1] → broadcasts to all joints

        # === 2. PER-JOINT SCALE (on top of the homogeneous scale) ===
        low, high = self.velocity_limit_distribution_params
        current_low = 1.0 + progress * (low - 1.0)
        current_high = 1.0 + progress * (high - 1.0)

        num_rand_joints = self.asset.num_joints if joint_ids == slice(None) else len(joint_ids)
        per_joint_scales = torch.rand((len(env_ids), num_rand_joints), device=self.asset.device) * (current_high - current_low) + current_low

        # === 3. Combine: homogeneous first, then per-joint on top ===
        initial_limits = self.initial_joint_vel_limits[env_ids]
        if joint_ids != slice(None):
            initial_limits = initial_limits[:, joint_ids]

        vel_limits = initial_limits * homogeneous_scales * per_joint_scales

        # Write to sim
        self.asset.write_joint_velocity_limit_to_sim(vel_limits, joint_ids=joint_ids, env_ids=env_ids)
        # print(vel_limits)

class curriculum_randomize_joint_effort_limit(ManagerTermBase):
    """Curriculum version of joint effort limit randomization.
    Same logic as velocity version but for effort limits.

    NEW LOGIC (identical to curriculum_randomize_joint_velocity_limit):
    - First apply a homogeneous scale (same multiplier for ALL joints of one hand/env)
      sampled from homogeneous_distribution_parameters
    - Then apply per-joint independent scale on top of the homogeneous one
      (from the original effort_limit_distribution_params)

    Initial limits stored once in __init__ - no more dangerous fallbacks.
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # === Store INITIAL effort limits ONCE ===
        self.initial_joint_effort_limits = self.asset.data.joint_effort_limits.clone()
        print(f"[DEBUG curriculum_randomize_joint_effort_limit __init__] Using asset.data.joint_effort_limits.clone() as initial (shape: {self.initial_joint_effort_limits.shape})")

        # Curriculum settings
        self.stage_start = cfg.params.get("n_iter_per_stage", 1500)
        self.effort_limit_distribution_params = cfg.params["effort_limit_distribution_params"]
        self.homogeneous_distribution_params = cfg.params["homogeneous_distribution_parameters"]

        # Validation
        for name, params in [("effort_limit_distribution_params", self.effort_limit_distribution_params),
                             ("homogeneous_distribution_parameters", self.homogeneous_distribution_params)]:
            low, high = params
            if not (low <= high):
                raise ValueError(f"{name} {params} must satisfy low <= high.")

        print(f"[DEBUG curriculum_randomize_joint_effort_limit __init__] Initialized with stage_start={self.stage_start}, "
              f"per_joint={self.effort_limit_distribution_params}, homogeneous={self.homogeneous_distribution_params}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        effort_limit_distribution_params: tuple[float, float],
        homogeneous_distribution_parameters: tuple[float, float],
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        n_iter_per_stage: int = 1500,
    ):
        """Called every reset by the EventManager."""
        current_iter = env.learning_iter

        # Resolve env_ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(self.asset.device)

        # Resolve joint indices (slice or specific)
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # Stage 1: Nominal (default limits) - no randomization
        if current_iter < self.stage_start:
            nominal_limits = self.initial_joint_effort_limits[env_ids]
            if joint_ids != slice(None):
                nominal_limits = nominal_limits[:, joint_ids]
            self.asset.write_joint_effort_limit_to_sim(nominal_limits, joint_ids=joint_ids, env_ids=env_ids)
            return

        # Stage 2: Linear ramp from 1.0 → full randomization range
        progress = min(1.0, max(0.0, (current_iter - self.stage_start) / float(n_iter_per_stage)))

        # === 1. HOMOGENEOUS SCALE (same for all joints of one hand/env) ===
        low_h, high_h = self.homogeneous_distribution_params
        current_low_h = 1.0 + progress * (low_h - 1.0)
        current_high_h = 1.0 + progress * (high_h - 1.0)

        homogeneous_scales = torch.rand(len(env_ids), device=self.asset.device) * (current_high_h - current_low_h) + current_low_h
        homogeneous_scales = homogeneous_scales.unsqueeze(1)   # [num_envs, 1] → broadcasts to all joints

        # === 2. PER-JOINT SCALE (on top of the homogeneous scale) ===
        low, high = self.effort_limit_distribution_params
        current_low = 1.0 + progress * (low - 1.0)
        current_high = 1.0 + progress * (high - 1.0)

        num_rand_joints = self.asset.num_joints if joint_ids == slice(None) else len(joint_ids)
        per_joint_scales = torch.rand((len(env_ids), num_rand_joints), device=self.asset.device) * (current_high - current_low) + current_low

        # === 3. Combine: homogeneous first, then per-joint on top ===
        initial_limits = self.initial_joint_effort_limits[env_ids]
        if joint_ids != slice(None):
            initial_limits = initial_limits[:, joint_ids]

        effort_limits = initial_limits * homogeneous_scales * per_joint_scales

        # Write to sim
        self.asset.write_joint_effort_limit_to_sim(effort_limits, joint_ids=joint_ids, env_ids=env_ids)

class curriculum_randomize_joint_parameters(ManagerTermBase):
    """Curriculum version of randomize_joint_parameters with gradual ramp.

    - Stage 1 (no randoms): Uses default joint parameters (nominal)
    - Stage 2: Linear ramp from nominal → full configured range over the stage
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # === Same parsing as original ===
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # === Curriculum parameters from cfg ===
        self.stage_start = 1 * cfg.params.get("n_iter_per_stage", 1500)
        self.n_iter_per_stage = cfg.params.get("n_iter_per_stage", 1500)

        # Store target parameters
        self.friction_distribution_params = cfg.params.get("friction_distribution_params")
        self.armature_distribution_params = cfg.params.get("armature_distribution_params")
        self.lower_limit_distribution_params = cfg.params.get("lower_limit_distribution_params")
        self.upper_limit_distribution_params = cfg.params.get("upper_limit_distribution_params")
        self.operation = cfg.params.get("operation", "abs")
        self.distribution = cfg.params.get("distribution", "uniform")

        # Validation (same as original)
        if self.operation == "scale":
            if self.friction_distribution_params is not None:
                _validate_scale_range(self.friction_distribution_params, "friction_distribution_params")
            if self.armature_distribution_params is not None:
                _validate_scale_range(self.armature_distribution_params, "armature_distribution_params")
        elif self.operation not in ("abs", "add"):
            raise ValueError(
                f"Randomization term 'curriculum_randomize_joint_parameters' does not support operation: '{self.operation}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        n_iter_per_stage: int = 1500,
    ):
        current_iter = env.learning_iter

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(self.asset.device)

        # resolve joint indices 
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)   # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # Stage 1: nominal (default values)
        if current_iter < self.stage_start:
            # joint friction coefficient
            if friction_distribution_params is not None:
                friction_coeff = self.asset.data.default_joint_friction_coeff.clone()
                self.asset.write_joint_friction_coefficient_to_sim(
                    friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
                )
            # joint armature
            if armature_distribution_params is not None:
                armature = self.asset.data.default_joint_armature.clone()
                self.asset.write_joint_armature_to_sim(
                    armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
                )
            # joint position limits
            if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
                joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()
                self.asset.write_joint_position_limit_to_sim(
                    joint_pos_limits[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
                )
            return

        # Stage 2: gradual ramp
        progress = min(1.0, max(0.0, (current_iter - self.stage_start) / float(self.n_iter_per_stage)))

        # joint friction coefficient
        if friction_distribution_params is not None:
            low, high = friction_distribution_params
            current_low = 1.0 + progress * (low - 1.0) if operation == "scale" else low + progress * (low - low)
            current_high = 1.0 + progress * (high - 1.0) if operation == "scale" else high + progress * (high - high)
            current_range = (current_low, current_high)

            friction_coeff = _randomize_prop_by_op(
                self.asset.data.default_joint_friction_coeff.clone(),
                current_range,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_friction_coefficient_to_sim(
                friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint armature
        if armature_distribution_params is not None:
            low, high = armature_distribution_params
            current_low = 1.0 + progress * (low - 1.0) if operation == "scale" else low + progress * (low - low)
            current_high = 1.0 + progress * (high - 1.0) if operation == "scale" else high + progress * (high - high)
            current_range = (current_low, current_high)

            armature = _randomize_prop_by_op(
                self.asset.data.default_joint_armature.clone(),
                current_range,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim(
                armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = self.asset.data.default_joint_pos_limits.clone()

            if lower_limit_distribution_params is not None:
                low, high = lower_limit_distribution_params
                current_low = 0.0 + progress * (low - 0.0)
                current_high = 0.0 + progress * (high - 0.0)
                current_range = (current_low, current_high)
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    current_range,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            if upper_limit_distribution_params is not None:
                low, high = upper_limit_distribution_params
                current_low = 0.0 + progress * (low - 0.0)
                current_high = 0.0 + progress * (high - 0.0)
                current_range = (current_low, current_high)
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    current_range,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'curriculum_randomize_joint_parameters' is setting lower joint limits that are greater than upper joint limits."
                )

            self.asset.write_joint_position_limit_to_sim(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )

class curriculum_randomize_actuator_gains(ManagerTermBase):
    """Curriculum version of randomize_actuator_gains with gradual ramp.

    NEW LOGIC (identical to velocity/effort versions):
    - First apply a homogeneous scale (same multiplier for ALL joints of one hand/env)
      sampled from homogeneous_stiffness_distribution_params / homogeneous_damping_distribution_params
      (ALWAYS treated as a strict scale operation)
    - Then apply the original per-joint/actuator randomization on top using _randomize_prop_by_op

    Stage 1: Uses default actuator gains (nominal)
    Stage 2: Linear ramp from nominal → full configured range
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        # === Curriculum parameters from cfg ===
        self.stage_2_start = 1 * cfg.params.get("n_iter_per_stage", 1500)
        self.n_iter_per_stage = cfg.params.get("n_iter_per_stage", 1500)

        # Store target parameters
        self.stiffness_distribution_params = cfg.params.get("stiffness_distribution_params")
        self.damping_distribution_params = cfg.params.get("damping_distribution_params")

        # === NEW: Homogeneous parameters (one scale per environment) ===
        self.homogeneous_stiffness_distribution_params = cfg.params.get("homogeneous_stiffness_distribution_params")
        self.homogeneous_damping_distribution_params = cfg.params.get("homogeneous_damping_distribution_params")

        self.operation = cfg.params.get("operation", "abs")
        self.distribution = cfg.params.get("distribution", "uniform")

        # Validation (original + new homogeneous)
        if self.operation == "scale":
            if self.stiffness_distribution_params is not None:
                _validate_scale_range(
                    self.stiffness_distribution_params, "stiffness_distribution_params", allow_zero=False
                )
            if self.damping_distribution_params is not None:
                _validate_scale_range(self.damping_distribution_params, "damping_distribution_params")
        elif self.operation not in ("abs", "add"):
            raise ValueError(
                f"Randomization term 'curriculum_randomize_actuator_gains' does not support operation: '{self.operation}'."
            )

        # Homogeneous params are always treated as scale → simple low/high check
        for name, params in [("homogeneous_stiffness_distribution_params", self.homogeneous_stiffness_distribution_params),
                             ("homogeneous_damping_distribution_params", self.homogeneous_damping_distribution_params)]:
            if params is not None:
                low, high = params
                if not (low <= high):
                    raise ValueError(f"{name} {params} must satisfy low <= high.")

        print(f"[DEBUG curriculum_randomize_actuator_gains __init__] Initialized with stage_2_start={self.stage_2_start}, "
              f"stiffness={self.stiffness_distribution_params}, damping={self.damping_distribution_params}, "
              f"homog_stiff={self.homogeneous_stiffness_distribution_params}, homog_damp={self.homogeneous_damping_distribution_params}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        stiffness_distribution_params: tuple[float, float] | None = None,
        damping_distribution_params: tuple[float, float] | None = None,
        homogeneous_stiffness_distribution_params: tuple[float, float] | None = None,
        homogeneous_damping_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        n_iter_per_stage: int = 1500,
    ):
        """All logic is inside __call__ (including the actuator loop)."""
        current_iter = env.learning_iter

        # Stage 1: nominal (default gains)
        if current_iter < self.stage_2_start:
            return

        # Stage 2: gradual ramp
        progress = min(1.0, max(0.0, (current_iter - self.stage_2_start) / float(self.n_iter_per_stage)))

        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)
        else:
            env_ids = env_ids.to(self.asset.device)

        # === Compute homogeneous scales per environment (if configured) ===
        # These are sampled once and reused for all actuators of the same hand
        homog_stiff_scale = None
        if self.homogeneous_stiffness_distribution_params is not None:
            low, high = self.homogeneous_stiffness_distribution_params
            curr_low = 1.0 + progress * (low - 1.0)
            curr_high = 1.0 + progress * (high - 1.0)
            homog_stiff_scale = torch.rand(len(env_ids), device=self.asset.device) * (curr_high - curr_low) + curr_low
            homog_stiff_scale = homog_stiff_scale.unsqueeze(1)  # [num_envs, 1] → broadcasts to joints

        homog_damp_scale = None
        if self.homogeneous_damping_distribution_params is not None:
            low, high = self.homogeneous_damping_distribution_params
            curr_low = 1.0 + progress * (low - 1.0)
            curr_high = 1.0 + progress * (high - 1.0)
            homog_damp_scale = torch.rand(len(env_ids), device=self.asset.device) * (curr_high - curr_low) + curr_low
            homog_damp_scale = homog_damp_scale.unsqueeze(1)  # [num_envs, 1]

        # Loop through actuators (exactly as original)
        for actuator in self.asset.actuators.values():
            if isinstance(self.asset_cfg.joint_ids, slice):
                actuator_indices = slice(None)
                if isinstance(actuator.joint_indices, slice):
                    global_indices = slice(None)
                else:
                    global_indices = torch.tensor(actuator.joint_indices, device=self.asset.device)
            elif isinstance(actuator.joint_indices, slice):
                global_indices = actuator_indices = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
            else:
                actuator_joint_indices = torch.tensor(actuator.joint_indices, device=self.asset.device)
                asset_joint_ids = torch.tensor(self.asset_cfg.joint_ids, device=self.asset.device)
                actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
                if len(actuator_indices) == 0:
                    continue
                global_indices = actuator_joint_indices[actuator_indices]

            # === Randomize stiffness ===
            if stiffness_distribution_params is not None:
                low, high = stiffness_distribution_params
                current_low = 1.0 + progress * (low - 1.0) if operation == "scale" else low
                current_high = 1.0 + progress * (high - 1.0) if operation == "scale" else high
                current_range = (current_low, current_high)

                stiffness = actuator.stiffness[env_ids].clone()
                # Reset slice to default (exactly as original)
                stiffness[:, actuator_indices] = self.asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()

                # 1. Apply homogeneous scale first (STRICT scale operation)
                if homog_stiff_scale is not None:
                    stiffness[:, actuator_indices] = stiffness[:, actuator_indices] * homog_stiff_scale

                # 2. Then apply the original per-joint randomization on top
                stiffness[:, actuator_indices] = _randomize_prop_by_op(
                    stiffness[:, actuator_indices],
                    current_range,
                    dim_0_ids=None,
                    dim_1_ids=actuator_indices,
                    operation=operation,
                    distribution=distribution,
                )

                actuator.stiffness[env_ids] = stiffness
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_stiffness_to_sim(
                        stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )

            # === Randomize damping ===
            if damping_distribution_params is not None:
                low, high = damping_distribution_params
                current_low = 1.0 + progress * (low - 1.0) if operation == "scale" else low
                current_high = 1.0 + progress * (high - 1.0) if operation == "scale" else high
                current_range = (current_low, current_high)

                damping = actuator.damping[env_ids].clone()
                # Reset slice to default (exactly as original)
                damping[:, actuator_indices] = self.asset.data.default_joint_damping[env_ids][:, global_indices].clone()

                # 1. Apply homogeneous scale first (STRICT scale operation)
                if homog_damp_scale is not None:
                    damping[:, actuator_indices] = damping[:, actuator_indices] * homog_damp_scale

                # 2. Then apply the original per-joint randomization on top
                damping[:, actuator_indices] = _randomize_prop_by_op(
                    damping[:, actuator_indices],
                    current_range,
                    dim_0_ids=None,
                    dim_1_ids=actuator_indices,
                    operation=operation,
                    distribution=distribution,
                )

                actuator.damping[env_ids] = damping
                if isinstance(actuator, ImplicitActuator):
                    self.asset.write_joint_damping_to_sim(
                        damping, joint_ids=actuator.joint_indices, env_ids=env_ids
                    )

            # print(stiffness)
            # print(damping)
