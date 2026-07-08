import torch
import numpy as np
import tf.transformations as tf
from sphere_torch_utils import *

def torch_compute_deformed_sphere_points_unfixed_vectors(
    driving_vectors, vector_offsets, n_pole_values, s_pole_values, points, base_radius=1.0, device='cpu', epsilon=1e-2, p=2
):
    """
    Compute deformed sphere points using inverse distance weighting, with driving vectors given as (x, y, z),
    and including north and south pole offsets. The radius is computed as base_radius * interpolated_offset.

    Parameters:
    - driving_vectors: [n_spheres, n_driving_vectors, 3] tensor of (x, y, z) coordinates (units:m)
    - vector_offsets: [n_spheres, n_driving_vectors] tensor of offsets at driving vector locations (units:base_radius)
    - n_pole_values: [n_spheres] tensor, offset at the north pole for each sphere (units:base_radius)
    - s_pole_values: [n_spheres] tensor, offset at the south pole for each sphere (units:base_radius)
    - points: [n_points, 2] tensor of (theta, phi) query points
    - base_radius: float or tensor, the base radius of the sphere
    - device: torch.device, the device to perform computations on
    - epsilon: float, small value to avoid division by zero
    - p: int, power for inverse distance weighting

    Returns:
    - xyz: [n_points, n_spheres, 3] tensor, the deformed sphere points in Cartesian coordinates
    """
    # Step 1: Move all inputs to the specified device
    driving_vectors = driving_vectors.to(device)
    vector_offsets = vector_offsets.to(device)
    n_pole_values = n_pole_values.to(device)
    s_pole_values = s_pole_values.to(device)
    points = points.to(device)
    if isinstance(base_radius, torch.Tensor):
        base_radius = base_radius.to(device)
    else:
        base_radius = torch.tensor(base_radius, dtype=torch.float, device=device)

    # Extract dimensions
    n_spheres = driving_vectors.size(0)          # Number of spheres

    # Step 2: Normalize driving vectors to unit directions
    # Shape: [n_spheres, n_driving_vectors] -> norms of each vector
    norms = torch.norm(driving_vectors, dim=2, keepdim=True)  # [n_spheres, n_driving_vectors, 1]
    p_driving = driving_vectors / (norms + epsilon)           # [n_spheres, n_driving_vectors, 3]

    # Step 3: Define pole directions (north: [0, 0, 1], south: [0, 0, -1])
    p_north = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n_spheres, 1, 3)  # [n_spheres, 1, 3]
    p_south = torch.tensor([[0.0, 0.0, -1.0]], device=device).expand(n_spheres, 1, 3) # [n_spheres, 1, 3]

    # Step 4: Combine all control directions
    # Concatenate along the second dimension: driving vectors + north + south
    all_directions = torch.cat([p_driving, p_north, p_south], dim=1)  # [n_spheres, n_driving_vectors + 2, 3]

    # Step 5: Combine all offsets
    # Reshape pole offsets to [n_spheres, 1] and concatenate with vector offsets
    offsets_north = n_pole_values.unsqueeze(1)  # [n_spheres, 1]
    offsets_south = s_pole_values.unsqueeze(1)  # [n_spheres, 1]

    all_offsets = torch.cat([vector_offsets, offsets_north, offsets_south], dim=1)  # [n_spheres, n_driving_vectors + 2]

    # Step 6: Convert query points from spherical to Cartesian coordinates
    theta, phi = points[:, 0], points[:, 1]  # [n_points]
    x = torch.sin(phi) * torch.cos(theta)    # [n_points] #! Can be precomputed
    y = torch.sin(phi) * torch.sin(theta)    # [n_points]
    z = torch.cos(phi)                       # [n_points]
    q = torch.stack([x, y, z], dim=1)        # [n_points, 3]

    # Step 7: Compute distances between query points and control directions
    # Expand dimensions for broadcasting: [n_spheres, 1, n_driving_vectors + 2, 3] - [1, n_points, 1, 3]
    distances = torch.norm(all_directions.unsqueeze(1) - q.unsqueeze(0).unsqueeze(2), dim=3)  # [n_spheres, n_points, n_driving_vectors + 2]

    # Step 8: Compute IDW weights
    weights = 1.0 / (distances + epsilon) ** p  # [n_spheres, n_points, n_driving_vectors + 2]

    # Step 9: Interpolate offsets using IDW
    sum_weights = weights.sum(dim=2)                    # [n_spheres, n_points]
    interpolated_offset = (weights * all_offsets.unsqueeze(1)).sum(dim=2) / sum_weights  # [n_spheres, n_points]

    # Step 10: Compute deformed radius
    radius = base_radius * interpolated_offset  # [n_spheres, n_points], broadcasts if base_radius is scalar

    # Step 11: Compute final deformed points
    xyz = radius.transpose(0, 1).unsqueeze(2) * q.unsqueeze(1)  # [n_points, n_spheres, 3]

    return xyz

def torch_compute_deformed_sphere_points_detached_fingers_unfixed_vectors(
    driving_vectors, vector_offsets, points, base_radius=1.0, device='cpu', epsilon=1e-6
):
    """
    Compute deformed sphere points using linear interpolation based on phi angles of anchors.
    
    Parameters:
    - driving_vectors: [n_envs, n_spheres, n_anchors, 3] tensor of (x, y, z) anchor coordinates
    - vector_offsets: [n_envs, n_spheres, n_anchors] tensor of offsets at anchor locations
    - points: [n_points, 2] tensor of (theta, phi) sample points
    - base_radius: float or tensor, base radius of the spheres
    - device: torch.device, device to perform computations on
    - epsilon: float, small value to avoid division by zero
    
    Returns:
    - xyz: [n_envs, n_spheres, n_points, 3] tensor of deformed Cartesian coordinates
    """
    
    # Move inputs to device
    driving_vectors = driving_vectors.to(device)
    vector_offsets = vector_offsets.to(device)
    points = points.to(device)
    if isinstance(base_radius, torch.Tensor):
        base_radius = base_radius.to(device)
    else:
        base_radius = torch.tensor(base_radius, dtype=torch.float, device=device)

    n_envs, n_spheres, n_anchors, _ = driving_vectors.shape
    n_points = points.size(0)

    # Compute phi angles for anchors (polar angle, 0 to π)
    norms = torch.norm(driving_vectors, dim=3, keepdim=True)  # [n_envs, n_spheres, n_anchors, 1]
    normalized_vectors = driving_vectors / (norms + epsilon)  # [n_envs, n_spheres, n_anchors, 3]
    x = normalized_vectors[..., 0]  # [n_envs, n_spheres, n_anchors]
    y = normalized_vectors[..., 1]  # [n_envs, n_spheres, n_anchors]
    z = normalized_vectors[..., 2]  # [n_envs, n_spheres, n_anchors]
    phi_anchors = torch.acos(torch.clamp(z, -1.0, 1.0))  # [n_envs, n_spheres, n_anchors]
    theta_anchors = torch.atan2(y,x) # [n_envs, n_spheres, n_anchors]
    anchor_spherical_coords = torch.cat([theta_anchors.unsqueeze(-1), phi_anchors.unsqueeze(-1)], dim = 3)

    # Sort phi angles and offsets for each environment and sphere
    sorted_phi, sort_idx = torch.sort(phi_anchors, dim=2)  # [n_envs, n_spheres, n_anchors]
    sorted_offsets = torch.gather(vector_offsets, 2, sort_idx)  # [n_envs, n_spheres, n_anchors]

    # Flatten for vectorized interpolation
    sorted_phi_flat = sorted_phi.view(n_envs * n_spheres, n_anchors)  # [n_envs * n_spheres, n_anchors]
    sorted_offsets_flat = sorted_offsets.view(n_envs * n_spheres, n_anchors)  # [n_envs * n_spheres, n_anchors]

    # Extract phi from sample points
    phi_samples = points[:, 1]  # [n_points]

    # Find interpolation indices
    idx = torch.searchsorted(sorted_phi_flat, phi_samples[None, :].repeat((sorted_phi_flat.size(0),1)), right=True)  # [n_envs * n_spheres, n_points]
    idx_a = torch.clamp(idx - 1, 0, n_anchors - 1)  # Lower bound
    idx_b = torch.clamp(idx, 0, n_anchors - 1)  # Upper bound

    # Gather phi and offset values for interpolation
    phi_a = torch.gather(sorted_phi_flat, 1, idx_a)  # [n_envs * n_spheres, n_points]
    phi_b = torch.gather(sorted_phi_flat, 1, idx_b)  # [n_envs * n_spheres, n_points]
    offsets_a = torch.gather(sorted_offsets_flat, 1, idx_a)  # [n_envs * n_spheres, n_points]
    offsets_b = torch.gather(sorted_offsets_flat, 1, idx_b)  # [n_envs * n_spheres, n_points]

    # Compute interpolation weights
    delta_phi = phi_b - phi_a
    diff = phi_samples[None, :] - phi_a  # Shape: [n_envs * n_spheres, n_points]
    weight = torch.where(
        delta_phi > 0,
        diff / (delta_phi + epsilon),
        torch.tensor(0.0, device=delta_phi.device)
    )  # Shape: [n_envs * n_spheres, n_points]
    weight = torch.clamp(weight, 0, 1)

    # Interpolate offsets
    interpolated_offset = offsets_a * (1 - weight) + offsets_b * weight  # [n_envs * n_spheres, n_points]
    interpolated_offset = interpolated_offset.view(n_envs, n_spheres, n_points)  # [n_envs, n_spheres, n_points]

    # Compute deformed radius
    radius = base_radius * interpolated_offset  # [n_envs, n_spheres, n_points]

    # Convert sample points to Cartesian unit vectors
    theta, phi = points[:, 0], points[:, 1]  # [n_points]
    x = torch.sin(phi) * torch.cos(theta)    # [n_points]
    y = torch.sin(phi) * torch.sin(theta)    # [n_points]
    z = torch.cos(phi)                       # [n_points]
    q = torch.stack([x, y, z], dim=1)        # [n_points, 3]

    # Compute final deformed points
    xyz = radius.unsqueeze(-1) * q  # [n_envs, n_spheres, n_points, 3]

    return xyz, anchor_spherical_coords

def torch_compute_deformed_sphere_points_detached_fingers(
    fixed_phi, fixed_offsets, points, base_radius=1.0, device='cpu', epsilon=1e-6
):
    """
    Compute deformed sphere points using linear interpolation based on fixed phi values.
    
    Parameters:
    - fixed_phi: [n_fixed_points] tensor of fixed phi values (same for all envs and spheres)
    - fixed_offsets: [n_envs, n_spheres, n_fixed_points] tensor of offsets at fixed phi locations
    - points: [n_points, 2] tensor of (theta, phi) sample points
    - base_radius: float or tensor, base radius of the spheres
    - device: torch.device, device to perform computations on
    - epsilon: float, small value to avoid division by zero
    
    Returns:
    - xyz: [n_envs, n_spheres, n_points, 3] tensor of deformed Cartesian coordinates
    """
    
    # Move inputs to the specified device
    fixed_phi = fixed_phi.to(device)
    fixed_offsets = fixed_offsets.to(device)
    points = points.to(device)
    if isinstance(base_radius, torch.Tensor):
        base_radius = base_radius.to(device)
    else:
        base_radius = torch.tensor(base_radius, dtype=torch.float, device=device)

    # Extract dimensions
    n_envs, n_spheres, n_fixed_points = fixed_offsets.shape

    # Sort fixed_phi once (since it's the same for all envs and spheres)
    sorted_phi, sort_idx = torch.sort(fixed_phi)  # sorted_phi: [n_fixed_points], sort_idx: [n_fixed_points]

    # Sort fixed_offsets using the sort indices along the last dimension
    sorted_offsets = fixed_offsets[:, :, sort_idx]  # [n_envs, n_spheres, n_fixed_points]

    # Extract phi values from the sample points
    phi_samples = points[:, 1].contiguous()  # [n_points]

    # Find interpolation indices using searchsorted
    idx = torch.searchsorted(sorted_phi, phi_samples, right=True)  # [n_points]
    idx_a = torch.clamp(idx - 1, 0, n_fixed_points - 1)  # [n_points]
    idx_b = torch.clamp(idx, 0, n_fixed_points - 1)  # [n_points]

    # Gather phi values for interpolation
    phi_a = sorted_phi[idx_a]  # [n_points]
    phi_b = sorted_phi[idx_b]  # [n_points]

    # Gather offset values using advanced indexing
    offsets_a = sorted_offsets[:, :, idx_a]  # [n_envs, n_spheres, n_points]
    offsets_b = sorted_offsets[:, :, idx_b]  # [n_envs, n_spheres, n_points]

    # Compute interpolation weights
    delta_phi = phi_b - phi_a  # [n_points]
    diff = phi_samples - phi_a  # [n_points]
    weight = torch.where(
        delta_phi > 0,
        diff / (delta_phi + epsilon),
        torch.tensor(0.0, device=device)
    )  # [n_points]
    weight = torch.clamp(weight, 0, 1)  # [n_points]

    # Perform linear interpolation of offsets
    interpolated_offset = offsets_a * (1 - weight) + offsets_b * weight  # [n_envs, n_spheres, n_points]

    # Compute the deformed radius
    radius = base_radius * interpolated_offset  # [n_envs, n_spheres, n_points]

    # Convert sample points to Cartesian unit vectors
    theta, phi = points[:, 0], points[:, 1]  # [n_points]
    x = torch.sin(phi) * torch.cos(theta)    # [n_points]
    y = torch.sin(phi) * torch.sin(theta)    # [n_points]
    z = torch.cos(phi)                       # [n_points]
    q = torch.stack([x, y, z], dim=1)        # [n_points, 3]

    # Compute final deformed Cartesian coordinates
    xyz = radius.unsqueeze(-1) * q  # [n_envs, n_spheres, n_points, 3]

    return xyz, phi, radius

def torch_compute_deformed_sphere_points_detached_fingers_static_anchors(
    fixed_phi, fixed_offsets, points, base_radius=1.0, device='cpu', epsilon=1e-6
):
    """
    Compute deformed sphere points using linear interpolation based on fixed phi values per sphere.
    
    Parameters:
    - fixed_phi: [n_spheres, n_fixed_points] tensor of fixed phi values per sphere
    - fixed_offsets: [n_envs, n_spheres, n_fixed_points] tensor of offsets at fixed phi locations
    - points: [n_points, 2] tensor of (theta, phi) sample points
    - base_radius: float or tensor, base radius of the spheres
    - device: torch.device, device to perform computations on
    - epsilon: float, small value to avoid division by zero
    
    Returns:
    - xyz: [n_envs, n_spheres, n_points, 3] tensor of deformed Cartesian coordinates
    - phi: [n_points] tensor of phi values of the sample points
    - radius: [n_envs, n_spheres, n_points] tensor of deformed radii
    """
    # Move inputs to the specified device
    fixed_phi = fixed_phi.to(device)
    fixed_offsets = fixed_offsets.to(device)
    points = points.to(device)
    if isinstance(base_radius, torch.Tensor):
        base_radius = base_radius.to(device)
    else:
        base_radius = torch.tensor(base_radius, dtype=torch.float, device=device)

    # Extract dimensions
    n_envs, n_spheres, n_fixed_points = fixed_offsets.shape

    # Sort fixed_phi for each sphere
    sorted_phi, sort_idx = torch.sort(fixed_phi, dim=-1)  # [n_spheres, n_fixed_points], [n_spheres, n_fixed_points]

    # Sort fixed_offsets for each sphere using the sort indices
    sort_idx_exp = sort_idx.unsqueeze(0).repeat(n_envs, 1, 1)  # [n_envs, n_spheres, n_fixed_points]
    sorted_offsets = fixed_offsets.gather(2, sort_idx_exp)  # [n_envs, n_spheres, n_fixed_points]

    # Extract phi values from the sample points
    phi_samples = points[:, 1].unsqueeze(0).repeat(n_spheres, 1).contiguous()  # [n_spheres, n_points]

    # Find interpolation indices for each sphere
    idx = torch.searchsorted(sorted_phi, phi_samples, right=True)  # [n_spheres, n_points]
    idx_a = torch.clamp(idx - 1, 0, n_fixed_points - 1)  # [n_spheres, n_points]
    idx_b = torch.clamp(idx, 0, n_fixed_points - 1)  # [n_spheres, n_points]

    # Gather phi values for interpolation
    phi_a = sorted_phi.gather(1, idx_a)  # [n_spheres, n_points]
    phi_b = sorted_phi.gather(1, idx_b)  # [n_spheres, n_points]

    # Gather offset values
    idx_a_exp = idx_a.unsqueeze(0).repeat(n_envs, 1, 1)  # [n_envs, n_spheres, n_points]
    idx_b_exp = idx_b.unsqueeze(0).repeat(n_envs, 1, 1)  # [n_envs, n_spheres, n_points]
    offsets_a = sorted_offsets.gather(2, idx_a_exp)  # [n_envs, n_spheres, n_points]
    offsets_b = sorted_offsets.gather(2, idx_b_exp)  # [n_envs, n_spheres, n_points]

    # Compute interpolation weights
    delta_phi = phi_b - phi_a  # [n_spheres, n_points]
    diff = phi_samples - phi_a  # [n_spheres, n_points], broadcasting phi_samples
    weight = torch.where(
        delta_phi > 0,
        diff / (delta_phi + epsilon),
        torch.tensor(0.0, device=device)
    )  # [n_spheres, n_points]
    weight = torch.clamp(weight, 0, 1)  # [n_spheres, n_points]

    # Perform linear interpolation of offsets
    interpolated_offset = offsets_a * (1 - weight) + offsets_b * weight  # [n_envs, n_spheres, n_points]

    # Compute the deformed radius
    radius = base_radius * interpolated_offset  # [n_envs, n_spheres, n_points]

    # Convert sample points to Cartesian unit vectors
    theta, phi = points[:, 0], points[:, 1]  # [n_points]
    x = torch.sin(phi) * torch.cos(theta)    # [n_points]
    y = torch.sin(phi) * torch.sin(theta)    # [n_points]
    z = torch.cos(phi)                       # [n_points]
    q = torch.stack([x, y, z], dim=1)        # [n_points, 3]

    # Compute final deformed Cartesian coordinates
    xyz = radius.unsqueeze(-1) * q  # [n_envs, n_spheres, n_points, 3]

    return xyz, phi, radius

def generate_random_translation_matrices(num_envs: int, num_fingers: int, std_dev: float) -> torch.Tensor:
    # Create batch of identity 4x4 matrices
    matrices = torch.eye(4).repeat(num_envs, num_fingers, 1, 1)
    
    # Generate random translations for x and y
    tx = torch.randn(num_envs, num_fingers) * std_dev
    ty = torch.randn(num_envs, num_fingers) * std_dev
    
    # Apply translations to matrices
    matrices[:, :, 0, 3] = tx
    matrices[:, :, 1, 3] = ty
    
    return matrices

def get_random_z_rotations(n, device):
        """
        Generate random 4x4 transformation matrices for rotations around the z-axis and their inverses.
        
        Args:
            n (int): Number of transformations.
            device (torch.device): Device to create the tensors on (e.g., 'cpu' or 'cuda').
            
        
        Returns:
            tuple: Two tensors (T, T_inv, theta), each of shape (n, 4, 4), where T contains the random rotation
                matrices and T_inv contains their inverses.
        """
        # Generate random angles in [-π, π)
        theta = (torch.rand(n, device=device) * 2 - 1) * torch.pi
        
        # Compute cos and sin
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        # Create the transformation matrices T
        T = torch.eye(4, device=device).repeat(n, 1, 1)
        T[:, 0, 0] = cos
        T[:, 0, 1] = -sin
        T[:, 1, 0] = sin
        T[:, 1, 1] = cos
        
        # Compute the inverse transformations T_inv as the transpose of T
        T_inv = T.transpose(1, 2)
        
        return T, T_inv, theta

def torch_compute_deformed_sphere_points_unfixed_planes(
    d_planes_theta, d_plane_offsets, d_vectors_phi, d_vector_offsets,
    n_pole_values, s_pole_values, points, base_radius=1.0, device='cpu'
):
    """
    Compute deformed sphere points using PyTorch tensors.

    Parameters:
    - d_planes_theta: [n_driving_planes, n_spheres] tensor
    - d_plane_offsets: [n_driving_planes, n_spheres] tensor
    - d_vectors_phi: [n_driving_vectors] tensor
    - d_vector_offsets: [n_driving_planes, n_driving_vectors, n_spheres] tensor
    - n_pole_values: [n_spheres] tensor
    - s_pole_values: [n_spheres] tensor
    - points: [n_points, 2] tensor (theta, phi)
    - base_radius: float or tensor
    - device: torch.device

    Returns:
    - xyz: [n_points, n_spheres, 3] tensor
    """
    # Define pi as a tensor on the specified device
    PI = torch.tensor(np.pi, dtype=torch.float, device=device)
    
    # Extract dimensions from input tensors
    n_spheres = d_plane_offsets.size(1)
    n_driving_planes = d_planes_theta.size(0)
    n_points = points.size(0)

    # Extend driving vectors by adding 0 and pi at the boundaries
    extended_d_vectors_phi = torch.cat([
        torch.tensor([0.0], device=device),
        d_vectors_phi,
        torch.tensor([PI], device=device)
    ]).contiguous()

    # Extend radius offsets by incorporating pole values
    n_pole_offsets = n_pole_values[None, None, :].expand(n_driving_planes, 1, -1)
    s_pole_offsets = s_pole_values[None, None, :].expand(n_driving_planes, 1, -1)
    extended_d_vector_offsets = torch.cat([n_pole_offsets, d_vector_offsets, s_pole_offsets], dim=1)

    # Extract theta and phi coordinates from points
    theta, phi = points[:, 0], points[:, 1].contiguous()

    # Adjust plane angles using per-sphere driving plane angles and offsets
    adjusted_d_planes_theta = (d_planes_theta + d_plane_offsets) % (2 * PI)
    
    # Create extended theta range for interpolation
    extended_theta = torch.cat([
        adjusted_d_planes_theta - 2 * PI,
        adjusted_d_planes_theta,
        adjusted_d_planes_theta + 2 * PI
    ], dim=0)
    extended_theta_sorted, sort_idx = torch.sort(extended_theta, dim=0)

    # Extend and sort radius offsets according to theta sorting
    extended_offsets = extended_d_vector_offsets.repeat(3, 1, 1)
    extended_offsets_sorted = torch.gather(
        extended_offsets,
        0,
        sort_idx[:, None, :].expand(-1, extended_offsets.size(1), -1)
    )

    # Transpose sorted theta for per-sphere search
    extended_theta_sorted_t = extended_theta_sorted.transpose(0, 1).contiguous()
    theta_expanded = theta[None, :].repeat(n_spheres, 1).contiguous()

    # Find insertion points for theta values in sorted theta per sphere
    idx = torch.searchsorted(extended_theta_sorted_t, theta_expanded, right=True)
    idx = torch.clamp(idx, 1, 3 * n_driving_planes - 1).T  # Shape: [n_points, n_spheres]

    # Define sphere indices for indexing
    s_indices = torch.arange(n_spheres, device=device)[None, :].expand(n_points, -1)
    
    # Get surrounding theta values for interpolation
    theta_a = extended_theta_sorted[idx - 1, s_indices]
    theta_b = extended_theta_sorted[idx, s_indices]
    fraction = (theta[:, None] - theta_a) / (theta_b - theta_a)

    # Perform phi interpolation
    idx_phi = torch.searchsorted(extended_d_vectors_phi, phi, right=True) - 1
    idx_phi = torch.clamp(
        idx_phi,
        torch.tensor(0, dtype=torch.int, device=device),
        extended_d_vectors_phi.size(0) - 2
    )
    phi_a = extended_d_vectors_phi[idx_phi]
    phi_b = extended_d_vectors_phi[idx_phi + 1]
    weight_phi = (phi - phi_a) / (phi_b - phi_a)

    # Interpolate radius offsets between surrounding theta planes
    row_indices_a, row_indices_b = idx - 1, idx
    offsets_a = extended_offsets_sorted[row_indices_a, :, s_indices]
    offsets_b = extended_offsets_sorted[row_indices_b, :, s_indices]
    p_indices = torch.arange(n_points, device=device)[:, None].expand(-1, n_spheres)
    phi_indices = idx_phi[:, None].expand(-1, n_spheres)
    offsets_a_at_phi = (1 - weight_phi[:, None]) * offsets_a[p_indices, s_indices, phi_indices] + \
                       weight_phi[:, None] * offsets_a[p_indices, s_indices, phi_indices + 1]
    offsets_b_at_phi = (1 - weight_phi[:, None]) * offsets_b[p_indices, s_indices, phi_indices] + \
                       weight_phi[:, None] * offsets_b[p_indices, s_indices, phi_indices + 1]

    # Compute final radius with interpolation
    radius_offsets = offsets_a_at_phi + fraction * (offsets_b_at_phi - offsets_a_at_phi)
    r = base_radius * (1 + radius_offsets)
    
    # Convert to Cartesian coordinates
    x = r * torch.sin(phi)[:, None] * torch.cos(theta)[:, None]
    y = r * torch.sin(phi)[:, None] * torch.sin(theta)[:, None]
    z = r * torch.cos(phi)[:, None]
    xyz = torch.stack([x, y, z], dim=2)

    return xyz

def torch_solve_for_A_joints(type_A_joints, joint_type_info, d_planes_theta, d_plane_offsets, joint_index_dict, q_all):
    """
    Solve for type A joints using PyTorch tensors.

    Parameters:
    - type_A_joints: list of joint names
    - joint_type_info: dictionary with joint information
    - d_planes_theta: [n_driving_planes] tensor
    - d_plane_offsets: [n_driving_planes, n_spheres] tensor
    - joint_index_dict: dictionary mapping joint names to indices in q_all
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    """
    q_A_anchors = torch.tensor([joint_type_info[q_A]["anchor"] for q_A in type_A_joints], device=d_planes_theta.device)
    q_A_res = torch.tensor([joint_type_info[q_A]["resolution"] for q_A in type_A_joints], device=d_planes_theta.device)
    q_A_zero_idx = torch.tensor([joint_type_info[q_A]["zero_idx"] for q_A in type_A_joints], device=d_planes_theta.device)
    q_A_max = torch.tensor([len(joint_type_info[q_A]["q_list"]) - 1 for q_A in type_A_joints], device=d_planes_theta.device)

    anchor_offsets = torch_compute_interpolated_offsets_at_angles(d_planes_theta, d_plane_offsets, q_A_anchors)
    offset_idx = torch.round(anchor_offsets / q_A_res[:, None] + q_A_zero_idx[:, None])
    offset_idx = torch.clamp(
        offset_idx,
        torch.tensor(0,dtype=torch.int, device=d_planes_theta.device), 
        q_A_max[:, None]).long()

    for i, q_A in enumerate(type_A_joints):
        q_list = torch.tensor(joint_type_info[q_A]["q_list"], device=d_planes_theta.device)
        q_all[:, joint_index_dict[q_A]] = q_list[offset_idx[i]]

def torch_get_theta_offset_from_q(type_A_joints, joint_values, res, q_A_zero_idx, q_A_max, q_anchor_dist, q_list_dict):
    """
    Compute theta offset values from given joint q_values using closest match in the lookup table.
    Handles batch inputs for joint_values and processes each joint separately.
    """
    theta_offsets = torch.zeros_like(joint_values, dtype=torch.float, device=joint_values.device)
    for i, q_A in enumerate(type_A_joints):
        q_list = q_list_dict[q_A]  # 1D tensor of q values for the joint
        # Compute absolute differences between q_list and each batch element's joint value
        diffs = torch.abs(q_list.unsqueeze(0) - joint_values[:, i].unsqueeze(1))
        # Find the index of the closest q value in the list for each batch element
        idx = torch.argmin(diffs, dim=1)
        # Compute the adjustment factor d = anchor_dist / res
        d = q_anchor_dist[i] / res[i]
        # Calculate the theta offset using the formula derived from the forward mapping
        offset = (idx.float() - q_A_zero_idx[i] - d) * res[i]
        theta_offsets[:, i] = offset
    return theta_offsets

def torch_solve_for_A_joints_unfixed_planes(type_A_joints, joint_type_info, d_planes_theta, d_plane_offsets, joint_index_dict, q_all):
    """
    Solve for type A joints using PyTorch tensors.

    Parameters:
    - type_A_joints: list of joint names
    - joint_type_info: dictionary with joint information
    - d_planes_theta: [n_driving_planes, n_spheres] tensor
    - d_plane_offsets: [n_driving_planes, n_spheres] tensor
    - joint_index_dict: dictionary mapping joint names to indices in q_all
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    """
    q_A_anchors = torch.tensor([joint_type_info[q_A]["anchor"] for q_A in type_A_joints], device=d_planes_theta.device)
    res = torch.tensor([joint_type_info[q_A]["resolution"] for q_A in type_A_joints], device=d_planes_theta.device)
    q_A_zero_idx = torch.tensor([joint_type_info[q_A]["zero_idx"] for q_A in type_A_joints], device=d_planes_theta.device)
    q_A_max = torch.tensor([len(joint_type_info[q_A]["q_list"]) - 1 for q_A in type_A_joints], device=d_planes_theta.device)

    anchor_offsets = torch_compute_interpolated_offsets_at_angles_unfixed_planes(d_planes_theta, d_plane_offsets, q_A_anchors)
    offset_idx = torch.round(anchor_offsets / res[:, None] + q_A_zero_idx[:, None])
    offset_idx = torch.clamp(
        offset_idx,
        torch.tensor(0,dtype=torch.int, device=d_planes_theta.device), 
        q_A_max[:, None]).long()

    for i, q_A in enumerate(type_A_joints):
        q_list = torch.tensor(joint_type_info[q_A]["q_list"], device=d_planes_theta.device)
        q_all[:, joint_index_dict[q_A]] = q_list[offset_idx[i]]

def torch_solve_for_A_joints_planar_offsets(type_A_joints, joint_type_info, d_planes_theta, d_plane_offsets, joint_index_dict, q_all):
    """
    Solve for type A joints using planar offsets of anchor theta plane.

    Parameters:
    - type_A_joints: list of joint names
    - joint_type_info: dictionary with joint information
    - d_planes_theta: [n_driving_planes, n_spheres] tensor
    - d_plane_offsets: [n_driving_planes, n_spheres] tensor
    - joint_index_dict: dictionary mapping joint names to indices in q_all
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    """


    for i, q_A in enumerate(type_A_joints):
        q_list = torch.tensor(joint_type_info[q_A]["q_list"], device=d_planes_theta.device)
        q_all[:, joint_index_dict[q_A]] = q_list[offset_idx[i]]

def torch_compute_interpolated_offsets_at_angles(driving_angles, driving_offsets, target_angles):
    """
    Compute interpolated offsets at target angles using PyTorch tensors.

    Parameters:
    - driving_angles: [n_driving_planes] tensor
        A 1D tensor containing the angles of the driving planes, shared across all spheres.
    - driving_offsets: [n_driving_planes, n_spheres] tensor
        A 2D tensor containing the offset values corresponding to each driving angle for each sphere.
    - target_angles: [n_targets] tensor
        A 1D tensor containing the target angles at which to compute the interpolated offsets.

    Returns:
    - offsets_at_targets: [n_targets, n_spheres] tensor
        A 2D tensor containing the interpolated offset values at each target angle for each sphere.
    """
    # Define PI as a tensor with the same device as driving_angles for consistent computations
    PI = torch.tensor(torch.pi, dtype=torch.float, device=driving_angles.device)
    
    # Wrap target angles to the range [0, 2π) to ensure they fall within one full rotation
    target_angles = target_angles % (2 * PI)
    
    # Extend driving angles by adding versions shifted by -2π and +2π to handle angle periodicity
    angles_extended = torch.cat([driving_angles - 2 * PI, driving_angles, driving_angles + 2 * PI])
    
    # Repeat the driving offsets three times to align with the extended angles
    offsets_extended = driving_offsets.repeat(3, 1)
    
    # Find the insertion points for target_angles within the sorted extended angles
    i = torch.searchsorted(angles_extended, target_angles, right=True)
    
    # Clamp indices to stay within valid bounds for interpolation
    i = torch.clamp(i, 1, len(angles_extended) - 1)
    
    # Define left and right indices for interpolation
    left_idx = i - 1
    right_idx = i
    
    # Extract the bracketing angles from the extended set
    angle_a = angles_extended[left_idx]
    angle_b = angles_extended[right_idx]
    
    # Extract the corresponding offsets for the bracketing angles
    offset_a = offsets_extended[left_idx]
    offset_b = offsets_extended[right_idx]
    
    # Calculate the interpolation fraction
    fraction = (target_angles - angle_a) / (angle_b - angle_a)
    
    # Prevent division by zero by setting fraction to 0 when angle_a equals angle_b
    fraction = torch.where(angle_a == angle_b, torch.zeros_like(fraction), fraction)
    
    # Perform linear interpolation to compute offsets at target angles
    offsets_at_targets = offset_a + fraction[:, None] * (offset_b - offset_a)
    
    return offsets_at_targets

def torch_compute_interpolated_offsets_at_angles_unfixed_planes(
    driving_angles, driving_offsets, target_angles
):
    """
    Compute interpolated offsets at target angles using PyTorch tensors, with per-sphere driving angles.

    Parameters:
    - driving_angles: [n_driving_planes, n_spheres] tensor of driving angles for each sphere
    - driving_offsets: [n_driving_planes, n_spheres] tensor of offsets corresponding to driving angles
    - target_angles: [n_targets] tensor of angles at which to compute interpolated offsets

    Returns:
    - offsets_at_targets: [n_targets, n_spheres] tensor of interpolated offsets at target angles for each sphere
    """
    # Define PI on the same device as inputs
    PI = torch.tensor(torch.pi, dtype=torch.float, device=driving_angles.device)
    
    # Wrap target angles to [0, 2π)
    target_angles = target_angles % (2 * PI)

    # Get dimensions
    n_driving_planes = driving_angles.size(0)
    n_spheres = driving_offsets.size(1)
    
    # Transpose to [n_spheres, n_driving_planes] for per-sphere operations
    driving_angles_t = driving_angles.t()
    driving_offsets_t = driving_offsets.t()
    
    # Extend angles and offsets for each sphere to handle periodicity
    angles_extended_t = torch.cat([
        driving_angles_t - 2 * PI,
        driving_angles_t,
        driving_angles_t + 2 * PI
    ], dim=1).contiguous()  # Shape: [n_spheres, 3 * n_driving_planes]
    
    offsets_extended_t = torch.cat([
        driving_offsets_t,
        driving_offsets_t,
        driving_offsets_t
    ], dim=1).contiguous()  # Shape: [n_spheres, 3 * n_driving_planes]
    
    target_angles = target_angles[None, :].repeat(n_spheres, 1).contiguous()

    # Find insertion points for target angles in each sphere's extended angles
    i = torch.searchsorted(angles_extended_t, target_angles, right=True)  # Shape: [n_spheres, n_targets]
    i = torch.clamp(i, 1, 3 * n_driving_planes - 1)
    
    # Get left and right indices for interpolation
    left_idx = i - 1   # Shape: [n_spheres, n_targets]
    right_idx = i      # Shape: [n_spheres, n_targets]
    
    # Gather surrounding angles and offsets
    angle_a = angles_extended_t.gather(1, left_idx)    # Shape: [n_spheres, n_targets]
    angle_b = angles_extended_t.gather(1, right_idx)   # Shape: [n_spheres, n_targets]
    offset_a = offsets_extended_t.gather(1, left_idx)  # Shape: [n_spheres, n_targets]
    offset_b = offsets_extended_t.gather(1, right_idx) # Shape: [n_spheres, n_targets]
    
    # Compute interpolation fraction
    fraction = (target_angles[None, :] - angle_a) / (angle_b - angle_a)  # Shape: [n_spheres, n_targets]
    fraction = torch.where(angle_a == angle_b, torch.zeros_like(fraction), fraction)
    
    # Compute interpolated offsets
    offsets_at_targets_t = offset_a + fraction * (offset_b - offset_a)  # Shape: [n_spheres, n_targets]
    
    # Transpose to match output shape [n_targets, n_spheres]
    # print("offsets_at target size", offsets_at_targets_t.size())
    offsets_at_targets = offsets_at_targets_t.squeeze(0).t()
    
    return offsets_at_targets

def torch_compute_sphere_to_single_joint_transforms(joint_tag, q_all, joints_idx, joint_info, chain, device='cpu'):
    """
    Compute transformations from sphere_frame to a single joint using PyTorch tensors.

    Parameters:
    - joint_tag: str, target joint name
    - q_all: [n_spheres, n_joints] tensor
    - joints_idx: dictionary mapping joint names to indices in q_all
    - joint_info: dictionary with joint information
    - chain: list of joint names in the kinematic chain
    - device: torch.device

    Returns:
    - T_sphere_to_joint_all: [n_spheres, 4, 4] tensor
    """
    # Extract the path from base_link to joint_tag
    path = chain[:chain.index(joint_tag) + 1]

    # Number of configurations
    n_spheres = q_all.size(0)
    T_quat = tf.quaternion_matrix(joint_info["sphere_frame"][1])

    # Compute fixed transformation from sphere_frame to base_link #! can be precomputed
    T_base_to_sphere = torch.tensor(T_quat, dtype=torch.float, device=device)
    T_base_to_sphere[:3, 3] = torch.tensor(joint_info["sphere_frame"][0], device=device)
    T_sphere_to_base = torch.inverse(T_base_to_sphere)
    

    # Initialize cumulative transformation as identity for all spheres
    T_cumulative = torch.eye(4, device=device).repeat(n_spheres, 1, 1)

    for joint in path[1:]: # Skip base_link
        # Get joint angles; use zeros if joint not in q_0_all
        if joint in joints_idx:
            q_values = q_all[:, joints_idx[joint]]
        else:
            q_values = torch.zeros(n_spheres, device=device)

        # Extract joint info
        xyz = torch.tensor(joint_info[joint][0],dtype=torch.float, device=device)
        T_quat = tf.quaternion_matrix(joint_info[joint][1])
        T_fixed = torch.tensor(T_quat, dtype=torch.float, device=device)
        joint_type = joint_info[joint][2]
        T_fixed[:3, 3] = xyz
        # print(f"Torch {joint} Transform \n", T_fixed)
        # print(f"Torch Q {joint} Values {q_values}")

        if joint_type == "revolute": 
            # Vectorized rotation matrices around z-axis
            cos_q = torch.cos(q_values)
            sin_q = torch.sin(q_values)
            R_z = torch.stack([
                cos_q, -sin_q, torch.zeros_like(cos_q),
                sin_q, cos_q, torch.zeros_like(cos_q),
                torch.zeros_like(cos_q), torch.zeros_like(cos_q), torch.ones_like(cos_q)
            ], dim=1).view(n_spheres, 3, 3)

            # Extend to 4x4 matrices
            T_rot = torch.eye(4, device=device).repeat(n_spheres, 1, 1)
            T_rot[:, :3, :3] = R_z
            T_joint = T_fixed @ T_rot
        else:
            # For fixed joints, tile the fixed transformation
            T_joint = T_fixed.repeat(n_spheres, 1, 1)

        T_cumulative = T_cumulative @ T_joint
        # print(f"Torch {joint} T_cum Transform \n", T_cumulative)
    
    # print("Torch Sphere Transform \n", T_sphere_to_base) 

    T_sphere_to_joint_all = T_sphere_to_base @ T_cumulative
    # print(f"Torch {joint_tag} T_complete Transform \n", T_sphere_to_joint_all[5])

    return T_sphere_to_joint_all

def torch_solve_for_B_joint_old(q_all, joint_tag, joint_index_dict, centroids, joint_points, joint_info, point_mask, l_ft, joint_type_info):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and centroids.

    Parameters:
    - q_all (torch.Tensor): Shape [n_spheres, n_joints], current joint angles.
    - joint_tag (str): Name of the B joint to solve for.
    - joint_index_dict (dict): Maps joint names to column indices in q_all.
    - centroids (torch.Tensor): Shape [n_spheres, 3], centroids in the joint frame (x, y, z).
    - joint_points (torch.Tensor): Shape [n_points, n_spheres, 3], points in the joint frame (x, y, z).
    - joint_info (dict): General joint information containing limits.
    - point_mask (torch.Tensor): Shape [n_points, n_spheres], boolean mask for prefiltered points.
    - l_ft (torch.Tensor): Shape [n_spheres], magnitude threshold for filtering points.

    Returns:
    - torch.Tensor: Shape [n_spheres], updated joint angles for the B joint.
    """
    # Get the column index of the B joint from the dictionary
    B_joint_idx = joint_index_dict[joint_tag]
    n_spheres = q_all.size(0)
    device = q_all.device

    if "og_type" in joint_type_info[joint_tag].keys():
        if joint_type_info[joint_tag]["og_type"] == "A":
            # Reshape centroids to [1, n_spheres, 3] to match joint_points dimensions
            centroids_reshaped = centroids.unsqueeze(0)
            # Concatenate joint_points and centroids along the points axis (dim=0)
            joint_points = torch.cat([joint_points, centroids_reshaped], dim=0)
            # Extend point_mask to include True for the centroid point
            centroid_mask = torch.ones((1, point_mask.size(1)), device=point_mask.device, dtype=torch.bool)
            point_mask = torch.cat([point_mask, centroid_mask], dim=0)

    # Compute arctan2 angles and magnitudes for all points in the x-y plane
    angles = torch.atan2(joint_points[..., 1], joint_points[..., 0])  # Shape: [n_points, n_spheres]
    magnitudes = torch.norm(joint_points[..., :2], dim=2)             # Shape: [n_points, n_spheres]
    
    # Extract joint limits from joint_info
    lower_limit = torch.tensor(joint_info[joint_tag][4],dtype = torch.float, device = device)
    upper_limit = torch.tensor(joint_info[joint_tag][5],dtype = torch.float, device = device)

    # Create angle mask for angles within [lower_limit, upper_limit]
    q_0_joint = q_all[0, B_joint_idx] # Always same for a gripper
    angle_mask = (angles >= (lower_limit-q_0_joint)) & (angles <= (upper_limit-q_0_joint))  # Shape: (n_points or n_points + 1, n_spheres)

    # Create filters to identify valid points
    mag_mask = magnitudes < l_ft[None, :]                          # Magnitude filter, shape: [n_points or n_points + 1, n_spheres]
    valid_mask = point_mask & mag_mask & angle_mask               # Combined filter, shape: [n_points or n_points + 1, n_spheres]

    # Initialize result tensor for updated angles
    result = torch.zeros(n_spheres, device=q_all.device)

    ref_dir = joint_type_info[joint_tag]["ref_dir"]

    if ref_dir[1]>0:
        result = torch.min(angles.masked_fill(~valid_mask, float('inf')), dim=0).values  
        default_result = upper_limit.repeat(n_spheres)
    else:
        result = torch.max(angles.masked_fill(~valid_mask, -float('inf')), dim=0).values 
        default_result = lower_limit.repeat(n_spheres)
    # Mask invalid angles with infinity for min calculation
    
    # masked_angles = angles.masked_fill(~valid_mask, float('inf'))     # Shape: [n_points, n_spheres]
    # min_angles = torch.min(masked_angles, dim=0).values               # Shape: [n_spheres]

    # # Mask invalid angles with negative infinity for max calculation
    # masked_angles = angles.masked_fill(~valid_mask, -float('inf'))    # Shape: [n_points, n_spheres]
    # max_angles = torch.max(masked_angles, dim=0).values               # Shape: [n_spheres]

    # Select min or max angles based on centroid's y-coordinate
    # result = torch.where(centroids[:, 1] > 0, min_angles, max_angles)  # Shape: [n_spheres]

    # Check if there are valid points for each sphere
    has_valid_points = torch.any(valid_mask, dim=0)                   # Shape: [n_spheres]

    # Apply default_result when there are no valid points
    result = torch.where(has_valid_points, result, default_result)

    # Update joint angles by adding the computed result
    new_q = q_all[:, B_joint_idx] + result

    # Clip the updated angles to stay within joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)

    return new_q

def sample_fibonacci_points_plane(N, radius=1.0, device='cpu'):
    """
    Generate N approximately equally spaced points on a unit disk using the Fibonacci spiral method.
    The points are distributed in a way that approximates uniform coverage inside the circle.
    Parameters:
    - N: int - Number of points to generate.
    - radius: float - Radius of the circle (default 1.0).
    - device: str - The device to place the tensor on (e.g., 'cpu' or 'cuda').
    Returns:
    - points: torch.Tensor (N, 2) - Tensor of (x, y) coordinates inside the circle on the specified device.
    """

    indices = torch.arange(0, N, dtype=torch.float, device=device) + 0.5
    phi = indices * (torch.pi * (3 - torch.sqrt(torch.tensor(5.0, device=device))))  # Golden angle in radians
    r = torch.sqrt(indices / N) * radius  # Radial distance for disk filling
    x = r * torch.cos(phi)
    z = r * torch.sin(phi)
    y = 0.0 * torch.sin(phi)
    points = torch.stack((x, y, z), dim=1)
    return points

def torch_solve_for_C_joint(q_all, joint_tag, joint_index_dict, centroids, joint_info, ft_ps, ft_normals):
    """
    Solve for type C joints using PyTorch tensors.

    Parameters:
    - q_all: [n_spheres, n_joints] tensor
    - joint_tag: str, joint name
    - joint_index_dict: dictionary mapping joint names to indices
    - centroids: [n_spheres, 3] tensor
    - joint_info: dictionary with joint information
    - ft_ps: [n_spheres, 3] tensor, fingertip positions
    - ft_normals: [n_spheres, 3] tensor, fingertip normals

    Returns:
    - new_q: [n_spheres] tensor, updated joint angles
    """
    # Get the column index of the C joint
    C_joint_idx = joint_index_dict[joint_tag]

    # Initialize result array
    d_c_ft = centroids - ft_ps
    theta_d = torch.atan2(d_c_ft[:, 1], d_c_ft[:, 0])
    theta_v = torch.atan2(ft_normals[:, 1], ft_normals[:, 0])
    result = theta_d - theta_v

    # Update joint angles
    new_q = q_all[:, C_joint_idx] + result

    lower_limit = joint_info[joint_tag][4]
    upper_limit = joint_info[joint_tag][5]

    # Clip the values in the specified joint column to the [lower, upper] range
    new_q = torch.clamp(new_q, lower_limit, upper_limit)
    return new_q

def torch_compute_joint_to_fingertip_transforms(joint_tag, q_all, joint_index_dict, joint_info, chain, device='cpu'):
    """
    Compute transformations from the joint to the fingertip using PyTorch tensors.

    Parameters:
    - joint_tag: str, joint name
    - q_all: [n_spheres, n_joints] tensor
    - joint_index_dict: dictionary mapping joint names to indices
    - joint_info: dictionary with joint information
    - chain: list of joint names in the kinematic chain
    - device: torch.device

    Returns:
    - T_cumulative: [n_spheres, 4, 4] tensor
    """
    idx = chain.index(joint_tag)
    subchain = chain[idx + 1:]
    n_spheres = q_all.size(0)
    T_cumulative = torch.eye(4, device=device).repeat(n_spheres, 1, 1)

    for joint in subchain:
        xyz = torch.tensor(joint_info[joint][0], device=device)
        T_quat = tf.quaternion_matrix(joint_info[joint][1])
        T_fixed = torch.tensor(T_quat, dtype=torch.float, device=device)
        joint_type = joint_info[joint][2]
        T_fixed[:3, 3] = xyz

        if joint_type == "revolute" and joint in joint_index_dict:
            q_values = q_all[:, joint_index_dict[joint]]
            cos_q = torch.cos(q_values)
            sin_q = torch.sin(q_values)
            R_z = torch.stack([
                cos_q, -sin_q, torch.zeros_like(cos_q),
                sin_q, cos_q, torch.zeros_like(cos_q),
                torch.zeros_like(cos_q), torch.zeros_like(cos_q), torch.ones_like(cos_q)
            ], dim=1).view(n_spheres, 3, 3)
            T_rot = torch.eye(4, device=device).repeat(n_spheres, 1, 1)
            T_rot[:, :3, :3] = R_z
            T_joint = T_fixed @ T_rot
        else:
            T_joint = T_fixed.repeat(n_spheres, 1, 1)

        T_cumulative = T_cumulative @ T_joint

    return T_cumulative

def torch_solve_for_D_joint_dual_behavior(q_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type D joints using PyTorch tensors.

    Parameters:
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    - joint: str, joint name
    - joint_index_dict: dictionary mapping joint names to indices
    - centroids: [n_spheres, 3] tensor
    - joint_points: [n_points, n_spheres, 3] tensor
    - joint_info: dictionary with joint information
    - chain: list of joint names in the kinematic chain
    - joint_type_info: dictionary with type-specific joint information
    - T_joint_to_fts: [n_spheres, 4, 4] tensor, transformations from joint to fingertip
    """
    device = q_all.device
    n_spheres = q_all.size(0)

    # Compute ft_mask based on fingertip position ---
    ft_x = T_joint_to_fts[:, 0, 3]  # x-coordinate of fingertip, Shape: [n_spheres]
    ft_y = T_joint_to_fts[:, 1, 3]  # y-coordinate of fingertip, Shape: [n_spheres]
    box_min = torch.tensor(joint_type_info[chain[-1]]["box_min"], device=device)  # Bounding box min #! Box min and max are in ft frame
    box_max = torch.tensor(joint_type_info[chain[-1]]["box_max"], device=device)  # Bounding box max 
    ft_mask = (ft_x >= box_min[2]) & (ft_x <= box_max[2]) & (ft_y >= box_min[2]) & (ft_y <= box_max[2])  # Shape: [n_spheres] #! SImplify


    # --- Extract fingertip position and normal for all spheres ---
    ft_ps = T_joint_to_fts[:, :3, 3]       # Fingertip positions, Shape: [n_spheres, 3]
    ft_normals = T_joint_to_fts[:, :3, 2]  # Fingertip normals, Shape: [n_spheres, 3]
    l_ft = torch.norm(ft_ps[:, :2], dim=1) # Length of fingertip in x-y plane, Shape: [n_spheres]
    
    # print("Torch solving as C:", ft_mask)
    
    # **Handle spheres where ft_mask is True (solve as Type C)**
    if ft_mask.any(): #! Just give the last dof command or don't change it or ignore first mask
        c_mask = ft_mask  # Mask for spheres to solve as Type C
        new_q_c = torch_solve_for_C_joint(
            q_all[c_mask],           # Subset of q_all, Shape: [n_c_spheres, n_joints]
            joint,
            joint_index_dict,
            centroids[c_mask],       # Shape: [n_c_spheres, 3]
            joint_info,
            ft_ps[c_mask],           # Shape: [n_c_spheres, 3]
            ft_normals[c_mask]       # Shape: [n_c_spheres, 3]
        )
        q_all[c_mask, joint_index_dict[joint]] = new_q_c  # Update q_all in place

        # print(f"Torch Type D (as C) result for {joint}:", new_q_c)

    # **Handle spheres where ft_mask is False (solve as Type B with modifications)**
    if (~ft_mask).any():
        b_mask = ~ft_mask  # Mask for spheres to solve as Type B
        n_b_spheres = b_mask.sum()  # Number of spheres where ft_mask is False
        joint_points_b = joint_points[:, b_mask, :]  # Shape: [n_points, n_b_spheres, 3]
        centroids_b = centroids[b_mask]              # Shape: [n_b_spheres, 3]
        T_joint_to_fts_b = T_joint_to_fts[b_mask]    # Shape: [n_b_spheres, 4, 4]
        l_ft_b = l_ft[b_mask]                        # Shape: [n_b_spheres]
        q_all_b = q_all[b_mask]                      # Shape: [n_b_spheres, n_joints]

        # Get original joint type and fingertip position
        og_type = joint_type_info[joint]["og_type"]
        ft_pos_b = T_joint_to_fts_b[:, :3, 3]  # Shape: [n_b_spheres, 3]

        # Initialize point mask for B spheres
        point_mask_b = torch.zeros((joint_points.shape[0], n_b_spheres), dtype=torch.bool, device=device)

        # Homogeneous coordinates for joint points
        joint_points_h_b = torch.cat([joint_points_b, torch.ones_like(joint_points_b[..., :1])], dim=2) 

        # --- Apply rotation based on original joint type ---
        if og_type == "C":
            # Rotation around x-axis to align based on y and z
            alpha = torch.atan2(-ft_pos_b[:, 1], ft_pos_b[:, 2])  # Shape: [n_b_spheres]
            cos_alpha = torch.cos(alpha)
            sin_alpha = torch.sin(alpha)
            R_x = torch.stack([
                torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
                torch.zeros_like(alpha), cos_alpha, -sin_alpha,
                torch.zeros_like(alpha), sin_alpha, cos_alpha
            ], dim=1).view(-1, 3, 3)  # Shape: [n_b_spheres, 3, 3]

            # Create and invert 4x4 transformation matrix
            T_rot = torch.eye(4, device=device).repeat(n_b_spheres, 1, 1)  # Shape: [n_b_spheres, 4, 4]
            T_rot[:, :3, :3] = R_x
            T_rot = torch.inverse(T_rot)

            # Apply rotation to points
            rotated_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h_b)  # Shape: [n_points, n_b_spheres, 4]
            rotated_points = rotated_points_h[:, :, :3]  # Shape: [n_points, n_b_spheres, 3]

            # Update point mask based on rotated y-component
            point_mask_b = (rotated_points[:, :, 1] >= box_min[2]) & (rotated_points[:, :, 1] <= box_max[2])

        elif og_type == "B":
            # Rotation around y-axis to align based on x and z
            beta = torch.atan2(-ft_pos_b[:, 2], ft_pos_b[:, 0])  # Shape: [n_b_spheres]
            cos_beta = torch.cos(beta)
            sin_beta = torch.sin(beta)
            R_y = torch.stack([
                cos_beta, torch.zeros_like(beta), sin_beta,
                torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
                -sin_beta, torch.zeros_like(beta), cos_beta
            ], dim=1).view(-1, 3, 3)  # Shape: [n_b_spheres, 3, 3]

            # Create and invert 4x4 transformation matrix
            T_rot = torch.eye(4, device=device).repeat(n_b_spheres, 1, 1)  # Shape: [n_b_spheres, 4, 4]
            T_rot[:, :3, :3] = R_y
            T_rot = torch.inverse(T_rot)

            # Apply rotation to points
            rotated_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h_b)  # Shape: [n_points, n_b_spheres, 4]
            rotated_points = rotated_points_h[:, :, :3]  # Shape: [n_points, n_b_spheres, 3]

            # Update point mask based on rotated z-component
            point_mask_b = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])

        else:
            raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

        # --- Collision check with next joint ---
        j_idx = chain.index(joint) + 1  # Index of the next joint in the chain
        l_nj = joint_info[chain[j_idx]][0]  # Translation to next joint (list or array)
        mag = torch.norm(torch.tensor(l_nj[:2], device=device))  # Magnitude in x-y plane

        # Get bounding box for the current joint (default to zeros if not specified)
        nj_box_min = torch.tensor(joint_type_info[joint].get("box_min", [0.0, 0.0, 0.0]), dtype=torch.float, device=device) if joint_type_info[joint]["box_min"] is not None else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)
        nj_box_max = torch.tensor(joint_type_info[joint].get("box_max", [0.0, 0.0, 0.0]), dtype=torch.float, device=device) if joint_type_info[joint]["box_max"] is not None else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)

        # Compute magnitudes of points in x-y plane
        magnitudes = torch.norm(joint_points_b[..., :2], dim=2)  # Shape: [n_points, n_b_spheres]

        # Create mask for collision check
        new_mask = (joint_points_b[:, :, 2] >= nj_box_min[2]) & \
                   (joint_points_b[:, :, 2] <= nj_box_max[2]) & \
                   (magnitudes < mag)  # Shape: [n_points, n_b_spheres]

        # Combine with existing point mask
        point_mask_b = new_mask | point_mask_b

        # --- Align fingertip to x-axis ---
        sigma = torch.atan2(ft_pos_b[:, 1], ft_pos_b[:, 0])  # Rotation angle, Shape: [n_b_spheres]
        cos_sigma = torch.cos(sigma)
        sin_sigma = torch.sin(sigma)
        R_z = torch.stack([
            cos_sigma, -sin_sigma, torch.zeros_like(sigma),
            sin_sigma, cos_sigma, torch.zeros_like(sigma),
            torch.zeros_like(sigma), torch.zeros_like(sigma), torch.ones_like(sigma)
        ], dim=1).view(-1, 3, 3)  # Rotation matrix around z-axis, Shape: [n_b_spheres, 3, 3]

        # Create and invert 4x4 transformation matrix
        T_rot = torch.eye(4, device=device).repeat(n_b_spheres, 1, 1)  # Shape: [n_b_spheres, 4, 4]
        T_rot[:, :3, :3] = R_z
        T_rot = torch.inverse(T_rot)

        # Apply rotation to align points
        new_joint_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h_b)  # Shape: [n_points, n_b_spheres, 4]
        joint_points_b = new_joint_points_h[:, :, :3]  # Update joint points, Shape: [n_points, n_b_spheres, 3]

        # --- Solve as Type B with transformed points ---
        new_q_b = torch_solve_for_B_joint(
            q_all_b,           # Shape: [n_b_spheres, n_joints]
            joint,
            joint_index_dict,
            centroids_b,       # Shape: [n_b_spheres, 3]
            joint_points_b,    # Transformed points, Shape: [n_points, n_b_spheres, 3]
            joint_info,
            point_mask_b,      # Shape: [n_points, n_b_spheres]
            l_ft_b             # Shape: [n_b_spheres]
        )
        q_all[b_mask, joint_index_dict[joint]] = new_q_b  # Update q_all in place

def torch_solve_for_D_joint(q_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info, T_joint_to_fts):
    """
    Solve for type D joints using PyTorch tensors.

    Parameters:
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    - joint: str, joint name
    - joint_index_dict: dictionary mapping joint names to indices
    - centroids: [n_spheres, 3] tensor
    - joint_points: [n_points, n_spheres, 3] tensor
    - joint_info: dictionary with joint information
    - chain: list of joint names in the kinematic chain
    - joint_type_info: dictionary with type-specific joint information
    - T_joint_to_fts: [n_spheres, 4, 4] tensor, transformations from joint to fingertip
    """
    device = q_all.device
    n_spheres = q_all.size(0)

    # --- Extract fingertip position and normal for all spheres ---
    ft_pos = T_joint_to_fts[:, :3, 3]       # Fingertip positions, Shape: [n_spheres, 3]
    l_ft = torch.norm(ft_pos[:, :2], dim=1) # Length of fingertip in x-y plane, Shape: [n_spheres]

    # Compute ft_mask based on fingertip position ---
    box_min = torch.tensor(joint_type_info[chain[-1]]["box_min"], device=device)  # Bounding box min
    box_max = torch.tensor(joint_type_info[chain[-1]]["box_max"], device=device)  # Bounding box max 


    # Get original joint type and fingertip position
    og_type = joint_type_info[joint]["og_type"]

    # **Solve as Type B with modifications)**

    # Initialize point mask for B spheres
    point_mask = torch.zeros((joint_points.shape[0], n_spheres), dtype=torch.bool, device=device)

    # Homogeneous coordinates for joint points
    joint_points_h = torch.cat([joint_points, torch.ones_like(joint_points[..., :1])], dim=2) 

    # --- Apply rotation based on original joint type ---
    if og_type == "C":
        # Rotation around x-axis to align based on y and z
        alpha = torch.atan2(-ft_pos[:, 1], ft_pos[:, 2])  # Shape: [n_spheres]
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)
        R_x = torch.stack([
            torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
            torch.zeros_like(alpha), cos_alpha, -sin_alpha,
            torch.zeros_like(alpha), sin_alpha, cos_alpha
        ], dim=1).view(-1, 3, 3)  # Shape: [n_spheres, 3, 3]

        # Create and invert 4x4 transformation matrix
        T_rot = torch.eye(4, device=device).repeat(n_spheres, 1, 1)  # Shape: [n_spheres, 4, 4]
        T_rot[:, :3, :3] = R_x
        T_rot = torch.inverse(T_rot)

        # Apply rotation to points
        rotated_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h)  # Shape: [n_points, n_spheres, 4]
        rotated_points = rotated_points_h[:, :, :3]  # Shape: [n_points, n_spheres, 3]

        # Update point mask based on rotated y-component
        point_mask = (rotated_points[:, :, 1] >= box_min[2]) & (rotated_points[:, :, 1] <= box_max[2])

    elif og_type == "B":
        # Rotation around y-axis to align based on x and z
        beta = torch.atan2(-ft_pos[:, 2], ft_pos[:, 0])  # Shape: [n_spheres]
        cos_beta = torch.cos(beta)
        sin_beta = torch.sin(beta)
        R_y = torch.stack([
            cos_beta, torch.zeros_like(beta), sin_beta,
            torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
            -sin_beta, torch.zeros_like(beta), cos_beta
        ], dim=1).view(-1, 3, 3)  # Shape: [n_spheres, 3, 3]

        # Create and invert 4x4 transformation matrix
        T_rot = torch.eye(4, device=device).repeat(n_spheres, 1, 1)  # Shape: [n_spheres, 4, 4]
        T_rot[:, :3, :3] = R_y
        T_rot = torch.inverse(T_rot)

        # Apply rotation to points
        rotated_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h)  # Shape: [n_points, n_spheres, 4]
        rotated_points = rotated_points_h[:, :, :3]  # Shape: [n_points, n_spheres, 3]

        # Update point mask based on rotated z-component
        point_mask = (rotated_points[:, :, 2] >= box_min[1]) & (rotated_points[:, :, 2] <= box_max[1])

    else:
        raise ValueError(f"Unknown og_type '{og_type}' for joint {joint}")

    # --- Collision check with next joint ---
    j_idx = chain.index(joint) + 1  # Index of the next joint in the chain
    l_nj = joint_info[chain[j_idx]][0]  # Translation to next joint (list or array)
    mag = torch.norm(torch.tensor(l_nj[:2], device=device))  # Magnitude in x-y plane

    # Get bounding box for the current joint (default to zeros if not specified)
    nj_box_min = torch.tensor(joint_type_info[joint].get("box_min", [0.0, 0.0, 0.0]), dtype=torch.float, device=device) if joint_type_info[joint]["box_min"] is not None else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)
    nj_box_max = torch.tensor(joint_type_info[joint].get("box_max", [0.0, 0.0, 0.0]), dtype=torch.float, device=device) if joint_type_info[joint]["box_max"] is not None else torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device)

    # Compute magnitudes of points in x-y plane
    magnitudes = torch.norm(joint_points[..., :2], dim=2)  # Shape: [n_points, n_spheres]

    # Create mask for collision check
    new_mask = (joint_points[:, :, 2] >= nj_box_min[2]) & \
                (joint_points[:, :, 2] <= nj_box_max[2]) & \
                (magnitudes < mag)  # Shape: [n_points, n_spheres]

    # Combine with existing point mask
    point_mask = new_mask | point_mask

    # --- Align fingertip to x-axis ---
    sigma = torch.atan2(ft_pos[:, 1], ft_pos[:, 0])  # Rotation angle, Shape: [n_spheres]
    cos_sigma = torch.cos(sigma)
    sin_sigma = torch.sin(sigma)
    R_z = torch.stack([
        cos_sigma, -sin_sigma, torch.zeros_like(sigma),
        sin_sigma, cos_sigma, torch.zeros_like(sigma),
        torch.zeros_like(sigma), torch.zeros_like(sigma), torch.ones_like(sigma)
    ], dim=1).view(-1, 3, 3)  # Rotation matrix around z-axis, Shape: [n_spheres, 3, 3]

    # Create and invert 4x4 transformation matrix
    T_rot = torch.eye(4, device=device).repeat(n_spheres, 1, 1)  # Shape: [n_spheres, 4, 4]
    T_rot[:, :3, :3] = R_z
    T_rot = torch.inverse(T_rot)

    # Apply rotation to align points
    new_joint_points_h = torch.einsum('bij,pbj->pbi', T_rot, joint_points_h)  # Shape: [n_points, n_spheres, 4]
    joint_points = new_joint_points_h[:, :, :3]  # Update joint points, Shape: [n_points, n_spheres, 3]

    # --- Solve as Type B with transformed points ---
    q_all[:, joint_index_dict[joint]] = torch_solve_for_B_joint(
        q_all,           # Shape: [n_spheres, n_joints]
        joint,
        joint_index_dict,
        centroids,       # Shape: [n_spheres, 3]
        joint_points,    # Transformed points, Shape: [n_points, n_spheres, 3]
        joint_info,
        point_mask,      # Shape: [n_points, n_spheres]
        l_ft,            # Shape: [n_spheres]
        joint_type_info
    )

def torch_solve_for_D_joint_ohne_fingertip(q_all, joint, joint_index_dict, centroids, joint_points, joint_info, chain, joint_type_info):
    """
    Solve for type D joints using PyTorch tensors.

    Parameters:
    - q_all: [n_spheres, n_joints] tensor to be updated in place
    - joint: str, joint name
    - joint_index_dict: dictionary mapping joint names to indices
    - centroids: [n_spheres, 3] tensor
    - joint_points: [n_points, n_spheres, 3] tensor
    - joint_info: dictionary with joint information
    - chain: list of joint names in the kinematic chain
    - joint_type_info: dictionary with type-specific joint information
    - T_joint_to_fts: [n_spheres, 4, 4] tensor, transformations from joint to fingertip
    """
    device = q_all.device

    # --- Extract fingertip position and normal for all spheres ---
    nj_pos = torch.tensor(joint_type_info[joint]["nj"], dtype = torch.float, device = device)
    l_nj = torch.norm(nj_pos[:2])

    # Get bounding box for the current joint (default to zeros if not specified)
    nj_box_min = torch.tensor(joint_type_info[joint].get("box_min", [0.0, 0.0, 0.0]), dtype=torch.float, device=device)
    nj_box_max = torch.tensor(joint_type_info[joint].get("box_max", [0.0, 0.0, 0.0]), dtype=torch.float, device=device) 

    # Compute magnitudes of points in x-y plane
    magnitudes = torch.norm(joint_points[..., :2], dim=2)  # Shape: [n_points, n_spheres]

    # Create mask for collision check
    point_mask = (joint_points[:, :, 2] >= nj_box_min[2]) & \
                (joint_points[:, :, 2] <= nj_box_max[2]) & \
                (magnitudes < l_nj)  # Shape: [n_points, n_spheres]

    # --- Solve as Type B with transformed points ---
    q_all[:, joint_index_dict[joint]] = torch_solve_for_B_joint(
        q_all,           # Shape: [n_spheres, n_joints]
        joint,
        joint_index_dict,
        centroids,       # Shape: [n_spheres, 3]
        joint_points,    # Transformed points, Shape: [n_points, n_spheres, 3]
        joint_info,
        point_mask,      # Shape: [n_points, n_spheres]
        l_nj.repeat(q_all.size(0)),            # Shape: [n_spheres]
        joint_type_info
    )

def solve_for_lateral_joints(joint_tag, q, joint_index_dict, joint_points, joint_limits, joint_type_info, T_joint_to_ft, device):
    """
    Compute updated joint angles for a lateral joint by finding the point closest to the y-axis in the fingertip frame
    per environment, calculating angles A and B in the joint frame, and selecting the angle difference with minimal
    absolute value, constrained by joint limits.
    
    Parameters:
    - joint_tag (str): Name of the lateral joint to solve for.
    - q: torch.Tensor - Joint values at initial positions, shape [num_envs, num_dofs].
    - joint_index_dict (dict): Maps joint names to indices in q.
    - joint_points (torch.Tensor): Shape [num_envs, num_points, 3], points in the joint frame (x, y, z).
    - joint_info (dict): General joint information containing limits.
    - joint_type_info (dict): Type joint information of joint.
    - T_joint_to_ft (torch.Tensor): Shape [num_envs, 4, 4], transformation from joint to fingertip frame.
    - device (str): Device for tensor operations (e.g., 'cpu' or 'cuda').
    
    Returns:
    - torch.Tensor: Shape [num_envs], updated joint angles for the lateral joint.
    """
    num_envs = joint_points.size(0)

    # print("solving for", joint_tag)
    joint_idx = joint_index_dict[joint_tag]

    # Transform points to fingertip frame
    T_ft_to_joint = torch.inverse(T_joint_to_ft)
    ones = torch.ones_like(joint_points[..., :1])  # [num_envs, num_points, 1]
    points_h = torch.cat((joint_points, ones), dim=-1)  # [num_envs, num_points, 4]
    points_ft_h = torch.einsum('eij,epj->epi', T_ft_to_joint, points_h)  # [num_envs, num_points, 4]
    points_ft = points_ft_h[..., :3]  # [num_envs, num_points, 3]

    # Compute distance to y-axis in fingertip frame: norm of (x, z) for each point
    dist_to_y_axis = torch.norm(points_ft[..., [0, 2]], dim=-1)  # [num_envs, num_points]

    # Find the index of the point with minimal distance per environment
    min_idx = torch.argmin(dist_to_y_axis, dim=1)  # [num_envs]

    # Gather the closest point per environment in joint frame (directly from input)
    closest_point_joint = joint_points[torch.arange(num_envs), min_idx]  # [num_envs, 3]
    # print("closest_point_joint", closest_point_joint)
    # print("closest_point_joint in ft", points_ft[torch.arange(num_envs), min_idx])

    # Compute angle A: arctan2(y, x) for fingertip and closest point in joint frame
    ft_pos_joint = T_joint_to_ft[..., :3, 3]  # [num_envs, 3], fingertip position in joint frame
    angle_A_ft = torch.atan2(ft_pos_joint[..., 1], ft_pos_joint[..., 0] + 1e-6)  # [num_envs]
    angle_A_point = torch.atan2(closest_point_joint[..., 1], closest_point_joint[..., 0] + 1e-6)  # [num_envs]

    # Compute angle B: arctan2(-y, -x) for fingertip and closest point
    angle_B_ft = torch.atan2(-ft_pos_joint[..., 1], -ft_pos_joint[..., 0] - 1e-6)  # [num_envs]
    angle_B_point = torch.atan2(-closest_point_joint[..., 1], -closest_point_joint[..., 0] - 1e-6)  # [num_envs]

    # print("ft_pos_joint", ft_pos_joint)

    # Compute angle differences
    diff_A = angle_A_point - angle_A_ft  # [num_envs]
    diff_B = angle_B_point - angle_B_ft  # [num_envs]
    # print("angle_A_point", angle_A_point)
    # print("angle_A_ft", angle_A_ft)
    # print("angle_B_point", angle_B_point)
    # print("angle_B_ft", angle_B_ft)

    # Select the difference with minimal absolute value, preserving sign
    abs_diff_A = torch.abs(diff_A)
    abs_diff_B = torch.abs(diff_B)
    use_A = abs_diff_A <= abs_diff_B
    result = torch.where(use_A, diff_A, diff_B)  # [num_envs]
    # result = diff_A

    # Extract joint limits
    lower_limit, upper_limit = joint_limits

    # Clip the result to stay within joint limits
    new_q = result + q[:, joint_idx]
    # new_q = result 
    result = torch.clamp(new_q, lower_limit, upper_limit)

    return result


def torch_solve_for_B_joint_with_eigenvectors_old(
    joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask,
    joint_type_info, c, th=0.0
):
    """
    Compute B joint angles with NaN safety: defaults to limit if invalid.
    """
    n_spheres = joint_points.size(1)
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits

    # Extract y for filtering
    y_values = joint_points[..., 1]  # [n_points, n_spheres]
    ref_dir = joint_type_info[joint_tag]["ref_dir"]

    # Compute reference y per sphere
    if ref_dir[1] > 0:
        y_ref = torch.min(y_values.masked_fill(~point_mask, float('inf')), dim=0).values
    else:
        y_ref = torch.max(y_values.masked_fill(~point_mask, -float('inf')), dim=0).values

    # y-distance mask
    y_diff = torch.abs(y_values - y_ref.unsqueeze(0))
    new_y_mask = y_diff <= c
    final_mask = point_mask & new_y_mask

    # xy points and mask
    points_xy = joint_points[..., :2]  # [n_points, n_spheres, 2]
    mask_float = final_mask.float()

    # Scatter matrix components
    sum_xx = torch.sum(points_xy[..., 0]**2 * mask_float, dim=0)
    sum_yy = torch.sum(points_xy[..., 1]**2 * mask_float, dim=0)
    sum_xy = torch.sum(points_xy[..., 0] * points_xy[..., 1] * mask_float, dim=0)

    # Build scatter matrix S
    S = torch.zeros(n_spheres, 2, 2, device=device)
    S[:, 0, 0] = sum_xx
    S[:, 1, 1] = sum_yy
    S[:, 0, 1] = S[:, 1, 0] = sum_xy

    S_det = sum_xx * sum_yy - sum_xy**2
    valid_S = S_det > 1e-8  # Avoid near-zero matrices

    # Compute eigenvectors only where valid
    eigenvalues = torch.zeros(n_spheres, 2, device=device)
    eigenvectors = torch.zeros(n_spheres, 2, 2, device=device)
    if valid_S.any():
        eigenvalues_valid, eigenvectors_valid = torch.linalg.eigh(S[valid_S])
        eigenvalues[valid_S] = eigenvalues_valid
        eigenvectors[valid_S] = eigenvectors_valid

    # Get principal direction
    idx = torch.argmax(eigenvalues, dim=1)
    v = eigenvectors[torch.arange(n_spheres), :, idx]  # [n_spheres, 2]

    # === NaN Guard 2: Safe mean_xy (avoid div by zero) ===
    num_valid = torch.sum(mask_float, dim=0).unsqueeze(-1)  # [n_spheres, 1]
    mean_xy = torch.zeros(n_spheres, 2, device=device)
    has_points = num_valid.squeeze(-1) > 0
    if has_points.any():
        safe_sum = torch.sum(points_xy * mask_float.unsqueeze(-1), dim=0)
        mean_xy[has_points] = safe_sum[has_points] / num_valid[has_points].clamp(min=1e-8)

    # Flip v to point toward mean_xy
    proj = torch.sum(v * mean_xy, dim=1)
    sign_flip = proj < 0
    v = torch.where(sign_flip.unsqueeze(1), -v, v)

    # === NaN Guard 3: Safe atan2 ===
    v_x, v_y = v[:, 0], v[:, 1]
    angles = torch.atan2(v_y, v_x)

    # Apply threshold
    if th > 0.0:
        angles = torch.where(torch.abs(angles) < th, torch.zeros_like(angles), angles)

    # === Final NaN Guard: Use default if invalid ===
    if ref_dir[1] > 0:
        default_result = upper_limit.repeat(n_spheres)
    else:
        default_result = lower_limit.repeat(n_spheres)

    # Invalid if: no valid points OR S degenerate OR v is zero vector
    v_norm = torch.norm(v, dim=1)
    valid_result = has_points & valid_S & (v_norm > 1e-6)

    result = q[:, joint_idx] + angles
    result = torch.where(valid_result, result, default_result)

    # Final update and clamp
    new_q = torch.clamp(new_q, lower_limit, upper_limit)

    # === FINAL NaN CLEANUP: Replace any NaN with default ===
    nan_mask = torch.isnan(new_q)
    if nan_mask.any():
        new_q = torch.where(nan_mask, default_result, new_q)

    return new_q

def torch_solve_for_B_joint_with_eigenvectors_no_NAN_guard(joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask, joint_type_info, c, th = 0.0):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and centroids.
    Parameters:
    - joint_tag (str): Name of the B joint to solve for.
    - q: Joint Values at initial positions
    - joint_index_dict (dict): Maps joint names to indices in q.
    - joint_points (torch.Tensor): Shape [n_points, n_spheres, 3], points in the joint frame (x, y, z).
    - joint_limits: Tuple of (lower_limit, upper_limit) for the joint.
    - point_mask (torch.Tensor): Shape [n_points, n_spheres], boolean mask for prefiltered points.
    - joint_type_info (dict): Type joint information of joint.
    - c (float): The constant for y-direction filtering.
    Returns:
    - torch.Tensor: Shape [n_spheres], updated joint angles for the B joint.
    """
    # Get the column index of the B joint from the dictionary
    n_spheres = joint_points.size(1)
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits

    # Extract y coordinates for filtering
    y_values = joint_points[..., 1]  # Shape: [n_points, n_spheres]
    
    # Fetch ref_dir early for both filtering and defaults
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    
    # Compute reference y value per sphere based on ref_dir[1]
    if ref_dir[1] > 0:
        y_ref = torch.min(y_values.masked_fill(~point_mask, float('inf')), dim=0).values  # Shape: [n_spheres]
    else:
        y_ref = torch.max(y_values.masked_fill(~point_mask, -float('inf')), dim=0).values  # Shape: [n_spheres]
    
    # Compute absolute y differences from reference
    y_diff = torch.abs(y_values - y_ref.unsqueeze(0))  # Shape: [n_points, n_spheres]
    
    # Create new mask based on y distance <= c
    new_y_mask = y_diff <= c  # Shape: [n_points, n_spheres]
    
    # Combine with original point_mask
    final_mask = point_mask & new_y_mask

    # Extract x and y coordinates for all points across spheres
    points_xy = joint_points[..., :2]  # Shape: [n_points, n_spheres, 2]

    # Convert mask to float for element-wise multiplication in sums
    mask_float = final_mask.float()  # Shape: [n_points, n_spheres]

    # Compute components of the 2x2 scatter matrix per sphere using masked sums
    sum_xx = torch.sum(points_xy[..., 0]**2 * mask_float, dim=0)  # Shape: [n_spheres]
    sum_yy = torch.sum(points_xy[..., 1]**2 * mask_float, dim=0)  # Shape: [n_spheres]
    sum_xy = torch.sum(points_xy[..., 0] * points_xy[..., 1] * mask_float, dim=0)  # Shape: [n_spheres]

    # Construct batched 2x2 scatter matrices for all spheres
    S = torch.zeros(n_spheres, 2, 2, device=device)  # Shape: [n_spheres, 2, 2]
    S[:, 0, 0] = sum_xx
    S[:, 1, 1] = sum_yy
    S[:, 0, 1] = sum_xy
    S[:, 1, 0] = sum_xy

    # Compute eigenvalues and eigenvectors for batched symmetric matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(S)  # eigenvalues: [n_spheres, 2], eigenvectors: [n_spheres, 2, 2]

    # Identify index of the largest eigenvalue per sphere
    idx = torch.argmax(eigenvalues, dim=1)  # Shape: [n_spheres]

    # Extract the corresponding eigenvector (direction vector) per sphere
    v = eigenvectors[torch.arange(n_spheres), :, idx]  # Shape: [n_spheres, 2]

    # Compute mean xy per sphere to orient the direction vector
    num_valid = torch.sum(mask_float, dim=0).unsqueeze(-1)  # Shape: [n_spheres, 1]
    mean_xy = torch.sum(points_xy * mask_float.unsqueeze(-1), dim=0) / num_valid.clamp(min=1e-6)  # Avoid div by zero, but overwritten later
    
    # Compute projection of v onto mean_xy
    proj = torch.sum(v * mean_xy, dim=1)  # Shape: [n_spheres]
    
    # Flip sign of v if projection is negative
    sign_flip = proj < 0  # Shape: [n_spheres]
    v = torch.where(sign_flip.unsqueeze(1), -v, v)

    # Compute the angle of the direction vector with respect to the x-axis per sphere
    angles = torch.atan2(v[:, 1], v[:, 0])  # Shape: [n_spheres]

    # Set angles to zero if their absolute value is below the threshold
    if th > 0.0:
        angles = torch.where(torch.abs(angles) < th, torch.zeros_like(angles), angles)

    # Determine default result based on reference direction
    if ref_dir[1] > 0:
        default_result = upper_limit.repeat(n_spheres)
    else:
        default_result = lower_limit.repeat(n_spheres)

    # Check for valid points per sphere
    has_valid_points = torch.any(final_mask, dim=0)  # Shape: [n_spheres]

    # Set result to computed angles or default where no valid points
    result = torch.where(has_valid_points, angles, default_result)

    # Add to original joint angles
    new_q = result + q[:, joint_idx]

    # Clip the updated angles to stay within joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)
    return new_q

def torch_solve_for_B_joint_with_mean_angles(joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask, joint_type_info, c, th=0.0):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and mean angles.
    Parameters:
    - joint_tag (str): Name of the B joint to solve for.
    - q: Joint Values at initial positions
    - joint_index_dict (dict): Maps joint names to indices in q.
    - joint_points (torch.Tensor): Shape [n_points, n_spheres, 3], points in the joint frame (x, y, z).
    - joint_limits: Tuple of (lower_limit, upper_limit) for the joint.
    - point_mask (torch.Tensor): Shape [n_points, n_spheres], boolean mask for prefiltered points.
    - joint_type_info (dict): Type joint information of joint.
    - c (float): The constant for y-direction filtering.
    Returns:
    - torch.Tensor: Shape [n_spheres], updated joint angles for the B joint.
    """
    import torch

    # Get the column index of the B joint from the dictionary
    n_spheres = joint_points.size(1)
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits

    # Extract y coordinates for filtering
    y_values = joint_points[..., 1]  # Shape: [n_points, n_spheres]
    
    # Fetch ref_dir early for both filtering and defaults
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    
    # Compute reference y value per sphere based on ref_dir[1]
    if ref_dir[1] > 0:
        y_ref = torch.min(y_values.masked_fill(~point_mask, float('inf')), dim=0).values  # Shape: [n_spheres]
    else:
        y_ref = torch.max(y_values.masked_fill(~point_mask, -float('inf')), dim=0).values  # Shape: [n_spheres]
    
    # Compute absolute y differences from reference
    y_diff = torch.abs(y_values - y_ref.unsqueeze(0))  # Shape: [n_points, n_spheres]
    
    # Create new mask based on y distance <= c
    new_y_mask = y_diff <= c  # Shape: [n_points, n_spheres]
    
    # Combine with original point_mask
    final_mask = point_mask & new_y_mask

    # Extract x and y coordinates for all points across spheres
    points_xy = joint_points[..., :2]  # Shape: [n_points, n_spheres, 2]

    # Convert mask to float for element-wise multiplication in sums
    mask_float = final_mask.float()  # Shape: [n_points, n_spheres]

    # Compute unit vectors
    norms = torch.norm(points_xy, dim=-1)  # Shape: [n_points, n_spheres]
    unit_xy = points_xy / norms.unsqueeze(-1).clamp(min=1e-6)  # Shape: [n_points, n_spheres, 2]

    # Compute sum of unit_xy per sphere
    sum_unit_xy = torch.sum(unit_xy * mask_float.unsqueeze(-1), dim=0)  # Shape: [n_spheres, 2]

    # Compute number of valid points per sphere
    num_valid = torch.sum(mask_float, dim=0).unsqueeze(-1)  # Shape: [n_spheres, 1]

    # Compute mean unit_xy per sphere
    mean_unit_xy = sum_unit_xy / num_valid.clamp(min=1e-6)  # Shape: [n_spheres, 2]

    # Compute the angle of the mean direction vector per sphere
    angles = torch.atan2(mean_unit_xy[:, 1], mean_unit_xy[:, 0])  # Shape: [n_spheres]

    # Set angles to zero if their absolute value is below the threshold
    if th > 0.0:
        angles = torch.where(torch.abs(angles) < th, torch.zeros_like(angles), angles)

    # Determine default result based on reference direction
    if ref_dir[1] > 0:
        default_result = upper_limit.repeat(n_spheres)
    else:
        default_result = lower_limit.repeat(n_spheres)

    # Check for valid points per sphere
    has_valid_points = torch.any(final_mask, dim=0)  # Shape: [n_spheres]

    # Set result to computed angles or default where no valid points
    result = torch.where(has_valid_points, angles, default_result)

    # Add to original joint angles
    new_q = result + q[:, joint_idx]

    # Clip the updated angles to stay within joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)
    return new_q


def torch_solve_for_A_joint_quadrant_care_multi_many(
    joint_tag, q, joint_index_dict, points_h, ft_joint, joint_limits,
    min_radius=2.5e-3  
):
    """
    Compute updated joint angles for an A joint across all environments based on multiple rotated fingertip sphere positions.
    For each env, filters pairs where dist1 and dist2 > min_radius, selects the pair with highest index among valid, and computes angle.
    If no valid pairs, keeps current q.
    Parameters:
    - joint_tag (str): Name of the A joint
    - q (torch.Tensor): [num_envs, num_joints]
    - joint_index_dict (dict): joint name → column index
    - T_joint_to_sphere (torch.Tensor): [num_envs, 4, 4]
    - points_h (torch.Tensor): [num_envs, num_points, 4] homogeneous positions
    - ft_joint (torch.Tensor): [num_envs, num_points, 3] or [num_envs, num_points, 4] reference vectors
    - joint_limits (tuple): (lower_limit, upper_limit)
    - min_radius (float): min distance from joint origin for reliability (default: 5e-3)
    Returns:
    - torch.Tensor: [num_envs] updated joint angles
    """
    num_envs = points_h.size(0)
    num_points = points_h.size(1)
    device = points_h.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits
    # Transform all points to joint frame
    # rotated_joint = torch.matmul(T_joint_to_sphere.unsqueeze(1), points_h.unsqueeze(-1)).squeeze(-1)  # [num_envs, num_points, 4]
    # Extract 3D vectors
    rotated_joint_vec = points_h[:, :, :3]  # [num_envs, num_points, 3]
    ft_joint_vec = ft_joint[:, :, :3]  # [num_envs, num_points, 3]
    # Compute distances from joint origin (xy norm)
    dist1 = torch.norm(rotated_joint_vec[:, :, :2], dim=-1)  # [num_envs, num_points]
    dist2 = torch.norm(ft_joint_vec[:, :, :2], dim=-1)  # [num_envs, num_points]
    # Valid mask
    valid = (dist1 > min_radius) & (dist2 > min_radius)  # [num_envs, num_points]
    # Find highest index of valid per env (last valid)
    reversed_valid = valid.flip(dims=[1])
    first_max_in_reversed = torch.argmax(reversed_valid.float(), dim=1)
    max_indices = num_points - 1 - first_max_in_reversed  # [num_envs]
    # Check if any valid per env
    has_valid = valid.any(dim=1)  # [num_envs]
    # Compute angles for all points
    angle_rotated = torch.atan2(rotated_joint_vec[:, :, 1], rotated_joint_vec[:, :, 0])  # [num_envs, num_points]
    angle_ft = torch.atan2(ft_joint_vec[:, :, 1], ft_joint_vec[:, :, 0])  # [num_envs, num_points]
    # Base difference
    result = angle_rotated - angle_ft  # [num_envs, num_points]
    # Check x-sign agreement
    same_sign = torch.sign(rotated_joint_vec[:, :, 0]) == torch.sign(ft_joint_vec[:, :, 0])  # [num_envs, num_points]
    # Adjustment for different signs
    pi_val = torch.tensor(3.141592653589793, device=device)
    adjustment = torch.where(same_sign,
                             torch.zeros_like(result),
                             -torch.sign(result) * pi_val)
    result = result + adjustment
    # Wrap to [-π, π]
    result = torch.atan2(torch.sin(result), torch.cos(result))
    # Select result for max index per env
    selected_result = result[torch.arange(num_envs), max_indices]  # [num_envs]
    # Delta: selected if valid, else 0
    delta = torch.where(has_valid, selected_result, torch.zeros_like(selected_result))
    # Apply to current joint value
    new_q = q[:, joint_idx] + delta
    # Clamp to joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)
    return new_q


def range_to_regex(low: int, high: int) -> str:
    """Generates a compact regex pattern matching integer strings from low to high inclusive."""
    if low > high:
        raise ValueError("low > high")
    if low == high:
        return str(low)
    low_str, high_str = str(low), str(high)
    if len(low_str) != len(high_str):
        mid = 10 ** len(low_str) - 1
        return f"(?:{range_to_regex(low, mid)}|{range_to_regex(mid + 1, high)})"
    i = 0
    while i < len(low_str) and low_str[i] == high_str[i]:
        i += 1
    prefix = low_str[:i]
    low_d = int(low_str[i])
    high_d = int(high_str[i])
    if i == len(low_str) - 1:
        return f"{prefix}[{low_d}-{high_d}]"
    res = []
    suffix_len = len(low_str) - i - 1
    low_suffix = int(low_str[i + 1 :])
    high_suffix = int(high_str[i + 1 :])
    suffix_max = 10**suffix_len - 1
    # Low digit part
    if low_suffix == 0:
        res.append(f"{prefix}{low_d}\\d{{{suffix_len}}}")
    else:
        res.append(f"{prefix}{low_d}{range_to_regex(low_suffix, suffix_max)}")
    # Middle digits
    for d in range(low_d + 1, high_d):
        res.append(f"{prefix}{d}\\d{{{suffix_len}}}")
    # High digit part
    if high_suffix == suffix_max:
        if res:
            res[-1] = f"{prefix}{high_d}\\d{{{suffix_len}}}"
    else:
        res.append(f"{prefix}{high_d}{range_to_regex(0, high_suffix)}")
    return f"(?:{'|'.join(res)})"

