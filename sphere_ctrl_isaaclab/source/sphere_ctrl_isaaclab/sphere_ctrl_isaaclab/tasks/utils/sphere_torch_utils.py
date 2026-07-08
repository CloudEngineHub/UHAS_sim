import torch
import numpy as np
import tf.transformations as tf
import json

def torch_compute_deformed_sphere_points_detached_fingers_dynamic_anchors(
    fixed_phi, fixed_offsets, points, base_radius=1.0, device='cpu', epsilon=1e-6
):
    """
    Compute deformed sphere points using linear interpolation based on fixed phi values per sphere.
    Parameters:
    - fixed_phi: [n_envs, n_spheres, n_fixed_points] tensor of fixed phi values per sphere per environment
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
    # Extract dimensions
    n_envs, n_spheres, n_fixed_points = fixed_phi.shape
    # Sort fixed_phi for each sphere
    sorted_phi, sort_idx = torch.sort(fixed_phi, dim=-1)  # [n_envs, n_spheres, n_fixed_points], [n_envs, n_spheres, n_fixed_points]
    # Sort fixed_offsets for each sphere using the sort indices
    sorted_offsets = torch.gather(fixed_offsets, dim=2, index=sort_idx)  # [n_envs, n_spheres, n_fixed_points]
    # Extract phi values from the sample points
    phi_samples = points[:, 1].view(1, 1, -1).expand(n_envs, n_spheres, -1).contiguous()  # [n_envs, n_spheres, n_points]
    # Find interpolation indices for each sphere
    idx = torch.searchsorted(sorted_phi, phi_samples, right=True)  # [n_envs, n_spheres, n_points]
    idx_a = torch.clamp(idx - 1, 0, n_fixed_points - 1)  # [n_envs, n_spheres, n_points]
    idx_b = torch.clamp(idx, 0, n_fixed_points - 1)  # [n_envs, n_spheres, n_points]
    # Gather phi values for interpolation
    phi_a = torch.gather(sorted_phi, dim=2, index=idx_a)  # [n_envs, n_spheres, n_points]
    phi_b = torch.gather(sorted_phi, dim=2, index=idx_b)  # [n_envs, n_spheres, n_points]
    # Gather offset values
    offsets_a = torch.gather(sorted_offsets, dim=2, index=idx_a)  # [n_envs, n_spheres, n_points]
    offsets_b = torch.gather(sorted_offsets, dim=2, index=idx_b)  # [n_envs, n_spheres, n_points]
    # Compute interpolation weights
    delta_phi = phi_b - phi_a  # [n_envs, n_spheres, n_points]
    diff = phi_samples - phi_a  # [n_envs, n_spheres, n_points]
    weight = torch.where(
        delta_phi > 0,
        diff / (delta_phi + epsilon),
        torch.tensor(0.0, device=device)
    )  # [n_envs, n_spheres, n_points]
    weight = torch.clamp(weight, 0, 1)  # [n_envs, n_spheres, n_points]
    # Perform linear interpolation of offsets
    interpolated_offset = offsets_a * (1 - weight) + offsets_b * weight # [n_envs, n_spheres, n_points]
    # Compute the deformed radius
    radius = base_radius * interpolated_offset # [n_envs, n_spheres, n_points]

    return radius

def add_random_T_new(
    num_envs: int,
    gravity_angle_range_deg: float,
    init_T: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    angles_y_deg = torch.rand(num_envs, device=device, dtype=dtype) * (-1 * gravity_angle_range_deg) 
    angles_y = torch.deg2rad(angles_y_deg)
    cy, sy = torch.cos(angles_y), torch.sin(angles_y)


    RotY = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1, -1).clone()
    RotY[:, 0, 0] = cy
    RotY[:, 0, 2] = sy
    RotY[:, 2, 0] = -sy
    RotY[:, 2, 2] = cy

    init_T_batch = init_T.unsqueeze(0).expand(num_envs, -1, -1)
    T = torch.matmul(init_T_batch, RotY)
    return T


def add_random_T(
    num_envs: int,
    gravity_angle_range_deg: float,
    init_T: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    angles_x_deg = torch.rand(num_envs, device=device, dtype=dtype) * (2 * gravity_angle_range_deg) - gravity_angle_range_deg
    angles_y_deg = torch.rand(num_envs, device=device, dtype=dtype) * (2 * gravity_angle_range_deg) - gravity_angle_range_deg
    angles_x = torch.deg2rad(angles_x_deg)
    angles_y = torch.deg2rad(angles_y_deg)

    cx, sx = torch.cos(angles_x), torch.sin(angles_x)
    cy, sy = torch.cos(angles_y), torch.sin(angles_y)

    RotX = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1, -1).clone()
    RotX[:, 1, 1] = cx
    RotX[:, 1, 2] = -sx
    RotX[:, 2, 1] = sx
    RotX[:, 2, 2] = cx

    RotY = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(num_envs, -1, -1).clone()
    RotY[:, 0, 0] = cy
    RotY[:, 0, 2] = sy
    RotY[:, 2, 0] = -sy
    RotY[:, 2, 2] = cy

    R = torch.matmul(RotY, RotX)
    init_T_batch = init_T.unsqueeze(0).expand(num_envs, -1, -1)
    T = torch.matmul(init_T_batch, R)
    return T

def torch_compute_deformed_sphere_points(
    d_planes_theta, d_plane_offsets, d_vectors_phi, d_vector_offsets,
    n_pole_values, s_pole_values, points, base_radius=1.0, device='cpu'
):
    """
    Compute deformed sphere points using PyTorch tensors.

    Parameters:
    - d_planes_theta: [n_driving_planes] tensor
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
    PI = torch.tensor(np.pi, dtype=torch.float, device=device)
    n_spheres = d_plane_offsets.size(1)
    n_driving_planes = d_planes_theta.size(0)
    n_points = points.size(0)

    # Extend driving vectors
    extended_d_vectors_phi = torch.cat([
        torch.tensor([0.0], device=device),
        d_vectors_phi,
        torch.tensor([PI], device=device)
    ]).contiguous()

    # Extend radius offsets
    n_pole_offsets = n_pole_values[None, None, :].expand(n_driving_planes, 1, -1)
    s_pole_offsets = s_pole_values[None, None, :].expand(n_driving_planes, 1, -1)
    extended_d_vector_offsets = torch.cat([n_pole_offsets, d_vector_offsets, s_pole_offsets], dim=1)

    # Extract theta and phi
    theta, phi = points[:, 0], points[:, 1].contiguous()

    # Adjust plane angles
    adjusted_d_planes_theta = (d_planes_theta[:, None] + d_plane_offsets) % (2 * PI)
    extended_theta = torch.cat([
        adjusted_d_planes_theta - 2 * PI,
        adjusted_d_planes_theta,
        adjusted_d_planes_theta + 2 * PI
    ], dim=0)#! Make it per sphere planes
    extended_theta_sorted, sort_idx = torch.sort(extended_theta, dim=0)

    # Extend and sort offsets
    extended_offsets = extended_d_vector_offsets.repeat(3, 1, 1)
    extended_offsets_sorted = torch.gather(
        extended_offsets,
        0,
        sort_idx[:, None, :].expand(-1, extended_offsets.size(1), -1)
    )

    extended_theta_sorted_t = extended_theta_sorted.transpose(0, 1).contiguous()
    theta_expanded = theta[None, :].repeat(n_spheres, 1).contiguous()

    # Perform searchsorted for each sphere
    idx = torch.searchsorted(extended_theta_sorted_t, theta_expanded, right=True) 
    idx = torch.clamp(idx, 1, 3 * n_driving_planes - 1).T # Back to (n_points, n_sphere)

    # Surrounding theta values
    s_indices = torch.arange(n_spheres, device=device)[None, :].expand(n_points, -1)
    theta_a = extended_theta_sorted[idx - 1, s_indices]
    theta_b = extended_theta_sorted[idx, s_indices]
    fraction = (theta[:, None] - theta_a) / (theta_b - theta_a)

    # Phi interpolation
    idx_phi = torch.searchsorted(extended_d_vectors_phi, phi, right=True) - 1 
    idx_phi = torch.clamp(
        idx_phi,
        torch.tensor(0,dtype=torch.int, device=device),
        extended_d_vectors_phi.size(0) - 2
    )
    phi_a = extended_d_vectors_phi[idx_phi]
    phi_b = extended_d_vectors_phi[idx_phi + 1]
    weight_phi = (phi - phi_a) / (phi_b - phi_a)

    # Interpolate offsets
    row_indices_a, row_indices_b = idx - 1, idx
    offsets_a = extended_offsets_sorted[row_indices_a, :, s_indices]
    offsets_b = extended_offsets_sorted[row_indices_b, :, s_indices]
    p_indices = torch.arange(n_points, device=device)[:, None].expand(-1, n_spheres)
    phi_indices = idx_phi[:, None].expand(-1, n_spheres)
    offsets_a_at_phi = (1 - weight_phi[:, None]) * offsets_a[p_indices, s_indices, phi_indices] + \
                       weight_phi[:, None] * offsets_a[p_indices, s_indices, phi_indices + 1]
    offsets_b_at_phi = (1 - weight_phi[:, None]) * offsets_b[p_indices, s_indices, phi_indices] + \
                       weight_phi[:, None] * offsets_b[p_indices, s_indices, phi_indices + 1]

    # Compute coordinates
    radius_offsets = offsets_a_at_phi + fraction * (offsets_b_at_phi - offsets_a_at_phi)
    r = base_radius * (radius_offsets) #! Proportional to radius (no offset)
    x = r * torch.sin(phi)[:, None] * torch.cos(theta)[:, None]
    y = r * torch.sin(phi)[:, None] * torch.sin(theta)[:, None]
    z = r * torch.cos(phi)[:, None]
    xyz = torch.stack([x, y, z], dim=2)

    return xyz

def torch_sample_fibonacci_points_sphere(n_points, device='cpu'):
    """
    Create a list of n_points on a sphere using the Fibonacci Sampling Algorithm.
    
    Parameters:
    - n_points: int, number of points to generate.
    - device: str or torch.device, device to place the tensors on (e.g., 'cpu' or 'cuda').
    
    Returns:
    - points: torch.Tensor, shape (n_points, 2), containing (theta, phi) coordinates.
    """
    indices = torch.arange(n_points, device=device)
    points = torch.zeros((n_points, 2), device=device)
    
    # Polar angle (phi) from 0 to ~π
    phi = torch.acos(1 - 2 * indices.float() / n_points)
    
    # Golden ratio for azimuthal angle (theta)
    golden_ratio = (1 + torch.sqrt(torch.tensor(5.0, device=device))) / 2
    theta = (2 * torch.pi * indices.float() / golden_ratio) % (2 * torch.pi)
    
    # Assign coordinates
    points[:, 0] = theta  # Azimuthal angle
    points[:, 1] = phi    # Polar angle
    
    # Ensure south pole is included
    points[-1] = torch.tensor([torch.pi, 0.0], device=device)
    
    return points

def torch_solve_for_A_joint_quadrant_care_multi(
    joint_tag, q, joint_index_dict, points_h, ft_joint, joint_limits,
    radius, radius_ratio=0.1
):
    """
    Compute updated joint angles for an A joint across all environments based on multiple rotated fingertip sphere positions.
    For each env, selects the first valid pair where dist1 > min_radius, and computes angle.
    If no valid pairs, keeps current q.
    Parameters:
    - joint_tag (str): Name of the A joint
    - q (torch.Tensor): [num_envs, num_joints]
    - joint_index_dict (dict): joint name → column index
    - points_h (torch.Tensor): [num_envs, num_points, 4] homogeneous positions
    - ft_joint (torch.Tensor): [num_envs, num_points, 3] or [num_envs, num_points, 4] reference vectors
    - joint_limits (tuple): (lower_limit, upper_limit)
    - radius (float): Radius used for min_dist calculation
    - radius_ratio (float): Ratio for min_dist (default: 0.1)
    Returns:
    - torch.Tensor: [num_envs] updated joint angles
    """
    num_envs = points_h.size(0)
    num_points = points_h.size(1)
    device = points_h.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits
    # Extract 3D vectors
    points_joint_vec = points_h[:, :, :3]  # [num_envs, num_points, 3]
    ft_joint_vec = ft_joint[:, :, :3]  # [num_envs, num_points, 3]
    # Compute distances from joint origin (xy norm)
    dist1 = torch.norm(points_joint_vec[:, :, :2], dim=-1)  # [num_envs, num_points]
    min_dist = radius * radius_ratio
    # Valid mask
    valid = (dist1 > min_dist)  # [num_envs, num_points]
    # Find first valid index per env
    indices = torch.arange(num_points, device=device).expand(num_envs, num_points)  # [num_envs, num_points]
    masked_indices = torch.where(valid, indices, num_points)  # Invalid set to num_points
    first_indices = torch.min(masked_indices, dim=1)[0]  # [num_envs]
    # Check if any valid per env
    has_valid = first_indices < num_points  # [num_envs]
    # Set invalid env indices to 0 to avoid index errors (not used due to has_valid)
    first_indices = torch.where(has_valid, first_indices, torch.zeros_like(first_indices))
    # Compute angles for all points
    angle_rotated = torch.atan2(points_joint_vec[:, :, 1], points_joint_vec[:, :, 0])  # [num_envs, num_points]
    angle_ft = torch.atan2(ft_joint_vec[:, :, 1], ft_joint_vec[:, :, 0])  # [num_envs, num_points]
    # Base difference
    result = angle_rotated - angle_ft  # [num_envs, num_points]
    # Check x-sign agreement
    same_sign = torch.sign(points_joint_vec[:, :, 0]) == torch.sign(ft_joint_vec[:, :, 0])  # [num_envs, num_points]
    # Adjustment for different signs
    pi_val = torch.tensor(3.141592653589793, device=device)
    adjustment = torch.where(same_sign,
                             torch.zeros_like(result),
                             -torch.sign(result) * pi_val)
    result = result + adjustment
    # Wrap to [-π, π]
    result = torch.atan2(torch.sin(result), torch.cos(result))
    # Select result for first index per env
    selected_result = result[torch.arange(num_envs), first_indices]  # [num_envs]
    # Delta: selected if valid, else 0
    delta = torch.where(has_valid, selected_result, torch.zeros_like(selected_result))
    # Apply to current joint value
    new_q = q[:, joint_idx] + delta
    # Clamp to joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)
    return new_q

def torch_solve_for_A_joints_ohne_interpolation(type_A_joints, anchor_offsets, res, q_A_zero_idx, q_A_max, q_anchor_dist, q_list_dict):
    """
    Solve for type A joints using PyTorch tensors.

    """
    
    joint_values = torch.zeros_like(anchor_offsets, dtype = torch.float, device = res.device)
    offset_idx = torch.round(anchor_offsets / res[None, :] + q_A_zero_idx[None, :] + (q_anchor_dist / res)[None,:])
    offset_idx = torch.clamp(
        offset_idx,
        torch.tensor(0,dtype=torch.int, device = res.device), 
        q_A_max[None, :] - torch.tensor(1,dtype=torch.int, device = res.device)).int()
    # print(offset_idx)
    # print(q_list_dict)
    for i, q_A in enumerate(type_A_joints):
        # q_list = q_list_dict[q_A]
        joint_values[:, i] = q_list_dict[q_A][offset_idx[:, i]]

    return joint_values

def torch_solve_for_A_joint_no_lookup(joint_tag, q, joint_index_dict, T_joint_to_sphere, points_h, ft_joint, joint_limits):
    """
    Compute updated joint angles for an A joint across all environments based on rotated fingertip sphere position.
    Parameters:
    - joint_tag (str): Name of the A joint to solve for.
    - q (torch.Tensor): Joint values at initial positions, shape [num_envs, num_joints].
    - joint_index_dict (dict): Maps joint names to indices in q.
    - points_h (torch.Tensor): Fingertip position to follow.
    - ft_joint_h (torch.Tensor): Static fingertip joint reference vector, shape [4].
    - q_sphere (torch.Tensor): Joint angle in perfect sphere configuration, scalar or shape [].
    - joint_limits (tuple): (lower_limit, upper_limit) for the joint.
    - joint_type_info (dict): Type joint information containing 'ref_dir'.
    Returns:
    - torch.Tensor: Shape [num_envs], updated joint angles for the A joint.
    """
    num_envs = points_h.size(0)
    device = points_h.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits

    # Transform to joint frame
    # rotated_joint = torch.matmul(T_joint_to_sphere, points_h.unsqueeze(-1)).squeeze(-1)  # [num_envs, 4]
    rotated_joint = points_h
    # Compute signed angle between ft_joint and rotated
    # rotated_joint_vec = rotated_joint[:, :3]
    # ft_joint_vec = ft_joint[:3].unsqueeze(0)
    # dot = (rotated_joint_vec * ft_joint_vec).sum(dim=1)  # [num_envs]
    # cross = ft_joint[0] * rotated_joint[:, 1] - ft_joint[1] * rotated_joint[:, 0]  # [num_envs], assuming z=0
    # angle = torch.atan2(cross, dot)  # [num_envs]

    # Compute signed angle between ft_joint and rotated
    rotated_joint_vec = rotated_joint[:, :3]
    ft_joint_vec = ft_joint[:3]
    # Compute angles
    angle_rotated = torch.atan2(rotated_joint_vec[:, 1], rotated_joint_vec[:, 0])
    angle_ft = torch.atan2(ft_joint_vec[1], ft_joint_vec[0])
    result = angle_rotated - angle_ft

    # New q = result + current q
    new_q = result + q[:, joint_idx]

    # Clip to joint limits
    new_q = torch.clamp(new_q, lower_limit, upper_limit)

    return new_q

def torch_solve_for_B_joint(joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask, joint_type_info):
    """
    Compute updated joint angles for a B joint across all spheres based on filtered points and centroids.
    Parameters:
    - joint_tag (str): Name of the B joint to solve for.
    - q: Joint Values at initial positions
    - joint_index_dict (dict): Maps joint names to indices in q.
    - joint_points (torch.Tensor): Shape [n_points, n_spheres, 3], points in the joint frame (x, y, z).
    - joint_limits (tuple): (lower_limit, upper_limit)
    - point_mask (torch.Tensor): Shape [n_points, n_spheres], boolean mask for prefiltered points.
    - joint_type_info (dict): Type joint information of joint.
    Returns:
    - torch.Tensor: Shape [n_spheres], updated joint angles for the B joint.
    """
    # Get the column index of the B joint from the dictionary
    n_spheres = joint_points.size(1)
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits
    # Extract x and y
    x = joint_points[..., 0]  # [n_points, n_spheres]
    y = joint_points[..., 1]  # [n_points, n_spheres]
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    # Additional filter based on ref_dir
    if ref_dir[1] >= 0:
        exclude = (x < 0) & (y < 0)
    else:
        exclude = (x < 0) & (y > 0)
    additional_mask = ~exclude  # [n_points, n_spheres]
    # Effective mask
    effective_mask = point_mask & additional_mask
    # Compute arctan2 angles for all points in the x-y plane
    angles = torch.atan2(joint_points[..., 1], joint_points[..., 0])  # Shape: [n_points, n_spheres]
    # Initialize result tensor for updated angles
    result = torch.zeros(n_spheres, device=device)
    if ref_dir[1] >= 0:
        result = torch.min(angles.masked_fill(~effective_mask, float('inf')), dim=0).values
        default_result = upper_limit.repeat(n_spheres)
    else:
        result = torch.max(angles.masked_fill(~effective_mask, -float('inf')), dim=0).values
        default_result = lower_limit.repeat(n_spheres)
    # Check if there are valid points for each sphere
    has_valid_points = torch.any(effective_mask, dim=0)  # Shape: [n_spheres]
    # Apply default_result when there are no valid points
    result = result + q[:, joint_idx]
    result = torch.where(has_valid_points, result, default_result)
    # Clip the updated angles to stay within joint limits
    new_q = torch.clamp(result, lower_limit, upper_limit)
    return new_q

def torch_solve_for_B_joint_with_eigenvectors(
    joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask,
    joint_type_info, c, init_position, radius, th=0.0
):
    """
    Compute B joint angles with NaN safety: defaults to limit if invalid.
    Uses arc length for point filtering based on initial spherical coordinates.
    """
    n_points, num_envs, _ = joint_points.shape
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits
    
    # Extract y for reference
    y_values = joint_points[..., 1]  # [n_points, num_envs]
    ref_dir = joint_type_info[joint_tag]["ref_dir"]
    
    # Compute reference y per env
    if ref_dir[1] > 0:
        y_ref = torch.min(y_values.masked_fill(~point_mask, float('inf')), dim=0).values  # [num_envs]
        ref_idx = torch.argmin(y_values.masked_fill(~point_mask, float('inf')), dim=0)  # [num_envs]
    else:
        y_ref = torch.max(y_values.masked_fill(~point_mask, -float('inf')), dim=0).values  # [num_envs]
        ref_idx = torch.argmax(y_values.masked_fill(~point_mask, -float('inf')), dim=0)  # [num_envs]
    
    # Get reference theta, phi
    ref_theta = init_position[ref_idx, 0]  # [num_envs]
    ref_phi = init_position[ref_idx, 1]  # [num_envs]
    
    # Expand point theta, phi across envs
    theta_points = init_position[:, 0].unsqueeze(-1).expand(-1, num_envs)  # [n_points, num_envs]
    phi_points = init_position[:, 1].unsqueeze(-1).expand(-1, num_envs)  # [n_points, num_envs]
    
    # Compute central angle using spherical law of cosines
    cos_c_angle = (
        torch.sin(theta_points) * torch.sin(ref_theta.unsqueeze(0)) +
        torch.cos(theta_points) * torch.cos(ref_theta.unsqueeze(0)) * torch.cos(phi_points - ref_phi.unsqueeze(0))
    )
    cos_c_angle = torch.clamp(cos_c_angle, -1.0, 1.0)
    c_angle = torch.acos(cos_c_angle)  # [n_points, num_envs]
    
    # Arc lengths
    arc_lengths = radius * c_angle  # Broadcasts if radius scalar
    
    # Arc length mask
    new_mask = arc_lengths <= c
    final_mask = point_mask & new_mask
    
    # xy points and mask
    points_xy = joint_points[..., :2]  # [n_points, num_envs, 2]
    mask_float = final_mask.float()  # [n_points, num_envs]
    
    # Scatter matrix components
    sum_xx = torch.sum(points_xy[..., 0]**2 * mask_float, dim=0)  # [num_envs]
    sum_yy = torch.sum(points_xy[..., 1]**2 * mask_float, dim=0)  # [num_envs]
    sum_xy = torch.sum(points_xy[..., 0] * points_xy[..., 1] * mask_float, dim=0)  # [num_envs]
    
    # Build scatter matrix S
    S = torch.zeros(num_envs, 2, 2, device=device)
    S[:, 0, 0] = sum_xx
    S[:, 1, 1] = sum_yy
    S[:, 0, 1] = S[:, 1, 0] = sum_xy
    S_det = sum_xx * sum_yy - sum_xy**2
    valid_S = S_det > 1e-8  # [num_envs]
    
    # Compute eigenvectors only where valid
    eigenvalues = torch.zeros(num_envs, 2, device=device)
    eigenvectors = torch.zeros(num_envs, 2, 2, device=device)
    if valid_S.any():
        eigenvalues_valid, eigenvectors_valid = torch.linalg.eigh(S[valid_S])
        eigenvalues[valid_S] = eigenvalues_valid
        eigenvectors[valid_S] = eigenvectors_valid
    
    # Get principal direction
    idx = torch.argmax(eigenvalues, dim=1)  # [num_envs]
    v = eigenvectors[torch.arange(num_envs), :, idx]  # [num_envs, 2]
    
    # Safe mean_xy
    num_valid = torch.sum(mask_float, dim=0).unsqueeze(-1)  # [num_envs, 1]
    mean_xy = torch.zeros(num_envs, 2, device=device)
    has_points = num_valid.squeeze(-1) > 0
    if has_points.any():
        safe_sum = torch.sum(points_xy * mask_float.unsqueeze(-1), dim=0)  # [num_envs, 2]
        mean_xy[has_points] = safe_sum[has_points] / num_valid[has_points].clamp(min=1e-8)
    
    # Flip v to point toward mean_xy
    proj = torch.sum(v * mean_xy, dim=1)  # [num_envs]
    sign_flip = proj < 0
    v = torch.where(sign_flip.unsqueeze(1), -v, v)
    
    # Safe atan2
    v_x, v_y = v[:, 0], v[:, 1]
    angles = torch.atan2(v_y, v_x)  # [num_envs]
    
    # Apply threshold
    if th > 0.0:
        angles = torch.where(torch.abs(angles) < th, torch.zeros_like(angles), angles)
    
    # Final NaN Guard: Use default if invalid
    if ref_dir[1] > 0:
        default_result = upper_limit.repeat(num_envs)
    else:
        default_result = lower_limit.repeat(num_envs)
    v_norm = torch.norm(v, dim=1)
    valid_result = has_points & valid_S & (v_norm > 1e-6)
    angles = q[:, joint_idx] + angles 
    result = torch.where(valid_result, angles, default_result)
    
    # Final update and clamp
    new_q = torch.clamp(result, lower_limit, upper_limit)
    
    # FINAL NaN CLEANUP
    nan_mask = torch.isnan(new_q)
    if nan_mask.any():
        new_q = torch.where(nan_mask, default_result, new_q)
    
    return new_q

def torch_solve_for_B_joint_with_average_angle(
    joint_tag, q, joint_index_dict, joint_points, joint_limits, point_mask,
    joint_type_info, th=0.0
):
    """
    Compute B joint angles by averaging the atan2 angles of all valid filtered points.
    Uses only the provided point_mask (no arc-length filtering).
    Falls back to joint limit if no valid points.
    """
    n_points, num_envs, _ = joint_points.shape
    device = joint_points.device
    joint_idx = joint_index_dict[joint_tag]
    lower_limit, upper_limit = joint_limits

    # Extract y for reference direction (used only for fallback consistency)
    y_values = joint_points[..., 1]  # [n_points, num_envs]
    ref_dir = joint_type_info[joint_tag]["ref_dir"]

    # Compute reference y per env (for fallback direction)
    if ref_dir[1] > 0:
        y_ref = torch.min(y_values.masked_fill(~point_mask, float('inf')), dim=0).values
    else:
        y_ref = torch.max(y_values.masked_fill(~point_mask, -float('inf')), dim=0).values

    # Use only the input point_mask
    final_mask = point_mask
    mask_float = final_mask.float()  # [n_points, num_envs]

    # Count valid points per environment
    num_valid = torch.sum(mask_float, dim=0)  # [num_envs]
    has_points = num_valid > 0

    # Compute individual angles for all points
    points_xy = joint_points[..., :2]  # [n_points, num_envs, 2]
    angles = torch.atan2(points_xy[..., 1], points_xy[..., 0])  # [n_points, num_envs]

    # Average angle per environment (masked)
    sum_angles = torch.sum(angles * mask_float, dim=0)  # [num_envs]
    avg_angles = torch.zeros(num_envs, device=device)
    avg_angles[has_points] = sum_angles[has_points] / num_valid[has_points]

    # Apply small angle threshold (optional)
    if th > 0.0:
        avg_angles = torch.where(torch.abs(avg_angles) < th, torch.zeros_like(avg_angles), avg_angles)

    # Final guard: use joint limit if no valid points
    if ref_dir[1] > 0:
        default_result = upper_limit.repeat(num_envs)
    else:
        default_result = lower_limit.repeat(num_envs)

    avg_angles = q[:, joint_idx] + avg_angles
    result = torch.where(has_points, avg_angles, default_result)

    # Update joint angle and clamp
    new_q = torch.clamp(result, lower_limit, upper_limit)

    # Final NaN cleanup
    nan_mask = torch.isnan(new_q)
    if nan_mask.any():
        new_q = torch.where(nan_mask, default_result, new_q)

    return new_q

def get_noise(value, std):
    noise = torch.randn_like(value) * std
    return noise

def get_uniform_noise(value, amplitude):
    """Generate uniform noise in range [-amplitude, +amplitude] with the same shape as `value`."""
    # Uniform random values in [-amplitude, +amplitude]
    noise = (2 * torch.rand_like(value) - 1) * amplitude
    return noise

def get_debug_finger_actions(env_time, actions, max_fingers, n_vectors, perfect_sphere_value, device):
    # Initialize output tensors
    # lateral_actions: [num_envs, max_fingers] for lateral finger movements
    # vector_actions: [num_envs, max_fingers, n_vectors] for vector components
    num_envs = actions.size(0)
    lateral_actions = torch.zeros((num_envs, max_fingers), dtype=torch.float, device=device)
    vector_actions = torch.ones((num_envs, max_fingers, n_vectors), dtype=torch.float, device=device)

    # Define sequence lengths
    lateral_sequence_l = 4  # Duration for lateral actions: [0.0, 1.0, -1.0, 0.0]
    vector_sequence_l = n_vectors  # Duration for vector updates (one vector per step)
    sequence_l = lateral_sequence_l + vector_sequence_l  # Total length of one phase
    total_cycle = sequence_l * 3  # Three phases (1.0, 0.5, -1.0)

    # Discretize env_time to advance cycles at integer seconds
    env_time = torch.floor(env_time)

    # Compute finger index and sequence timing
    finger_idx = (env_time % (total_cycle * max_fingers) // total_cycle).to(dtype=torch.int, device=device)  # Shape: []
    r_time = env_time % total_cycle  # Time within the total cycle
    curr_sequence = r_time // sequence_l  # Current phase (0, 1, or 2)
    rs_time = r_time % sequence_l  # Time within the current phase

    # Convert to Python integers for indexing
    finger_idx = int(finger_idx.item())
    curr_sequence = int(curr_sequence.item())
    rs_time = int(rs_time.item())

    # Set lateral actions and vector actions for each environment
    for env_idx in range(num_envs):
        # Set non-active fingers: lateral_actions = 0.0, vector_actions = 1.0
        for f_idx in range(max_fingers):
            if f_idx != finger_idx:
                lateral_actions[env_idx, f_idx] = 0.0
                vector_actions[env_idx, f_idx, :] = 2.0

        # Handle active finger
        if rs_time < lateral_sequence_l:
            # Lateral action sequence: [0.0, 1.0, -1.0, 0.0]
            if rs_time == 0:
                lateral_actions[env_idx, finger_idx] = 0.0
            elif rs_time == 1:
                lateral_actions[env_idx, finger_idx] = 1.0
            elif rs_time == 2:
                lateral_actions[env_idx, finger_idx] = -1.0
            elif rs_time == 3:
                lateral_actions[env_idx, finger_idx] = 0.0
        else:
            # During vector update phase, lateral actions remain 0.0
            lateral_actions[env_idx, finger_idx] = 0.0

        # Vector actions based on phase and rs_time
        v_idx = rs_time - lateral_sequence_l  # Index of vector to update (if in vector phase)
        if curr_sequence == 0:
            # Phase 1: Vectors start at 1.0, transition to 0.5 one at a time
            if rs_time >= lateral_sequence_l:
                # Set vectors up to v_idx to 0.5, others remain 1.0
                if v_idx >= 0 and v_idx < n_vectors:
                    vector_actions[env_idx, finger_idx, :v_idx + 1] = perfect_sphere_value
            else:
                # During lateral phase, all vectors remain 1.0
                vector_actions[env_idx, finger_idx, :] = 1.0
        elif curr_sequence == 1:
            # Phase 2: Vectors start at 0.5, transition to -1.0 one at a time
            vector_actions[env_idx, finger_idx, :] = perfect_sphere_value  # Base value
            if rs_time >= lateral_sequence_l:
                if v_idx >= 0 and v_idx < n_vectors:
                    vector_actions[env_idx, finger_idx, :v_idx + 1] = -1.0
        elif curr_sequence == 2:
            # Phase 3: Vectors start at -1.0, transition to 1.0 one at a time
            vector_actions[env_idx, finger_idx, :] = -1.0  # Base value
            if rs_time >= lateral_sequence_l:
                if v_idx >= 0 and v_idx < n_vectors:
                    vector_actions[env_idx, finger_idx, :v_idx + 1] = 1.0

    # Flatten vector_actions and concatenate with lateral_actions
    vector_actions_flat = vector_actions.flatten(start_dim=1)  # Shape: [num_envs, max_fingers * n_vectors]
    actions = torch.cat((lateral_actions, vector_actions_flat), dim=1)  # Shape: [num_envs, max_fingers + max_fingers * n_vectors]

    return actions

import json
from pathlib import Path


def import_reference_sphere(
    device: str | torch.device = "cpu",
    json_filename: str = "reference_sphere.json",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the reference sphere data (r + xyz_sphere_points) from a JSON file
    located in the SAME folder as this Python script.

    This works reliably even if you import and call this function from another script
    in a completely different directory.

    The JSON must contain:
        - "r": radial distances (first environment only)
        - "xyz_sphere_points": unit sphere direction vectors

    Args:
        device: Device to place the returned tensors on (e.g. "cpu", "cuda", torch.device("cuda:0"))
        json_filename: Name of the JSON file (default: "reference_sphere.json").
                       You can also pass an absolute path if you ever need to override.

    Returns:
        ref_xyz: torch.Tensor of shape (num_points, 3) — unit sphere directions
        ref_r:   torch.Tensor of shape matching the saved r (usually (num_points,) or (num_fingers, num_points))
    """
    # Resolve path relative to the folder where THIS file lives
    script_dir = Path(__file__).parent.absolute()
    full_path = script_dir / json_filename

    if not full_path.exists():
        raise FileNotFoundError(
            f"Reference sphere JSON not found at:\n"
            f"   {full_path}\n"
            f"Please make sure reference_sphere.json is in the same folder as this script."
        )

    with open(full_path, "r") as f:
        data = json.load(f)

    # Load the two required keys
    r_list = data.get("r")
    xyz_list = data.get("xyz_sphere_points")

    if r_list is None or xyz_list is None:
        # Backward compatibility with old JSON format (only "points")
        raise ValueError(
            f"JSON file {full_path} must contain either both 'r' and 'xyz_sphere_points' "
            f"or the old 'points' key."
        )

    # Convert lists → tensors on the requested device
    ref_r = r_list
    ref_xyz = xyz_list

    print(f"✅ Loaded reference sphere data → "
          f"on device '{device}' (from {full_path.name})")

    return {"xyz": ref_xyz, "r": ref_r}