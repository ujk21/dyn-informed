from typing import Tuple
import torch

__all__ = ["differentiable_kabsch", "rototranslate", "random_rotation_matrix"]


def differentiable_kabsch(
    p1: torch.Tensor, p2: torch.Tensor, step: int, max_iter: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the differentiable Kabsch algorithm to align two sets of points in 3D space.

    Args:
        p1 (torch.Tensor): A tensor of shape (N, 3) representing the first set of points.
        p2 (torch.Tensor): A tensor of shape (N, 3) representing the second set of points.
        max_iter (int): Maximum number of iterations to perform if SVD is numerically unstable.

    Returns:
        A tuple containing two tensors:
        - rot_mat (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix that aligns p2 with p1.
        - trans_vec (torch.Tensor): A tensor of shape (3,) representing the translation vector that aligns p2 with p1.
    """
    num_points, num_coords = p1.shape
    assert p1.shape == p2.shape, "p1 and p2 must have the same shape."

    # Compute centers of mass
    p1_com = p1.mean(dim=0)  # (num_coords)
    p2_com = p2.mean(dim=0)  # (num_coords)

    # Compute (spatial) covariance matrix
    cov = torch.einsum("ni,nj->ij", p1 - p1_com, p2 - p2_com)  # (num_coords, num_coords)
    if torch.isnan(cov).any():
        print("you got a problem")
    assert not torch.isnan(cov).any(), f"Covariance matrix is NaN. Step: {step}"

    # Compute SVD
    U, S, Vt = torch.linalg.svd(
        cov, full_matrices=True
    )  # (num_coords, num_coords), (num_coords), (num_coords, num_coords)

    # Ensure numerical stability:
    #  Sometimes SVD is poorly conditioned and has zero or degenerate singular values for a square covariance matrix.
    #  This leads to unstable gradients in the backward pass. To avoid this we add a small amount of noise to the
    #  covariance matrix and recompute the SVD until all singular values are non-zero and non-degenerate.
    _has_zero_singular_value = lambda S: S.min() < 1e-3
    # Generate all (sigma_i)^2 - (sigma_j)^2 and check if any are close to zero
    _has_degenerate_singular_values = (
        lambda S: (S**2 - (S**2).view(num_coords, 1) + torch.eye(num_coords, device=cov.device))
        .abs()
        .min()
        < 1e-2
    )
    num_it = 0
    while _has_zero_singular_value(S) or _has_degenerate_singular_values(S):
        # Add a small amount of noise to the covariance matrix and recompute SVD
        noise_scale = (
            cov.abs().max() * 5e-2
        )  # set noise scale to be 5% of the largest value in the covariance matrix

        cov = cov + (torch.rand(3, device=cov.device) * noise_scale).diag()
        U, S, Vt = torch.linalg.svd(
            cov, full_matrices=True
        )  # (num_coords, num_coords), (num_coords), (num_coords, num_coords)

        num_it += 1

        if num_it > max_iter:
            raise ValueError("SVD is consistently numerically unstable.")

    # Compute rotation matrix
    flip_mat = torch.tensor(
        [1, 1, cov.det().sign()], device=cov.device
    ).diag()  # (num_coords, num_coords)
    rot_mat = U @ flip_mat @ Vt  # (num_coords, num_coords)
    trans_vec = p1_com - (rot_mat @ p2_com)  # (num_coords)

    return rot_mat, trans_vec


def rototranslate(
    p: torch.Tensor,
    rot_mat: torch.Tensor,
    trans_vec: torch.Tensor,
    inverse: bool = False,
    validate: bool = False,
) -> torch.Tensor:
    """
    Apply rototranslation to a set of 3D points.
        p' = R @ p + t
    (First rotate, then translate.)

    Args:
        p (torch.Tensor): A tensor of shape (N, 3) representing the set of points.
        rot_mat (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.
        trans_vec (torch.Tensor): A tensor of shape (3,) representing the translation vector.
        validate (bool): Whether to validate the input.

    Returns:
        A tensor of shape (N, 3) representing the set of points after rototranslation.
    """
    if validate:
        num_points, num_coords = p.shape
        assert rot_mat.shape == (num_coords, num_coords)
        assert trans_vec.shape == (num_coords,)
        assert torch.allclose(R @ R.T, torch.eye(num_coords), atol=1e-3, rtol=1e-3)

    if inverse:
        return (p - trans_vec) @ rot_mat

    return p @ rot_mat.T + trans_vec


def random_rotation_matrix(validate: bool = False, **tensor_kwargs) -> torch.Tensor:
    """
    Generates a random (3,3) rotation matrix.

    Args:
        tensor_kwargs: Keyword arguments to pass to the tensor constructor. E.g. `device`, `dtype`.

    Returns:
        A tensor of shape (3, 3) representing the rotation matrix.
    """

    # Generate a random quaternion
    q = torch.rand(4, **tensor_kwargs)
    q /= torch.linalg.norm(q)

    # Compute the rotation matrix from the quaternion
    rot_mat = torch.Tensor(
        [
            [
                1 - 2 * q[2] ** 2 - 2 * q[3] ** 2,
                2 * q[1] * q[2] - 2 * q[0] * q[3],
                2 * q[1] * q[3] + 2 * q[0] * q[2],
            ],
            [
                2 * q[1] * q[2] + 2 * q[0] * q[3],
                1 - 2 * q[1] ** 2 - 2 * q[3] ** 2,
                2 * q[2] * q[3] - 2 * q[0] * q[1],
            ],
            [
                2 * q[1] * q[3] - 2 * q[0] * q[2],
                2 * q[2] * q[3] + 2 * q[0] * q[1],
                1 - 2 * q[1] ** 2 - 2 * q[2] ** 2,
            ],
        ],
        **tensor_kwargs,
    )

    if validate:
        assert torch.allclose(
            rot_mat @ rot_mat.T, torch.eye(3), atol=1e-5, rtol=1e-5
        ), "Not a rotation matrix."

    return rot_mat


def align_and_compute_rmsd(motif_pos, motif_pos_sample, step):
    rot_mat, trans_vec = differentiable_kabsch(motif_pos, motif_pos_sample, step)
    aligned_motif_pos_sample = rototranslate(motif_pos_sample, rot_mat, trans_vec)
    rmsd = torch.linalg.norm(motif_pos - aligned_motif_pos_sample, dim=1).mean()

    return rmsd
