import einops
import torch

from src.forces.anm import ANM
from src.forces.force_fields import HinsenForceField


def get_anm_hessian(X: torch.Tensor, ff: HinsenForceField) -> torch.Tensor:
    """Get the Hessian matrix from the ANM model.
    Args:
        X: torch.Tensor, shape (n_nodes, dim)
        ff: HinsenForceField object
    Returns:
        anm: ANM object that stores Hessians inside
    """

    anm = ANM(X, ff, masses=False, device=X.device)
    anm.hessian
    return anm

def get_anm_eigens(anm) -> torch.Tensor:
    """Get the eigenvalues and eigenvectors of the ANM Hessian matrix.
    Args:
        anm: ANM object that stores Hessian inside
    """
    
    origvals, origvecs = anm.eigen()

    return origvals.type(torch.float), origvecs.type(torch.float)


def get_displacement_vecs(eigvec: torch.tensor, node_inds: list):
    """Get the displacement vectors of the eigenvectors for the query nodes.
    Args:
        eigvecs: torch.Tensor, shape (3*n_nodes)
        node_inds: list of ints, indices of the target/conditioned nodes
    Returns:
        disp_vecs: torch.Tensor, shape (n_nodes, 3): displacement vectors in this normal mode"""

    num_nodes = eigvec.shape[0] // 3

    all_mode_disp = eigvec.view(num_nodes, 3)
    mode_disp = all_mode_disp[node_inds, :]

    return mode_disp

def get_displacement_cosines(displacements):
    """Get the cosines between the oscillation directions between neighbouring nodes for the given fragment of the eigenvector.
    Args:
        displacements: torch.Tensor, shape (n_nodes, 3)
    Returns:
        cosines: torch.Tensor, shape (n_nodes * n_nodes)"""

    rows_norms = torch.norm(displacements, dim=-1)
    rows_norms = torch.where(
        rows_norms == 0, torch.tensor([1], device=rows_norms.device), rows_norms
    )  # avoid division by zero in the calculation of cosines
    normalised_disp = torch.einsum("nd,n->nd", displacements, 1 / rows_norms)

    cosines = torch.einsum("ij,kj->ik", normalised_disp, normalised_disp)
    cosines = einops.rearrange(cosines, "i j -> (i j)")

    return cosines

def subtract_genie_com(noise: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Subtracts center of mass (used for the the conditional noise)
    Args:
        noise: [max_n_res, coords]
    """

    real_noise = noise[0:num_nodes, :] # select only the real nodes (not the padded ones)

    real_noise_mean = torch.mean(real_noise, dim=0) # calculate the mean of the real nodes
    real_noise = real_noise - real_noise_mean[None,:]
    noise = torch.cat([real_noise, torch.zeros(noise.shape[0] - num_nodes, 3).to(noise.device)], dim=0)

    return noise