import biotite.structure as struc
import einops
import torch


def compute_hessian(coord, force_field, device, use_cell_list=False):
    """
    Compute the *Hessian* matrix for atoms with given coordinates and
    the chosen force field.
    Parameters
    ----------
    coord : tensor, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.
    Returns
    -------
    hessian : tensor, shape=(n*3,n*3), dtype=float
        The computed *Hessian* matrix.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
    pairs : tensor, shape=(k,2), dtype=int
        Indices for interacting atoms, i.e. atoms within
        `cutoff_distance`.
    """
    # Convert into higher precision to avert numerical issues in
    # pseudoinverse calculation
    coord = coord.to(torch.float64)
    pairs, disp, sq_dist = _prepare_values_for_interaction_matrix(
        coord, force_field, device, use_cell_list
    )


    hessian = torch.zeros((len(coord), len(coord), 3, 3), dtype=torch.float64, device=device)
    force_constants = force_field.force_constant(pairs[:, 0], pairs[:, 1], sq_dist)

    hessian[pairs[:, 0], pairs[:, 1]] = (
        -force_constants.view(-1, 1, 1)
        / sq_dist.view(-1, 1, 1)
        * disp.view(-1, 3, 1)
        * disp.view(-1, 1, 3)
    )
    # Set values for main diagonal
    indices = torch.arange(len(coord), device=device)
    hessian[indices, indices] = -torch.sum(hessian, dim=0)


    hessian = einops.rearrange(hessian, "a b c d -> (a c) (b d)")
    hessian = hessian.reshape(len(coord) * 3, len(coord) * 3)

    return hessian, pairs


def _prepare_values_for_interaction_matrix(coord, force_field, device, use_cell_list):
    """
    Check input values and calculate common intermediate values for
    :func:`compute_kirchhoff()` and :func:`compute_hessian()`.
    Parameters
    ----------
    coord : ndarray, shape=(n,3), dtype=float
        The coordinates.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of a brute-force approach.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
    Returns
    -------
    pairs : ndarray, shape=(k,2), dtype=int
        Indices for interacting atoms, i.e. atoms within
        `cutoff_distance`.
    disp : ndarray, shape=(k,3), dtype=float
        The displacement vector for the atom `pairs`.
    sq_dist : ndarray, shape=(k,3), dtype=float
        The squared distance for the atom `pairs`.
    """
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"Expected coordinates with shape (n,3), got {coord.shape}")
    if force_field.natoms is not None and len(coord) != force_field.natoms:
        raise ValueError(
            f"Got coordinates for {len(coord)} atoms, "
            f"but forcefield was built for {force_field.natoms} atoms"
        )

    # Find interacting atoms within cutoff distance
    cutoff_distance = force_field.cutoff_distance
    if cutoff_distance is None:
        # Include all possible interactions, except an atom with itself
        adj_matrix = torch.ones((len(coord), len(coord)), dtype=bool, device=device)
    elif use_cell_list:
        cell_list = struc.CellList(
            coord,
            cutoff_distance,
        )
        adj_matrix = cell_list.create_adjacency_matrix(cutoff_distance)
    else:


        disp_matrix = torch.cdist(coord.view(1, -1, 3), coord.view(-1, 1, 3), p=2)

        sq_dist_matrix = disp_matrix**2
        adj_matrix = sq_dist_matrix <= cutoff_distance**2

    # Remove interactions of atoms with themselves
    adj_matrix = adj_matrix.squeeze(-1)
    adj_matrix = adj_matrix.fill_diagonal_(False)
    _patch_adjacency_matrix(
        adj_matrix,
        force_field.contact_shutdown,
        force_field.contact_pair_off,
        force_field.contact_pair_on,
    )

    # Convert matrix to indices where interaction exists
    atom_i, atom_j = torch.where(adj_matrix)

    pairs = torch.cat([atom_i.unsqueeze(-1), atom_j.unsqueeze(-1)], dim=-1)
    disp = coord[pairs[:, 0]] - coord[pairs[:, 1]]

    # Get displacement vector for ANMs
    # and squared distances for distance-dependent force fields
    if cutoff_distance is None or use_cell_list:
        disp = struc.index_displacement(coord, pairs)
        sq_dist = torch.sum(disp * disp, axis=-1)
    else:
        sq_dist = sq_dist_matrix[pairs[:, 0], pairs[:, 1]]

    return pairs, disp, sq_dist


def _patch_adjacency_matrix(matrix, contact_shutdown, contact_pair_off, contact_pair_on):
    """
    Apply contacts that are artificially switched off/on to an
    adjacency matrix.
    The matrix is modified in-place.
    """
    if contact_shutdown is not None:
        matrix[:, contact_shutdown] = False
        matrix[contact_shutdown, :] = False
    if contact_pair_off is not None:
        atom_i, atom_j = contact_pair_off.T
        matrix[atom_i, atom_j] = False
        matrix[atom_j, atom_i] = False
    if contact_pair_on is not None:
        atom_i, atom_j = contact_pair_on.T
        if (atom_i == atom_j).any():
            raise ValueError("Cannot turn on interaction of an atom with itself")
        matrix[atom_i, atom_j] = True
        matrix[atom_j, atom_i] = True
