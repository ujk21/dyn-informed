import numpy as np
import torch

K_B = 1.380649e-23
N_A = 6.02214076e23


## NMA functions for ANMs
def eigen(enm):
    """
    Compute the Eigenvalues and Eigenvectors of the
    *Hessian* matrix for ANMs.
    Parameters
    ----------
    enm : ANM
        Elastic network model; ANM
        object.
    Returns
    -------
    eig_values : tensor, shape=(k,), dtype=float
        Eigenvalues of the *Hessian* matrix
        in ascending order.
    eig_vectors : tensor, shape=(k,n), dtype=float
        Eigenvectors of the *Hessian* matrix.
        ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
    """

    from .anm import ANM

    if isinstance(enm, ANM):
        mech_matrix = enm.hessian
    else:
        raise ValueError("Instance of ANM class expected.")

    eig_values, eig_vectors = torch.linalg.eigh(mech_matrix)

    return eig_values, eig_vectors


def frequencies(enm):
    """
    Computes the frequency associated with each mode.
    The modes corresponding to rigid-body translations/rotations are
    omitted in the return value.
    The returned units are arbitrary and should only be compared
    relative to each other.

    Parameters
    ----------
    enm : ANM or GNM
        Elastic network model; an instance of either an GNM or ANM
        object.
    Returns
    -------
    freq : tensor, shape=(n,), dtype=float
        The frequency in ascending order of the associated modes'
        Eigenvalues.
    """

    from .anm import ANM

    if isinstance(enm, ANM):
        ntriv_modes = 6
    else:
        raise ValueError("Instance of ANM class expected.")

    eig_values, _ = eigen(enm)


    eig_values[0:ntriv_modes] = torch.abs(eig_values[0:ntriv_modes])

    freq = 1 / (2 * torch.pi) * torch.sqrt(eig_values)

    return freq


def normal_mode(anm, index, amplitude, frames, movement="sine"):
    """
    Create displacements for a trajectory depicting the given normal
    mode for ANMs.
    Parameters
    ----------
    anm : ANM
        Instance of ANM object.
    index : int
        The index of the oscillation.
        The index refers to the Eigenvalues obtained from
        :meth:`eigen()`:
        Increasing indices refer to oscillations with increasing
        frequency.
        The first 6 modes represent tigid body movements
        (rotations and translations).
    amplitude : int
        The oscillation amplitude is scaled so that the maximum
        value for an atom is the given value.
    frames : int
        The number of frames (models) per oscillation.
    movement : {'sinusoidal', 'triangle'}
        Defines how to depict the oscillation.
        If set to ``'sine'`` the atom movement is sinusoidal.
        If set to ``'triangle'`` the atom movement is linear with
        *sharp* amplitude.

    Returns
    -------
    displacement : ndarray, shape=(m,n,3), dtype=float
        Atom displacements that depict a single oscillation.
        *m* is the number of frames.
    """
    from .anm import ANM

    if not isinstance(anm, ANM):
        raise ValueError("Instance of ANM class expected.")
    else:
        _, eig_vectors = eigen(anm)
        # Extract vectors for given mode and reshape to (n,3) array
        mode_vectors = eig_vectors[index].reshape((-1, 3))
        # Rescale, so that the largest vector has the length 'amplitude'
        vector_lenghts = np.sqrt(np.sum(mode_vectors**2, axis=-1))
        scale = amplitude / np.max(vector_lenghts)
        mode_vectors *= scale

        time = np.linspace(0, 1, frames, endpoint=False)
        if movement == "sine":
            normed_disp = np.sin(time * 2 * np.pi)
        elif movement == "triangle":
            normed_disp = 2 * np.abs(2 * (time - np.floor(time + 0.5))) - 1
        else:
            raise ValueError(f"Movement '{movement}' is unknown")
        disp = normed_disp[:, np.newaxis, np.newaxis] * mode_vectors

        return disp
