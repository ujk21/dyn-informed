import torch

import src.forces.nma as nma
from src.forces.interactions import compute_hessian

K_B = 1.380649e-23
N_A = 6.02214076e23


class ANM:
    """
    This class represents an *Anisotropic Network Model*.
    Parameters
    ----------
    atoms : tensor, shape=(n,3), dtype=float
        Atom coordinates that are part of the model.
        It usually contains only CA atoms.
    force_field : ForceField, natoms=n
        The :class:`ForceField` that defines the force constants between
        the given `atoms`.
    masses : bool or ndarray, shape=(n,), dtype=float, optional
        If an array is given, the Hessian is weighted with the inverse
        square root of the given masses.
        If set to true, these masses are automatically inferred from the
        ``res_name`` annotation of `atoms`, instead.
        This requires `atoms` to be an :class:`AtomArray`.
        By default no mass-weighting is applied.
    use_cell_list : bool, optional
        If true, a *cell list* is used to find atoms within cutoff
        distance instead of checking all pairwise atom distances.
        This significantly increases the performance for large number of
        atoms, but is slower for very small systems.
        If the `force_field` does not provide a cutoff, no cell list is
        used regardless.
    Attributes
    ----------
    hessian : tensor, shape=(n*3,n*3), dtype=float
        The *Hessian* matrix for this model.
        Each dimension is partitioned in the form
        ``[x1, y1, z1, ... xn, yn, zn]``.
        This is not a copy: Create a copy before modifying this matrix.
    covariance : tensor, shape=(n*3,n*3), dtype=float
        The covariance matrix for this model, i.e. the inverted
        *Hessian*.
        This is not a copy: Create a copy before modifying this matrix.
    masses : tensor, shape=(n,), dtype=float
        The mass for each atom, `None` if no mass weighting is applied.
    """

    def __init__(self, atoms, force_field, masses=None, device="cuda;", use_cell_list=False):
        self._coord = atoms
        self._ff = force_field
        self._use_cell_list = use_cell_list
        self.device = device

        if masses is None or masses is False:
            self._masses = None
        elif masses is True:
            raise Exception("Automatic mass inference not implemented for simplified proteins")
        else:
            # if len(masses) != atoms.array_length():
            if len(masses) != atoms.shape[0]:
                raise IndexError(f"{len(masses)} masses for " f"{atoms.shape[0]} atoms given")
            if torch.any(masses == 0):
                raise ValueError("Masses must not be 0")
            self._masses = torch.tensor(masses, dtype=float)

        if self._masses is not None:
            mass_weights = 1 / torch.sqrt(self._masses)

            mass_weights = torch.repeat(mass_weights, 3)
            self._mass_weight_matrix = torch.outer(mass_weights, mass_weights)
        else:
            self._mass_weight_matrix = None

        self._hessian = None
        self._covariance = None

    @property
    def masses(self):
        return self._masses

    @property
    def hessian(self):
        if self._hessian is None:
            if self._covariance is None:
                self._hessian, _ = compute_hessian(
                    self._coord, self._ff, device=self.device, use_cell_list=False
                )
                if self._mass_weight_matrix is not None:
                    self._hessian *= self._mass_weight_matrix
            else:
                self._hessian = torch.linalg.pinv(self._covariance, hermitian=True, rcond=1e-6)
        return self._hessian

    @hessian.setter
    def hessian(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._hessian = value
        # Invalidate dependent values
        self._covariance = None

    @property
    def covariance(self):
        if self._covariance is None:
            self._covariance = torch.linalg.pinv(self.hessian, hermitian=True, rcond=1e-6)

        return self._covariance

    @covariance.setter
    def covariance(self, value):
        if value.shape != (len(self._coord) * 3, len(self._coord) * 3):
            raise IndexError(
                f"Expected shape "
                f"{(len(self._coord) * 3, len(self._coord) * 3)}, "
                f"got {value.shape}"
            )
        self._covariance = value
        # Invalidate dependent values
        self._hessian = None

    def eigen(self):
        """
        Compute the Eigenvalues and Eigenvectors of the
        *Hessian* matrix.
        The first six Eigenvalues/Eigenvectors correspond to
        trivial modes (translations/rotations) and are usually omitted
        in normal mode analysis.

        Returns
        -------
        eig_values : ndarray, shape=(k,), dtype=float
            Eigenvalues of the *Hessian* matrix in ascending order.
        eig_vectors : ndarray, shape=(k,n), dtype=float
            Eigenvectors of the *Hessian* matrix.
            ``eig_values[i]`` corresponds to ``eig_vectors[i]``.
        """
        return nma.eigen(self)

    def normal_mode(self, index, amplitude, frames, movement="sine"):
        """
        Create displacements for a trajectory depicting the given normal
        mode.
        This is especially useful for molecular animations of the chosen
        oscillation mode.
        Note, that the first six modes correspond to rigid-body translations/
        rotations and are usually omitted in normal mode analysis.
        Parameters
        ----------
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
        return nma.normal_mode(self, index, amplitude, frames, movement)

    def frequencies(self):
        """
        Computes the frequency associated with each mode.
        The first six modes correspond to rigid-body translations/
        rotations and are omitted in the return value.
        The returned units are arbitrary and should only be compared
        relative to each other.
        Returns
        -------
        freq : ndarray, shape=(n,), dtype=float
            The frequency in ascending order of the associated modes'
            Eigenvalues.
        """
        return nma.frequencies(self)
