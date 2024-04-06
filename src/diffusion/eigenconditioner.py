import torch
from torch_geometric.data import Data

from src.diffusion.conditioner import Conditioner
from src.utils.nma_diffusion_utils import *
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.affine_utils import T

class GenieEigenconditioner(Conditioner):
	def __init__(self, predict_noise_func, alphas_schedule, alphabars_schedule, ff_cutoff, device):
		super().__init__()

		self._pred_noise = predict_noise_func
		self.loss_func = torch.nn.functional.l1_loss
		self.alphas_schedule = alphas_schedule
		self.alphabars_schedule = alphabars_schedule
		self.n_steps = len(alphas_schedule)
		self.ff_cutoff = ff_cutoff
		self.device = device

		self.calc_condition = self._apply_cond_disp
		
		self.ff = HinsenForceField(cutoff_distance=self.ff_cutoff)

	def set_monitor(self):
		"""Quantites to monitor in the generation process."""

		self.monitor_cosines = []
		self.monitor_amplitudes = []
		self.monitor_total = []

	def set_condition(
		self,
		target_disp: torch.Tensor = None,
		cond_node_inds: torch.Tensor = None,
		gs: float = None,
		cond_frac: float = None,
		):
		"""Set the condition of the diffusion process.
		Args:
		cond_frac (float): fraction of the last timesteps when we apply the condition
		target_disp (torch.Tensor): Target displacements of shape (n, 3).
		gs (float): const factor for the guidance scale for the bayesian method
		"""
		self.cond_frac = cond_frac
		self.gs = gs
		self.cond_index = 6  # lowest non trivial normal mode

		self.target_displacements = target_disp

		self.cond_node_inds = cond_node_inds
		self.target_amplitudes = torch.linalg.norm(self.target_displacements, dim=-1)
		self.target_amplitudes = torch.nn.functional.normalize(self.target_amplitudes, p=2, dim=-1).to(self.device)
		self.target_cosines = get_displacement_cosines(self.target_displacements).to(self.device)

	def _temp_eigs(self, new_vecs):
		"""Find displacements, cosines and amplitudes for the conditioned nodes in the self.cond_index normal mode.
		Args:
			new_vecs (torch.Tensor): Eigenvectors of the Hessian matrix of shape (3*num_nodes, 3*num_nodes).
		Returns:
			temp_disp (torch.Tensor): Displacement vectors of shape (num_nodes, 3).
			temp_cosines (torch.Tensor): Cosines of shape (num_target_nodes^2).
			temp_amplitudes (torch.Tensor): Normalised (such that their L2 norm is 1) amplitudes of shape (num_target_nodes).
		"""
		temp_eigvec = new_vecs[:, self.cond_index]
		temp_disp = get_displacement_vecs(temp_eigvec, self.cond_node_inds)  # (n_target_nodes, 3)
		temp_cosines = get_displacement_cosines(temp_disp)  # (n_target_nodes^2)

		temp_amplitudes = torch.linalg.norm(temp_disp, dim=-1)
		temp_amplitudes = torch.nn.functional.normalize(temp_amplitudes, p=2, dim=-1)

		return temp_disp, temp_cosines, temp_amplitudes

	def _denoise_positions(self, pos, pred, step):
		return self._total_denoise(pos, pred, step)
	
	def gs_time_scaling(self, a):
		"Time scaling to modify guidance scale. Override if needed."
		return super().gs_time_scaling(a)
		
	def _compute_cond_loss(self, temp_cosines: torch.Tensor, temp_amplitudes: torch.Tensor):
		loss_angles = self.loss_func(temp_cosines, self.target_cosines)
		loss_amplitudes = self.loss_func(temp_amplitudes, self.target_amplitudes)

		graph_loss = loss_angles + 2*loss_amplitudes

		self.monitor_cosines.append(loss_angles.item())
		self.monitor_amplitudes.append(loss_amplitudes.item())
		self.monitor_total.append(loss_angles.item() + loss_amplitudes.item())
		return graph_loss

	def _apply_cond_disp(self, sample: Data, step: int):

		a = self.alphas_schedule[step]
		mask = sample.mask

		if self._conditioning_started(step) == False:
			return torch.zeros_like(sample.pos, device=self.device)
        
		with torch.enable_grad():

			sample_grad = self.initialize_sample_grad(sample)

			trans = sample_grad.pos.unsqueeze(0)
			rots_mean = compute_frenet_frames(trans, mask)
			sample_grad.frames = T(trans=trans, rots=rots_mean)
			new_pred = self._pred_noise(sample_grad, torch.tensor([step] * mask.shape[0], dtype=torch.long, device=self.device), mask)
			temp_disp, temp_cosines, temp_amplitudes = self._get_expected_dynamics(sample_grad.pos, sample_grad, new_pred, torch.tensor([step] * mask.shape[0], dtype=torch.long, device=self.device))

			cond_loss = self._compute_cond_loss(temp_cosines, temp_amplitudes)
			cond_loss.backward()

			pos_grad = -sample_grad.pos.grad.clone()  

			gs = self._get_condition_scale(a)

			pred_cond = (1 - a) * (pos_grad) * gs
			
		return pred_cond

	def _get_expected_dynamics(self, pos_, sample_grad, pred, step):
		"""From the current position get the expected pos at t=0, and compute the dynamics."""
		pos_denoised = self._denoise_positions(pos_, pred, step)
	
		pos_denoised = pos_denoised[0:sample_grad.num_nodes,:] # only real nodes, no padding
		anm = get_anm_hessian(pos_denoised, self.ff)
		_, new_vecs = get_anm_eigens(anm)  # [3n], [3n, 3n]

		temp_disp, temp_cosines, temp_amplitudes = self._temp_eigs(new_vecs)

		return temp_disp, temp_cosines, temp_amplitudes
    
	def record_results(self, sample):
		"""Record the results of the conditioning at t=0.
		Args:
		sample (Data): sample with the final positions."""

		pos_ = sample.pos
		anm = get_anm_hessian(pos_, self.ff)
		_, new_vecs = get_anm_eigens(anm) # [3n], [3n, 3n] 

		temp_disp, temp_cosines, temp_amplitudes = self._temp_eigs(new_vecs)
		_ = self._compute_cond_loss(temp_cosines, temp_amplitudes)

		sample.final_displacements = temp_disp
		sample.target_displacements = self.target_displacements

		sample.cosines_loss = torch.tensor(self.monitor_cosines)
		sample.amplitudes_loss = torch.tensor(self.monitor_amplitudes)
		sample.d_loss = torch.tensor(self.monitor_total)

		sample.cond_node_inds_dynamics = self.cond_node_inds   

		return sample
    