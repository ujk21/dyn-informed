import torch
from torch_geometric.data import Data
import src.utils.geometry as geometry
from src.diffusion.conditioner import Conditioner
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.affine_utils import T

class GenieStructconditioner(Conditioner):
	def __init__(self, predict_noise_func, alphas_schedule, alphabars_schedule, device):
		super().__init__()

		self._pred_noise = predict_noise_func

		self.loss_func = torch.nn.functional.l1_loss
		self.alphas_schedule = alphas_schedule
		self.alphabars_schedule = alphabars_schedule
		self.n_steps = len(alphas_schedule)
		self.device = device

		self.calc_condition = self.apply_cond_motif
		
		self.counter = 0

	def set_monitor(self):
		"""Quantity to monitor during sampling."""

		self.monitor_graph_loss = []

	def set_condition(self, target_pos: torch.Tensor, motif_inds: torch.Tensor, cond_frac: float, gs: float):
		"""Set the condition of the diffusion process.
		Args:
			target_pos (torch.Tensor): the target position of the motif
			motif_inds (torch.Tensor): the indices of the sample residues that belong to the motif e.g. [15,16,20,31]
			cond_frac (float): fraction of the last timesteps when we apply the condition
			gs (float): guidance scale
		"""
		self.motif_pos_target = target_pos
		self.motif_inds = motif_inds
		self.cond_frac = cond_frac
		self.gs = gs
		
	def _denoise_positions(self, pos, pred, step):
		return self._total_denoise(pos, pred, step)
	
	def gs_time_scaling(self, a):
		"Time scaling to modify guidance scale. Override if needed."
		return torch.sqrt(1.5 - a)
	
	def _compute_cond_loss(self, pos_denoised, motif_pos_target, motif_inds, step):

		motif_pos_sample = pos_denoised[motif_inds]  # (n_motif_nodes, 3)

		# align denoised motif to reference motif via Kabsch algorithm
		if self.counter == 0: # time to recompute the aligned target motif
			rot_mat, trans_vec = geometry.differentiable_kabsch(motif_pos_sample, motif_pos_target, step) # instead of rotating the motif in sample, we rotate orig target
			aligned_motif_pos_target = geometry.rototranslate(motif_pos_target, rot_mat, trans_vec)
			self.stored_target_motif = aligned_motif_pos_target
			self.counter = 5
		else:
			aligned_motif_pos_target = self.stored_target_motif.detach().clone()
			self.counter -= 1
	
		# rot_mat, trans_vec = geometry.differentiable_kabsch(motif_pos_target, motif_pos_sample, step)
		# aligned_motif_pos_sample = geometry.rototranslate(motif_pos_sample, rot_mat, trans_vec)

		graph_loss = self.loss_func(motif_pos_sample, aligned_motif_pos_target)
		self.monitor_graph_loss.append(graph_loss.item()) 

		return graph_loss

	def apply_cond_motif(self, sample: Data, step: int):
		"""Condition the predictions of the model on a given structural motif.
		Args:
			sample (Data): the sample graph
			step (int): the current step of the training
		Returns:
			pred_cond (torch.Tensor): the conditional score for the positions
		"""
		a = self.alphas_schedule[step]
		mask = sample.mask

		if self._conditioning_started(step) == False:
			return torch.zeros_like(sample.pos, device=self.device)

		with torch.enable_grad():

			sample_grad = self.initialize_sample_grad(sample)

			trans = sample_grad.pos.unsqueeze(0)
			rots_mean = compute_frenet_frames(trans, mask)
			sample_grad.frames = T(trans=trans, rots=rots_mean)
			new_pred = self._pred_noise(sample_grad, torch.tensor([step], device=sample.pos.device), mask=mask)
			pos_denoised = self._denoise_positions(sample_grad.pos, new_pred, step)
			batch_loss = self._compute_cond_loss(pos_denoised, self.motif_pos_target, self.motif_inds, step)

			batch_loss.backward()

			pos_grad = - sample_grad.pos.grad.clone()

		gs = self._get_condition_scale(a)
		pred_cond = (1 - a)*pos_grad*gs

		return pred_cond
	
	def record_results(self, sample):
		
		sample.struct_loss = torch.tensor(self.monitor_graph_loss)
		sample.motif_pos_sample = sample.pos[self.motif_inds]
		sample.motif_pos_target = self.motif_pos_target

		rmsd = geometry.align_and_compute_rmsd(sample.motif_pos_sample, sample.motif_pos_target, step=0)
		sample.rmsd = rmsd
		
		sample.motif_inds = self.motif_inds
        
		return sample
