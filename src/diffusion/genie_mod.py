import torch
from tqdm import tqdm
from src.utils.nma_diffusion_utils import subtract_genie_com

from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.diffusion.genie import Genie
from src.diffusion.eigenconditioner import GenieEigenconditioner
from src.diffusion.structconditioner import GenieStructconditioner
from torch_geometric.data import Data, Batch


class GenieMod(Genie):
	def __init__(self, config):
		super().__init__(config)
		self.max_n_res = config.io["max_n_res"]

	def empty_conditioners(self):
		self.conditioners = []

	def reset_conditioners(self):
		del self.conditioners

	def init_eigenconditioner(
		self,
		cond_frac: float,
		target_disp: torch.Tensor,
		cond_node_inds: torch.Tensor,
		gs: float,
		ff_cutoff: float = 13.0,
	):
		self.conditioners.append(GenieEigenconditioner(predict_noise_func=self.predict_noise,
								alphas_schedule=self.alphas,
								alphabars_schedule=self.alphas_cumprod,
								ff_cutoff=ff_cutoff,
								device=self.device,
								))
		self.conditioners[-1].set_condition(cond_frac=cond_frac, target_disp=target_disp, cond_node_inds=cond_node_inds, gs=gs)

	def init_structconditioner(
		self,
		motif_pos: torch.Tensor,
		cond_frac: float,
		motif_inds: torch.Tensor,
		gs: float,
	):
		self.conditioners.append(GenieStructconditioner(predict_noise_func=self.predict_noise,
								alphas_schedule=self.alphas,
								alphabars_schedule=self.alphas_cumprod,
								device=self.device,
								))
		self.conditioners[-1].set_condition(target_pos=motif_pos, motif_inds=motif_inds, cond_frac=cond_frac, gs=gs)
		
	def p_custom(self, ts, s, mask, noise_pred, eta=1.0):

		# [b, 1, 1]
		w_noise = ((1. - self.alphas[s].to(self.device)) / torch.sqrt(1 - self.alphas_cumprod[s]))[:,None,None]

		# [b, n_res, 3]
		trans_mean = (1. / torch.sqrt(self.alphas[s])).view(-1, 1, 1).to(self.device) * (ts.trans - w_noise * noise_pred.trans)
		trans_mean = trans_mean * mask.unsqueeze(-1)

		if (s == 0.0).all():
			rots_mean = compute_frenet_frames(trans_mean, mask)
			return T(rots_mean.detach(), trans_mean.detach())
		else:
			# [b, n_res, 3]
			trans_z = torch.randn_like(ts.trans).to(self.device)

			# [b, 1, 1]
			trans_sigma = torch.sqrt(self.betas[s]).view(-1, 1, 1).to(self.device)

			# [b, n_res, 3]
			trans = trans_mean + trans_sigma * trans_z * eta
			trans = trans * mask.unsqueeze(-1)

			# [b, n_res, 3, 3]
			rots = compute_frenet_frames(trans, mask)

			return T(rots.detach(), trans.detach())


	def predict_noise(self, sample, tt, mask):
		"""Predict noise on the positions.
		Args:
			sample (Batch): batch of denoised graphs
			tt (int): time steps """
		
		noise_pred_trans = sample.frames.trans - self.model(sample.frames, tt, mask).trans 
		noise_pred_trans = noise_pred_trans.reshape(-1,3) # (1, n_max_res, 3) -> (1*n_max_res,3)
		return noise_pred_trans
	
	def init_blobs(self, num_blobs, num_nodes):
		max_n_res = self.max_n_res
		assert num_nodes <= max_n_res
		mask = torch.cat([torch.ones((num_blobs, num_nodes)), torch.zeros((num_blobs, max_n_res - num_nodes))], dim=1).to(
            self.device
        )
		blobs = [] 
		sample_frames = self.sample_frames(mask)
		for i in range(num_blobs):

			single_blob = Data(edge_index=torch.tensor([[max_n_res],[0]]).to(self.device)) # dummy edge index
			single_blob.node_order = torch.arange(max_n_res, dtype=torch.long).to(self.device)
			blobs.append(single_blob)
			
		if len(blobs) == 1:
			blobs = blobs[0]
		else:
			blobs = Batch.from_data_list(blobs)
		blobs.mask = mask
		blobs.num_nodes = num_nodes
		blobs.frames = sample_frames
		return blobs

	def sample(
		self, sample, eta, keep_trajectory=False
	):

		self.eval() 

		n_steps = len(self.alphas)

		num_conds = len(self.conditioners)
		for c_i in range(num_conds):
			self.conditioners[c_i].set_monitor()
	
		# sample = sample.clone()
		trajectory = [] if keep_trajectory else None
		mask = sample.mask


		for t in tqdm(
			reversed(range(n_steps)),
			desc="sampling loop time step",
			total=n_steps,
		):
			tt = torch.Tensor([t] * mask.shape[0]).long().to(self.device)
			
			noise_pred_trans = sample.frames.trans - self.model(sample.frames, tt, mask).trans
			noise_pred_rots = torch.eye(3).view(1, 1, 3, 3).repeat(sample.frames.shape[0], sample.frames.shape[1], 1, 1)
			noise_pred = T(noise_pred_rots, noise_pred_trans)


			sample.pos = sample.frames.trans.reshape(-1,3) # (1, n_max_res, 3) -> (n_max_res,3)
			pred_cond = torch.zeros_like(sample.pos, device=self.device)
			for c_i in range(num_conds):
				
				pred_cond_ci = self.conditioners[c_i].calc_condition(sample, t) 
				pred_cond_ci = subtract_genie_com(pred_cond_ci, num_nodes=sample.num_nodes)
				pred_cond += pred_cond_ci


			sample.frames.trans += pred_cond.reshape(1, -1, 3)

			sample_frames = self.p_custom(ts=sample.frames, s=tt, mask=mask, noise_pred=noise_pred, eta=eta)
			sample.frames = sample_frames
			if keep_trajectory:
				trajectory.append(sample.frames.trans[:,0:sample.num_nodes,:].reshape(-1,3))

		sample.frames.trans = sample.frames.trans[0,0:sample.num_nodes,:] # select only the real nodes 
		sample_out = Data(pos=sample.frames.trans) 
		for c_i in range(num_conds):
			sample_out = self.conditioners[c_i].record_results(sample_out)  

		if keep_trajectory:
			return torch.stack(trajectory, dim=0)
		else:
			return sample_out