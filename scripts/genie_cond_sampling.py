from genie.config import Config
from src.diffusion.genie_mod import GenieMod
from src.utils.data_utils import load_target
from src.constants import WEIGHTS_PATH, PROJECT_PATH
import os
import torch
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--cf', type=float, default=0.5, help='Fraction of timesteps with active conditioning')
	parser.add_argument('--length', type=int, default=150, help='Length of the novel sample')
	parser.add_argument('--eta', type=float, default=0.4, help='Genie noise scale')
	parser.add_argument('--gs_struct', type=int, default=2000, help='Guidance scale for structure conditioning')
	parser.add_argument('--gs_dyn', type=int, default=3000, help='Guidance scale for dynamics conditioning')
	parser.add_argument('--structure_on', action='store_true', help='Enable structure conditioning')
	parser.add_argument('--dynamics_on', action='store_true', help='Enable dynamics conditioning')
	args = parser.parse_args()
	
	cf = args.cf
	length = args.length
	eta = args.eta
	gs_struct = args.gs_struct
	gs_dyn = args.gs_dyn
	device='cuda'

	assert args.structure_on or args.dynamics_on, "At least one conditioner should be on"

	SAMPLE_PATH = os.path.join(PROJECT_PATH, '3adk_samples')
	if not os.path.exists(SAMPLE_PATH):
		os.makedirs(SAMPLE_PATH)

	target_filename = "3adk_target.npz"
	hinge_disp, hinge_inds, hinge_pos, hinge_pdb_id = load_target(target_filename, device=device)
	
	assert hinge_inds.max() < length, "Protein too short for the target"

	config = Config(os.path.join(WEIGHTS_PATH,'scope_l_256', 'configuration'))
	model = GenieMod.load_from_checkpoint(os.path.join(WEIGHTS_PATH, 'scope_l_256', 'epoch=29999.ckpt'), config=config)

	model.setup_schedule()
	for k in range(10):
		model.empty_conditioners()
		if args.structure_on:
			model.init_structconditioner(motif_pos=hinge_pos, motif_inds=hinge_inds, cond_frac=cf, gs=gs_struct)
		if args.dynamics_on:
			model.init_eigenconditioner(target_disp=hinge_disp, cond_node_inds=hinge_inds, cond_frac=cf, gs=gs_dyn)

		smp_init = model.init_blobs(1,length)
		smp = model.sample(smp_init, eta)
		rmsd = float(smp.rmsd)
		dloss = float(smp.d_loss[-1])
		torch.save(smp, os.path.join(SAMPLE_PATH, f'rmsd_{rmsd:.4f}_dloss_{dloss:.4f}.pt'))
		model.reset_conditioners()
