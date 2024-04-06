import os
import numpy as np
import torch
from src.constants import DATA_PATH

def load_target(target_filename, device='cuda'):
	target_dict = np.load(os.path.join(DATA_PATH, target_filename))

	loaded_hinge_disp = target_dict['hinge_disp']
	loaded_hinge_inds = target_dict['hinge_inds']
	loaded_hinge_pos = target_dict['hinge_pos']
	loaded_hinge_pdb_id = target_dict['hinge_pdb_id']

	loaded_hinge_disp = torch.tensor(loaded_hinge_disp, device=device)
	loaded_hinge_inds = torch.tensor(loaded_hinge_inds, device=device)
	loaded_hinge_pos = torch.tensor(loaded_hinge_pos, device=device)

	return loaded_hinge_disp, loaded_hinge_inds, loaded_hinge_pos, loaded_hinge_pdb_id

