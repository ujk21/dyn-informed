from abc import ABC, abstractmethod
from torch_geometric.data import Data



class Conditioner(ABC):
    """
    Abstract class Conditioner that defines the blueprint for a conditioner with
    several methods.
    """

    @abstractmethod
    def set_condition(self, *args, **kwargs):
        """
        Method to set the condition.
        """

    @abstractmethod
    def _compute_cond_loss(self, *args, **kwargs):
        """
        Method to compute the conditioning loss.
        """

    @abstractmethod
    def _denoise_positions(self, *args, **kwargs):
        """
        Method to denoise positions.
        """

    @abstractmethod
    def record_results(self, *args, **kwargs):
        """
        Method to record the results.
        """

    @abstractmethod
    def set_monitor(self, *args, **kwargs):
        """
        Method to set the monitoring of relevant physical quantities.
        """
        
    def gs_time_scaling(self, a):
        """
        If required, the guidance scale can become dependent on time and alpha.
        """
        return a
        
    def _conditioning_started(self, step):
        if not hasattr(self, "cond_start_step"):
            self.cond_start_step = self.n_steps * self.cond_frac

        if step < self.cond_start_step:
            return True
        else:
            return False

    def _total_denoise(self, x, pred_noise, step):
        """Get the expected positions at t=0 by totally denoising positions at time t>0.
        Args:
            x (torch.Tensor): Noised positions
            pred_noise (torch.Tensor): Noise tensor of shape [sum(batch_num_nodes), ...].
            steps (torch.LongTensor): Timesteps at which to denoise.
        Returns:
            torch.Tensor: Fully denoised positions"""

        abar = self.alphabars_schedule[step]
        x_denoised = (
            x - (1.0 - abar).sqrt() * pred_noise
        ) / abar.sqrt()

        return x_denoised

    def _get_condition_scale(self, a):
        gs = self.gs * self.gs_time_scaling(a)
        return gs

    def initialize_sample_grad(self, sample):
        sample_grad = Data(
            pos=sample.pos.data.clone().detach().requires_grad_(True),
            edge_index=sample.edge_index.clone().detach(),
            node_order=sample.node_order.clone().detach(),
            num_nodes=sample.num_nodes,
        )
        return sample_grad
