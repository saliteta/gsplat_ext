from gsplat.strategy.default import DefaultStrategy
import torch

class FeatureStrategy(DefaultStrategy):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prune_scale3d = 3e-3
    

    def step_pre_backward(self, params, optimizers, state, step, info):
        """Callback function to be executed before the `loss.backward()` call."""
        # We need to retain gradients for the 2D means like the parent class does
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(self, params, optimizers, state, step, info, packed):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        # Update state to track gradients and statistics
        self._update_state(params, state, info, packed=packed)

        if (
             step % self.refine_every == 0
        ):
            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()
    
    def _prune_gs(self, params, optimizers, state, step):
        """
        Clamp large scale GSs to normal size and initialize their optimizer state
        """
        is_too_big = (
            torch.exp(params["scales"]).max(dim=-1).values
            > self.prune_scale3d * state["scene_scale"]
        )
        
        if is_too_big.any():
            """Instead remove, we can make larger scale GS shrink their scale to normal"""
            max_scale = torch.tensor(self.prune_scale3d * state["scene_scale"], device=params["scales"].device)
            params["scales"] = params["scales"].clamp(max=torch.log(max_scale))
            
            # Initialize optimizer state for the rescaled parameters
            self._initialize_optimizer_state_for_rescaled(params, optimizers, is_too_big)
        
        return is_too_big.sum().item()
    
    def _initialize_optimizer_state_for_rescaled(self, params, optimizers, mask):
        """
        Initialize optimizer state for parameters that were just rescaled
        """
        for param_name, optimizer in optimizers.items():
            if param_name in params:
                # Get the parameter that was rescaled
                param = params[param_name]
                
                # Find which parameters were affected by the rescaling
                if param_name == "scales":
                    # For scales, we need to reset the optimizer state for the rescaled parameters
                    if hasattr(optimizer, 'state'):
                        for i in torch.where(mask)[0]:
                            if i.item() in optimizer.state:
                                # Reset the optimizer state for this parameter
                                del optimizer.state[i.item()]
                
                # Alternative approach: reset momentum for affected parameters
                if hasattr(optimizer, 'param_groups'):
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            if p is param:
                                # Reset momentum for the entire parameter tensor
                                # This is a more aggressive approach that resets all momentum
                                if hasattr(optimizer, 'state'):
                                    for key in list(optimizer.state.keys()):
                                        if optimizer.state[key] is p:
                                            del optimizer.state[key]
        


