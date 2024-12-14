# ---------------------------------------------------------------
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for TCM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch

# TCM parameterization
class TCMPrecond:
    def __init__(self,
        model_t,             # Teacher model (stage-1)
        model_s,             # Student model (stage-2)
        transition_t = 2.,    # Transition time step (t')
        max_t = 80.,         # Maximum time step
        teacher_pkl = None,  # Not used
    ):
        
        self.model_t = model_t
        self.model_s = model_s
        self.transition_t = transition_t

    def __call__(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        mask = sigma >= self.transition_t # If this is true, use the second-stage model. Otherwise, use the first-stage model
        mask = mask.squeeze()
        

        rng_state = torch.cuda.get_rng_state()
        if (~mask).any():
            with torch.no_grad():
                out_t = self.model_t(
                    x, 
                    sigma, 
                    class_labels, 
                    force_fp32, 
                    **model_kwargs
                )
        else:
            out_t = torch.zeros_like(x).to(torch.float32)
        torch.cuda.set_rng_state(rng_state)
        if mask.any():
            out_s = self.model_s(
                x, 
                sigma, 
                class_labels, 
                force_fp32, 
                **model_kwargs
            )
        else:
            out_s = torch.zeros_like(x).to(torch.float32)

        out = mask.view(-1,1,1,1) * out_s + (~mask.view(-1,1,1,1)) * out_t
        return out


