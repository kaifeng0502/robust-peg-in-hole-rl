"""Tensor math for the versioned local-contact action and reward candidate."""
import torch


def relative_z_target(tool_z, filtered_z_action, step_m):
    """One target per control interval: positive action lifts, negative lowers."""
    return tool_z + filtered_z_action * step_m


def entry_potentials(xy_error, depth, fine_sigma, coarse_sigma, approach_sigma, entry_depth):
    """Coarse/fine alignment and continuous approach-to-entry potential.

    This is a candidate shaping objective, not a policy-invariance claim. A
    closed geometric path has zero undiscounted potential-change sum; temporary
    unloading loses potential that can be recovered by returning to the state.
    Geometry is simulator truth used for training rewards only.
    """
    fine = torch.exp(-torch.square(xy_error / fine_sigma))
    coarse = torch.exp(-torch.square(xy_error / coarse_sigma))
    alignment = 0.5 * (fine + coarse)
    approach = torch.exp(-torch.clamp(-depth, min=0.0) / approach_sigma)
    entered = torch.clamp(depth / entry_depth, min=0.0, max=1.0)
    return alignment, fine * (approach + entered)
