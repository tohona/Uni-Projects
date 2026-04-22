import argparse
from os import stat
import numpy as np
import torch
from typing import Dict
import copy


# def random_unstructured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:
#     """Set valus in conv filters to zero acoordings to to the prune ratio."""
#     state_dict = copy.deepcopy(state_dict)
# 
#     for i in range(1, 7):
#         size = state_dict[f"conv{i}.weight"].size()
#         mask = (torch.rand(size) > prune_ratio)
#         state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask
# 
#     return state_dict
# 
#
# def l1_unstructured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:
#     """Set valus in conv filters to zero acoordings to to the prune ratio."""
#     state_dict = copy.deepcopy(state_dict)
#     
#     for i in range(1, 7):
#         threshold = np.percentile(torch.abs(state_dict[f"conv{i}.weight"]), prune_ratio * 100)
#         mask = torch.abs(state_dict[f"conv{i}.weight"]) > threshold
#         state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask
# 
#     return state_dict


def l1_structured_pruning(state_dict: Dict, prune_ratio: float) -> Dict:
    """Set entire output channels to zero based on magnitude."""
    state_dict = copy.deepcopy(state_dict)
    
    for i in range(3, 9):
        l1_norm = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1, 2, 3))

        threshold = np.percentile(l1_norm.to('cpu'), prune_ratio * 100)
        mask_1d = l1_norm > threshold
        mask_4d = mask_1d.view(-1, 1, 1, 1)

        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"] * mask_4d
        # state_dict[f"bn{i}.weight"] = state_dict[f"bn{i}.weight"] * mask_1d

    return state_dict


def densify_state_dict(state_dict: Dict) -> Dict:
    """Iteratively drop entire output channels based on magnitude.

    This function is meant to be called after setting the weights of the
    weakst channels to zero using 'l1_structured_pruning()'.
    """
    state_dict = copy.deepcopy(state_dict)
    
    for i in range(3, 9):
        l1_norm = torch.sum(torch.abs(state_dict[f"conv{i}.weight"]), dim=(1, 2, 3))

        epsilon = 1e-6
        mask_1d = l1_norm > epsilon

        state_dict[f"conv{i}.weight"] = state_dict[f"conv{i}.weight"][mask_1d]
        state_dict[f"conv{i+1}.weight"] = state_dict[f"conv{i+1}.weight"][:, mask_1d]
        state_dict[f"bn{i}.weight"] = state_dict[f"bn{i}.weight"][mask_1d]
        state_dict[f"bn{i}.bias"] = state_dict[f"bn{i}.bias"][mask_1d]
        state_dict[f"bn{i}.running_mean"] = state_dict[f"bn{i}.running_mean"][mask_1d]
        state_dict[f"bn{i}.running_var"] = state_dict[f"bn{i}.running_var"][mask_1d]

    return state_dict


def prune_model(state_dict: Dict, prune_ratio: float) -> Dict:
    """Prune network based on l1 norm."""
    zeroed = l1_structured_pruning(state_dict, prune_ratio)
    pruned = densify_state_dict(zeroed)

    return pruned


def main():
    parser = argparse.ArgumentParser(prog="prune.py",
                                     description="Prune TinyYoloV2 net")

    parser.add_argument('state_dict_path', type=str,
                        help='the path to the model file to prune')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Ouptut filename for the pruned model')
    parser.add_argument('-r', '--ratio', required=True, type=float,
                        help='Ratio by whitch the net ist to be pruned')

    args = parser.parse_args()

    state_dict = torch.load(args.state_dict_path)
    prune_model(state_dict, args.ratio)


if __name__=="__main__":
    main()
    
