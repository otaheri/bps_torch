# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#

import torch
import numpy as np
import chamfer_distance as chd

from .utils import to_np, to_tensor
from .tools import sample_grid_cube
from .tools import sample_grid_sphere
from .tools import sample_sphere_nonuniform
from .tools import sample_sphere_uniform
from .tools import normalize, denormalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bps_torch():
    def __init__(self,
                 bps_type='random_uniform',
                 n_bps_points=1024,
                 radius=1.,
                 n_dims=3,
                 random_seed=13,
                 custom_basis=None,
                 **kwargs):

        if custom_basis is not None:
            bps_type = 'custom'

        if bps_type == 'random_uniform':
            basis_set = sample_sphere_uniform(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
        elif bps_type == 'random_nonuniform':
            basis_set = sample_sphere_nonuniform(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
        elif bps_type == 'grid_cube':
            # in case of a grid basis, we need to find the nearest possible grid size
            grid_size = int(np.round(np.power(n_bps_points, 1 / n_dims)))
            basis_set = sample_grid_cube(grid_size=grid_size, minv=-radius, maxv=radius)
        elif bps_type == 'grid_sphere':
            basis_set = sample_grid_sphere(n_points=n_bps_points, n_dims=n_dims, radius=radius)
        elif bps_type == 'custom':
            # in case of a grid basis, we need to find the nearest possible grid size
            if custom_basis is not None:
                basis_set = to_tensor(custom_basis).to(device)
            else:
                raise ValueError("Custom BPS arrangement selected, but no custom_basis provided.")
        else:
            raise ValueError("Invalid basis type. Supported types: \'random_uniform\', \'random_nonuniform\', \'grid_cube\', \'grid_sphere\', and \'custom\'")

        self.bps = basis_set.view(1,-1,n_dims)

    def encode(self,
               x,
               feature_type=['dists'],
               x_features=None,
               custom_basis=None,
               **kwargs):


        x = to_tensor(x).to(device)
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(0)

        bps = self.bps if custom_basis is None else custom_basis
        bps = to_tensor(bps).to(device)
        _, P_bps, D = bps.shape
        N, P_x  , D = x.shape

        deltas = torch.zeros([N, P_bps, D]).to(device)
        b2x_idxs = torch.zeros([N, P_bps],dtype=torch.long).to(device)

        ch_dist = chd.ChamferDistance()

        for fid in range(0, N):
            X = x[fid:fid+1]
            b2x, x2b, b2x_idx, x2b_idx = ch_dist(bps, X)
            deltas[fid] = X[:,b2x_idx.to(torch.long)] - bps
            b2x_idxs[fid] = b2x_idx

        x_bps = {}
        if 'dists' in feature_type:
            # x_bps.append(torch.sqrt(torch.pow(deltas, 2).sum(2, keepdim=True)))
            x_bps['dists'] = torch.sqrt(torch.pow(deltas, 2).sum(2))
        if 'deltas' in feature_type:
            # x_bps.append(deltas)
            x_bps['deltas'] = deltas
        if 'closest' in feature_type:
            b2x_idxs_expanded = b2x_idxs.view(N, P_bps, 1).expand(N, P_bps, D)
            # x_bps.append(x.gather(1, b2x_idxs_expanded))
            x_bps['closest'] = x.gather(1, b2x_idxs_expanded)
        if 'features' in feature_type:
            try:
                F = x_features.shape[2]
                b2x_idxs_expanded = b2x_idxs.view(N, P_bps, 1).expand(N, P_bps, F)
                # x_bps.append(x.gather(1, b2x_idxs_expanded))
                x_bps['features'] = x.gather(1, b2x_idxs_expanded)
            except:
                raise ValueError("No x_features parameter is provided!")
        if len(x_bps) < 1:
            raise ValueError("Invalid cell type. Supported types: \'dists\', \'deltas\', \'closest\', \'features\'")

        # return torch.cat(x_bps,dim=2)
        return x_bps

    def decode(self,
               x_deltas,
               custom_basis=None,
               **kwargs):

        x = to_tensor(x_deltas).to(device)
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(dim=0)

        bps = self.bps if custom_basis is None else custom_basis
        bps = to_tensor(bps).to(device)
        if len(bps)<2:
            bps = bps.unsqueeze(dim=0)

        _, P_bps, D = bps.shape
        N, P_x, D = x.shape

        bps_expanded = bps.expand(N, P_bps, D)

        return bps_expanded + x



