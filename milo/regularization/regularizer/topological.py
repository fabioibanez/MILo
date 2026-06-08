from topologylayer.nn import LevelSetLayer, SumBarcodeLengths, PartialSumBarcodeLengths
from torch import nn

import numpy as np
import torch
from topologylayer.functional.persistence import SimplicialComplex
import time

#indices that are faces/exges of the tets
T = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
E = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])

def init_complex(tets, n_verts, max_dim = 2):
    if torch.is_tensor(tets):
        tets = tets.detach().cpu().numpy()
    tets = tets.astype(np.int64)

    triangles = tets[:, T].reshape(-1, 3)
    triangles = np.unique(np.sort(triangles, axis=1), axis=0)

    edges = tets[:, E].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    tets_sorted = np.unique(np.sort(tets, axis=1), axis=0)

    sc = SimplicialComplex()
    sc.bulk_append([[v] for v in range(n_verts)])
    sc.bulk_append(edges.tolist())
    if max_dim >= 1: 
        sc.bulk_append(triangles.tolist())
    if max_dim == 2:
        sc.bulk_append(tets_sorted.tolist())
    return sc


class LevelSetLayer3D(LevelSetLayer):
    def __init__(self, tets, n_verts, maxdim=2, sublevel=True):
        t = time.time()
        tmpcomplex = init_complex(tets, n_verts, max_dim=maxdim)
        super(LevelSetLayer3D, self).__init__(tmpcomplex, maxdim=maxdim, sublevel=sublevel)
        print('construction time', time.time()-t)

    def rebuild(tets,n_verts):
        self.complex = init_complex(tets, n_verts)


class TopLoss3D(nn.Module):
    def __init__(self, tets, n_verts,b0,b1,b2):
        super(TopLoss3D, self).__init__()
        self.pdfn = LevelSetLayer3D(tets,n_verts, sublevel=True)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=b0)
        self.topfn1 = PartialSumBarcodeLengths(dim=1, skip=b1)
        self.topfn2 = PartialSumBarcodeLengths(dim=2, skip=b2)

    def forward(self, beta):
        t = time.time()
        dgminfo = self.pdfn(beta)
        l = self.topfn(dgminfo) + self.topfn1(dgminfo) + self.topfn2(dgminfo)
        print('forward time', time.time()-t)
        return l
    
class CCLoss(nn.Module):
    def __init__(self, tets,n_verts):
        super(CCLoss, self).__init__()        
        self.pdfn = LevelSetLayer3D(tets,n_verts, maxdim=0, sublevel=True)
        self.topfn = SumBarcodeLengths(dim=0)
    
    def forward(self,beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)
    
    
class DTU_test(nn.Module):
    def __init__(self, tets, n_verts,b0=1,b1=0):
        super(DTU_test, self).__init__()
        self.pdfn = LevelSetLayer3D(tets,n_verts, maxdim=1,sublevel=True)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=b0)
        self.topfn1 = PartialSumBarcodeLengths(dim=1, skip=b1)

    def forward(self, beta):
        t = time.time()
        dgminfo = self.pdfn(beta)
        l = self.topfn(dgminfo) + self.topfn1(dgminfo)
        print('forward time', time.time()-t)
        return l
    
