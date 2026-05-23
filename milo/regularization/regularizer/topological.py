from topologylayer.nn import LevelSetLayer, SumBarcodeLengths, PartialSumBarcodeLengths
from torch import nn

import numpy as np
import torch
from topologylayer.functional.persistence import SimplicialComplex

#indices that are faces/exges of the tets
T = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
E = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])

def init_complex(tets, n_verts):
    if torch.is_tensor(tets):
        tets = tets.detach().cpu().numpy()
    tets = tets.astype(np.int64)

    triangles = tets[:, T].reshape(-1, 3)
    triangles = np.unique(np.sort(triangles, axis=1), axis=0)

    edges = tets[:, E].reshape(-1, 2)
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    tets_sorted = np.unique(np.sort(tets, axis=1), axis=0)

    sc = SimplicialComplex()
    for v in range(n_verts):
        sc.append([v])
    for e in edges.tolist():
        sc.append(e)
    for t in triangles.tolist():
        sc.append(t)
    for tet in tets_sorted.tolist():
        sc.append(tet)
    return sc

class LevelSetLayer3D(LevelSetLayer):
    def __init__(self, tets, n_verts, maxdim=2, sublevel=True):
        tmpcomplex = init_complex(tets, n_verts)
        super(LevelSetLayer3D, self).__init__(tmpcomplex, maxdim=maxdim, sublevel=sublevel)

    def rebuild(tets,n_verts):
        self.complex = init_complex(tets, n_verts)


class TopLoss3D(nn.Module):
    def __init__(self, tets, n_verts,b0,b1,b2):
        super(TopLoss3D, self).__init__()
        self.pdfn = LevelSetLayer3D(tets,n_verts, sublevel=True)
        assert b0 > 0, 'there must be at least one connected component in the target'
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=b0)
        self.topfn1 = PartialSumBarcodeLengths(dim=1, skip=b1)
        self.topfn2 = PartialSumBarcodeLengths(dim=2, skip=b2)

    def forward(self, beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo) + self.topfn1(dgminfo) + self.topfn2(dgminfo)
    
class CCLoss(nn.Module):
    def __init__(self, tets,n_verts):
        super(CCLoss, self).__init__()        
        self.pdfn = LevelSetLayer3D(tets,n_verts, maxdim=0, sublevel=True)
        self.topfn = PartialSumBarcodeLengths(dim=0, skip=1)
    
    def forward(self,beta):
        dgminfo = self.pdfn(beta)
        return self.topfn(dgminfo)

