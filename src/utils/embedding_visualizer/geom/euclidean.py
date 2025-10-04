""" Geometric utility functions, mostly for standard Euclidean operations."""

import torch

MIN_NORM = 1e-15


# def orthonormal(Q):
#     """Return orthonormal basis spanned by the vectors in Q.

#     Q: (..., k, d) k vectors of dimension d to orthonormalize
#     """
#     k = Q.size(-2)
#     _, _, v = torch.svd(Q, some=False)  # Q = USV^T
#     Q_ = v[:, :k]
#     return Q_.transpose(-1, -2)  # (k, d) rows are orthonormal basis for rows of Q
######################################################
# def orthonormal(Q):
#     """Return orthonormal basis spanned by the vectors in Q.

#     Q: (..., k, d) k vectors of dimension d to orthonormalize
#     """
#     k = Q.size(-2)
#     noise = torch.randn_like(Q) * 1e-6
#     Q_noisy = Q + noise
#     _, _, v = torch.svd(Q_noisy, some=False)  # Q = USV^T
#     Q_ = v[:, :k]
#     return Q_.transpose(-1, -2)  # (k, d) rows are orthonormal basis for rows of Q
######################################################
# def orthonormal(Q):
#     print("this method is called")
#     k = Q.size(-2)
#     _, _, v = torch.linalg.svd(Q, full_matrices=False)
#     Q_ = v[:k].transpose(-1, -2)
#     return Q_
######################################################
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(torch.dot(v, b) * b for b in basis)
        if torch.norm(w) > 1e-10:
            basis.append(w / torch.norm(w))
    return torch.stack(basis)

def orthonormal(Q):
    try:
        k = Q.size(-2)
        _, _, v = torch.svd(Q, some=False)
        Q_ = v[:, :k]
        return Q_.transpose(-1, -2)
    except:
        return gram_schmidt(Q)
######################################################


def euc_reflection(x, a):
    """
    Euclidean reflection (also hyperbolic) of x
    Along the geodesic that goes through a and the origin
    (straight line)

    NOTE: this should be generalized by reflect()
    """
    xTa = torch.sum(x * a, dim=-1, keepdim=True)
    norm_a_sq = torch.sum(a ** 2, dim=-1, keepdim=True)
    proj = xTa * a / norm_a_sq.clamp_min(MIN_NORM)
    return 2 * proj - x


def reflect(x, Q):
    """Reflect points (euclidean) with respect to the space spanned by the rows of Q.

    Q: (k, d) set of k d-dimensional vectors (must be orthogonal)
    """
    ref = 2 * Q.transpose(0, 1) @ Q - torch.eye(x.shape[-1], device=x.device)
    return x @ ref
