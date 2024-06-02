#python 3.10
"""
    align the monocular relative depth
NOTE: Give a monocular depth D1 and D_gt, we want to find a group of a, b
D1_abs = 1/(a * D1 + b), a and b are computed by aligning 1/D_gt and D1  
"""

def align_mde(D1, D_gt, valid=None):
    """
        align the monocular relative depth
    Args:
        D1: monocular depth, (B H W)
        D_gt: ground truth depth, (B H W)
        valid: valid mask, (B H W)
    """
    if valid is not None:
        D1 = D1[valid]
        D_gt = D_gt[valid]
    import ipdb; ipdb.set_trace()
    # align the monocular depth
    a = (D1 * D_gt).mean() / (D1 * D1).mean()
    b = (D_gt / D1).mean()
    return a, b

def ortho_proj(pts):
    return NotImplemented