import numpy as np

'''
:param points1 n1 * 2
       points2 n2 * 2
'''
def cal_hausdorff_distance(lines, k_center):
    dh_1_2 = -1
    for i in range(len(lines)):
        dis = np.min(np.sqrt(np.sum((lines[i, :] - k_center) ** 2, axis=1)))
        dh_1_2 = max(dh_1_2,dis)
    dh_2_1 = -1
    for j in range(len(k_center)):
        dis = np.min(np.sqrt(np.sum((k_center[j, :] - lines) ** 2, axis=1)))
        dh_2_1 = max(dh_2_1,dis)

    return min(dh_1_2,dh_2_1)

def cal_membership(lines, k_centers, dh):
    M = lines.shape[0]
    sums = np.sum(dh)
    r_cl = [M / (dh[i]**2*sum()) for i in range(len(lines))]
    return r_cl




