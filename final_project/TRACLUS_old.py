import numpy as np
import matplotlib.pyplot as plt

'''
input example
    [[{'x': 0.0, 'y': 0.4624491},
      {'x': 0.4382199, 'y': 0.351921}, 
      {'x': 0.552756, 'y': 0.201239}], 
     [{'x': 0.1834925, 'y': 0.3242857}, 
      {'x': 0.2914377, 'y': 0.2762406}, 
      {'x': 0.7174087, 'y': 0.0909936}]
    ]
output example
    [
     [
      [0.0,0.42624491],
      [0.4382199,0.351921],
      [0.552766,0.201239]
     ],
     [
      [0.1834925,0.3242857],
      [0.2914377,0.2762406],
      [0.7174087,0.0909936]
     ]
    ]
'''

epsilon = 1
min_lns = 2
w_per = 1
w_par = 1
w_the = 1
line_segments = []
final_cluster = []
gama = 0.3

class Point:

    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Segment:

    def __init__(self,origin,point1: Point, point2: Point):
        self.origin = origin
        # When calculate the average vector,
        # the direction depends on the sequence of start and end.
        # So I just take the point with smaller X as start and the other as end.
        if point1.x < point2.x:
            self.start = point1
            self.end = point2
        else:
            self.start = point2
            self.end = point1

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end


def calEuclideanDistance(point1: Point, point2: Point):
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calPoint2LineDistance(point, start_point, end_point):
    k = (end_point.y - start_point.y) / (end_point.x - start_point.x)
    b = end_point.y - k * end_point.x
    return abs((k*point.x - point.y + b)/np.sqrt(k**2+1)), k

# TODO: According to the paper, this part can be directly calculated through vector!
def calThreeDistance(point1: Point,point2: Point,point3: Point,point4: Point):
    if point1 == point3 and point2 == point4:
        return 0,0,0
    elif point1 == point4 and point2 == point3:
        return 0,0,0
    length1 = calEuclideanDistance(point1,point2)
    length2 = calEuclideanDistance(point3,point4)
    if length1 < length2:
        point_is = Point(point3.x,point3.y)
        point_ie = Point(point4.x,point4.y)
        point_js = Point(point1.x,point1.y)
        point_je = Point(point2.x,point2.y)
    else:
        point_is = Point(point1.x,point1.y)
        point_ie = Point(point2.x,point2.y)
        point_js = Point(point3.x,point3.y)
        point_je = Point(point4.x,point4.y)
        temp = length1
        length1 = length2
        length2 = temp

    # compute perpendicular distance
    l_per1, k = calPoint2LineDistance(point_js,point_is,point_ie)
    l_per2, _ = calPoint2LineDistance(point_je,point_is,point_ie)
    perpendicular_distance = (l_per1**2 + l_per2**2) / (l_per1+l_per2)

    # compute parallel distance
    k_per = - 1 / k
    b_per_s = point_js.y - k_per*point_js.x
    b_per_e = point_je.y - k_per*point_je.x
    l_par1,_ = calPoint2LineDistance(point_is,point_js,Point(point_js.x+0.1,
                                                           k_per*(point_js.x+0.1)+b_per_s))
    l_par2,_ = calPoint2LineDistance(point_ie,point_je,Point(point_je.x+0.1,
                                                           k_per*(point_je.x+0.1)+b_per_e))
    parallel_distance = min(l_par1,l_par2)

    # compute angle distance
    # FIXME: This part'naming doesn't follow the paper. Don't misunderstand d_theta!
    d_theta,_ = calPoint2LineDistance(point_js,point_je,Point(point_je.x+0.1,
                                                            k_per*(point_je.x+0.1)+b_per_e))
    angle_distance = np.sqrt(length1**2 - d_theta**2)
    return perpendicular_distance, parallel_distance, angle_distance

# TODO: May optimize by vectoring computation!
def calMdlPar(points_i2j):
    l_h = np.log2(calEuclideanDistance(points_i2j[0],points_i2j[-1]))
    point_c1 = points_i2j[0]
    point_c2 = points_i2j[-1]
    per_sum = 0
    ang_sum = 0
    for i in range(len(points_i2j)-1):
        per,_,ang = calThreeDistance(point_c1,point_c2,points_i2j[i],points_i2j[i+1])
        per_sum += per
        ang_sum += ang
    if per_sum == 0 or ang_sum == 0:
        l_dh = 0
    else:
        l_dh = np.log2(per_sum) + np.log2(ang_sum)
    return l_h + l_dh

def calMdlNoPar(points_i2j):
    result = 0
    for i in range(len(points_i2j)-1):
        result += calEuclideanDistance(points_i2j[i],points_i2j[i+1])
    if result == 0:
        return 0
    else:
        return np.log2(result)

def partition(trajectorySet):
    X = []
    X_point = []
    for line in trajectorySet:
        temp = [[point['x'], point['y']] for point in line]
        X.append(temp)
        temp = [Point(point['x'],point['y']) for point in line]
        X_point.append(temp)
    X = np.array(X)

    line_cp = []
    # Approximate Trajectory Partitioning
    for line in X_point:
        characteristic_points = [line[0]]
        start_index = 0
        length = 1
        while start_index + length < len(line):
            curr_index = start_index + length
            cost_par = calMdlPar(line[start_index:curr_index+1])
            cost_no_par = calMdlNoPar(line[start_index:curr_index])
            if cost_par > cost_no_par:
                characteristic_points.append(line[curr_index-1])
                start_index = curr_index-1
                length = 1
            else:
                length += 1
        characteristic_points.append(line[-1])
        line_cp.append(characteristic_points)
    for index, line in enumerate(line_cp):
        line_segments.append([Segment(index,line[i],line[i+1]) for i in range(len(line)-1)])

def calENeighbor(segment: Segment, segments: np.ndarray):
    e_neighbor = []
    for index, s in np.ndenumerate(segments):
        if s != segment:
            d1, d2, d3 = calThreeDistance(segment.start,segment.end,s.start,s.end)
            if w_per*d1 + w_par*d2 + w_the*d3 <= epsilon:
                e_neighbor.append(index[0])
        else:
            e_neighbor.append(index[0])
    return e_neighbor

def expandCluster(queue: list, segments, cluster_id, segment_cluster, clusters):
    while len(queue) != 0:
        m = queue[0]
        e_neighbor = calENeighbor(segments[m], segments)
        if len(e_neighbor) >= min_lns:
            for x in e_neighbor:
                if segment_cluster[x] == 0:
                    queue.append(x)
                if segment_cluster[x] == 0 or segment_cluster[x] == -1:
                    segment_cluster[x] = cluster_id
                    clusters[cluster_id].append(segments[x])
        del queue[0]

# TODO: According to this project situation, the cluster number should always be one!!!
def group():
    # ATTENTION: The cluster_id is accumulating from 1
    cluster_id = 1
    segments = np.array(line_segments)
    segments = segments.reshape(1,-1)[0]
    segment_cluster = np.zeros(segments.shape)
    clusters = {1:[]}
    for index, segment in np.ndenumerate(segments):
        if segment_cluster[index[0]] == 0:
            e_neighbor = calENeighbor(segment, segments)
            if len(e_neighbor) >= min_lns:
                for i in e_neighbor:
                    clusters[cluster_id].append(segments[i])
                    segment_cluster[i] = cluster_id
                queue = [l for l in e_neighbor if l != index[0]]
                expandCluster(queue,segments,cluster_id,segment_cluster,clusters)
                cluster_id += 1
                clusters[cluster_id] = []
            else:
                segment_cluster[index[0]] = -1
    for cluster_id,cluster in clusters.items():
        if len(cluster) >= min_lns:
            final_cluster.append(cluster)

def calAverageDirectionVector(cluster):
    v = np.sum([[segment.end.x-segment.start.x,segment.end.y-segment.start.y] for segment in cluster], axis=0)
    return v / len(cluster)


def represent():
    rtr = []
    # TODO: This for-loop should finally run only once!!!
    for cluster in final_cluster:
        aver_direct_v = calAverageDirectionVector(cluster)
        v_len = np.sqrt(aver_direct_v[0]**2+aver_direct_v[1]**2)
        trans_matrix = np.array([[aver_direct_v[0]/v_len,aver_direct_v[1]/v_len],
                                 [-aver_direct_v[1]/v_len,aver_direct_v[0]/v_len]])
        transformed_cluster = []
        start_end_points = []
        for segment in cluster:
            point1 = trans_matrix.dot(np.array([[segment.start.x],[segment.start.y]]))
            point2 = trans_matrix.dot(np.array([[segment.end.x],[segment.end.y]]))
            transformed_cluster.append(Segment(segment.origin,
                                               Point(point1[0][0],point1[1][0]),
                                               Point(point2[0][0],point2[1][0])))
            start_end_points.append(Point(point1[0][0],point1[1][0]))
            start_end_points.append(Point(point2[0][0],point2[1][0]))
        points_index = [index for index, value in sorted(enumerate(start_end_points),
                                                         key=lambda p: p[1].x)]
        no_sort_points = start_end_points.copy()
        start_end_points.sort(key=lambda p: p.x)
        segment_state = [0]*len(cluster)
        num_p = 0
        rtr_i = []
        pre_index = []
        for index, point_ in enumerate(start_end_points):
            if segment_state[points_index[index]//2] == 0:
                num_p += 1
                segment_state[points_index[index]//2] = 1
                pre_index.append(points_index[index])
            elif segment_state[points_index[index]//2] == 1:
                num_p -= 1
                segment_state[points_index[index]//2] = 2
                del pre_index[0]
            if num_p >= min_lns:
                if len(rtr_i) != 0:
                    diff = calEuclideanDistance(rtr_i[len(rtr_i)-1],point_)
                    if diff >= gama:
                        ys = []
                        for i in pre_index:
                            if points_index[i]%2 == 0:
                                k = (no_sort_points[points_index[i]+1].y -
                                    no_sort_points[points_index[i]].y) / \
                                    (no_sort_points[points_index[i]+1].x-
                                     no_sort_points[points_index[i]].x)
                                b = no_sort_points[points_index[i]].y - \
                                    k*no_sort_points[points_index[i]].x
                            else:
                                k = (no_sort_points[points_index[i]].y -
                                    no_sort_points[points_index[i]-1].y) / \
                                    (no_sort_points[points_index[i]].x-
                                     no_sort_points[points_index[i]-1].x)
                                b = no_sort_points[points_index[i]].y - \
                                    k*no_sort_points[points_index[i]].x
                            ys.append(k*point_.x+b)
                        x_, y_ = point_.x, np.mean(ys)
                        aver_p = np.linalg.solve(trans_matrix,[[x_],[y_]])
                        rtr_i.append(Point(aver_p[0][0],aver_p[1][0]))
                else:
                    ys = []
                    for i in pre_index:
                        if points_index[i]%2 == 0:
                            k = (start_end_points[points_index[i]+1].y -
                                start_end_points[points_index[i]].y) / \
                                (start_end_points[points_index[i]+1].x-
                                start_end_points[points_index[i]].x)
                            b = start_end_points[points_index[i]].y - \
                                k*start_end_points[points_index[i]].x
                        else:
                            k = (start_end_points[points_index[i]].y -
                                start_end_points[points_index[i]-1].y) / \
                                (start_end_points[points_index[i]].x-
                                start_end_points[points_index[i]-1].x)
                            b = start_end_points[points_index[i]].y - \
                                k*start_end_points[points_index[i]].x
                        ys.append(k*point_.x+b)
                    x_, y_ = point_.x, np.mean(ys)
                    aver_p = np.linalg.solve(trans_matrix,[[x_],[y_]])
                    rtr_i.append(Point(aver_p[0][0],aver_p[1][0]))
        rtr.append(rtr_i)
    return rtr

trajectory = [[{'x': 0.0, 'y': 0.4624491},
               {'x': 0.4382199, 'y': 0.351921},
               {'x': 0.552756, 'y': 0.201239}],
              [{'x': 0.1834925, 'y': 0.3242857},
               {'x': 0.2914377, 'y': 0.2762406},
               {'x': 0.3756488, 'y': 0.2415424},
               {'x': 0.4622663, 'y': 0.2021603},
               {'x': 0.5345702, 'y': 0.1621667},
               {'x': 0.6215744, 'y': 0.1395555},
               {'x': 0.7174087, 'y': 0.0909936}],
              [{'x': 0.2273461, 'y': 0.4338646},
               {'x': 0.3315855, 'y': 0.2659882},
               {'x': 0.3895973, 'y': 0.199056},
               {'x': 0.5203431, 'y': 0.0562468}],
              [{'x': 0.7691297, 'y': 0.2487781},
               {'x': 0.6248497, 'y': 0.2952703},
               {'x': 0.4940669, 'y': 0.361487},
               {'x': 0.3478698, 'y': 0.4270456},
               {'x': 0.2019791, 'y': 0.4854371}],
              [{'x': 0.7678767, 'y': 0.4330209},
               {'x': 0.5879771, 'y': 0.4612372},
               {'x': 0.4268758, 'y': 0.5141916},
               {'x': 0.2826771, 'y': 0.5846225}],
              [{'x': 0.8076445, 'y': 0.2767516},
               {'x': 0.678145, 'y': 0.364564},
               {'x': 0.5498706, 'y': 0.4237074},
               {'x': 0.418986, 'y': 0.4923134},
               {'x': 0.3289207, 'y': 0.5865978}],
              [{'x': 0.2097509, 'y': 0.4067849},
               {'x': 0.3265044, 'y': 0.3591161},
               {'x': 0.4509653, 'y': 0.3117766},
               {'x': 0.5740192, 'y': 0.2715576},
               {'x': 0.6934422, 'y': 0.2646922},
               {'x': 0.8119466, 'y': 0.2793288}],
              [{'x': 1.0, 'y': 0.6798908},
               {'x': 0.9142579, 'y': 0.7504246},
               {'x': 0.8112056, 'y': 0.8130386},
               {'x': 0.6873577, 'y': 0.8460429},
               {'x': 0.5507057, 'y': 0.8689265},
               {'x': 0.4190487, 'y': 0.903991},
               {'x': 0.2937134, 'y': 0.9201781}],
              [{'x': 0.8399396, 'y': 0.3212136},
               {'x': 0.7211877, 'y': 0.3640094},
               {'x': 0.5863286, 'y': 0.3965434},
               {'x': 0.4504491, 'y': 0.4529686},
               {'x': 0.2337426, 'y': 0.5681715}],
              [{'x': 0.1717783, 'y': 0.572705},
               {'x': 0.2740729, 'y': 0.579468},
               {'x': 0.3913364, 'y': 0.5198538},
               {'x': 0.4994856, 'y': 0.4670307},
               {'x': 0.5764409, 'y': 0.3697929},
               {'x': 0.6166754, 'y': 0.2542323}],
              [{'x': 0.0893345, 'y': 0.2819685},
               {'x': 0.1771548, 'y': 0.240244},
               {'x': 0.2504575, 'y': 0.2026863},
               {'x': 0.3162571, 'y': 0.1600213},
               {'x': 0.4043839, 'y': 0.1111297},
               {'x': 0.5015232, 'y': 0.0578363},
               {'x': 0.6043721, 'y': 0.0}],
              [{'x': 0.7020681, 'y': 0.2435196},
               {'x': 0.5878008, 'y': 0.3104417},
               {'x': 0.4723522, 'y': 0.353379},
               {'x': 0.3476872, 'y': 0.4054966},
               {'x': 0.2330338, 'y': 0.4556486}]]


partition(trajectory)
group()
r = represent()
for j in range(len(r[0])-1):
    plt.plot([r[0][j].x,r[0][j+1].x],[r[0][j].y,r[0][j+1].y],c="blue")
plt.show()







