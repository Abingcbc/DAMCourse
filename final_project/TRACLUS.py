import numpy as np
import matplotlib.pyplot as plt
import datetime

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

segment_id = 0
line_count = 0


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class EndPoint:

    def __init__(self, x, y, segment_id):
        self.x = x
        self.y = y
        self.segment_id = segment_id


class Segment:

    def __init__(self, origin, point1: Point, point2: Point):
        global segment_id
        self.id = segment_id
        segment_id += 1
        self.origin = origin
        self.k = 0
        self.b = 0
        # When calculate the average vector,
        # the direction depends on the sequence of start and end.
        # So I just take the point with smaller X as start and the other as end.
        if point1.x < point2.x:
            self.start = point1
            self.end = point2
        else:
            self.start = point2
            self.end = point1
        self.k = (self.end.y - self.start.y) / (self.end.x - self.start.x)
        self.b = self.start.y - self.k * self.start.x

    def __eq__(self, other):
        return self.start == other.start and self.end == other.end



def TRACLUS(trajectorySet):
    global segment_id
    segment_id = 0
    scoreList = []
    mintlist = []
    min_lns = 3
    epsilon = 1
    gama = 0.1
    w_per = 1
    w_par = 1
    w_the = 1
    line_segments = []
    final_cluster = []

    def calEuclideanDistance(point1: Point, point2: Point):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def calPoint2LineDistance(point, start_point, end_point):
        k = (end_point.y - start_point.y) / (end_point.x - start_point.x)
        b = end_point.y - k * end_point.x
        return abs((k * point.x - point.y + b) / np.sqrt(k ** 2 + 1)), k

    # TODO: According to the paper, this part can be directly calculated through vector!
    def calThreeDistance(point1: Point, point2: Point, point3: Point, point4: Point):
        if point1 == point3 and point2 == point4:
            return 0, 0, 0
        elif point1 == point4 and point2 == point3:
            return 0, 0, 0
        length1 = calEuclideanDistance(point1, point2)
        length2 = calEuclideanDistance(point3, point4)
        if length1 < length2:
            point_is = Point(point3.x, point3.y)
            point_ie = Point(point4.x, point4.y)
            point_js = Point(point1.x, point1.y)
            point_je = Point(point2.x, point2.y)
        else:
            point_is = Point(point1.x, point1.y)
            point_ie = Point(point2.x, point2.y)
            point_js = Point(point3.x, point3.y)
            point_je = Point(point4.x, point4.y)
            temp = length1
            length1 = length2
            length2 = temp

        # compute perpendicular distance
        l_per1, k = calPoint2LineDistance(point_js, point_is, point_ie)
        l_per2, _ = calPoint2LineDistance(point_je, point_is, point_ie)
        if l_per1 + l_per2 == 0:
            perpendicular_distance = 0
        else:
            perpendicular_distance = (l_per1 ** 2 + l_per2 ** 2) / (l_per1 + l_per2)

        # compute parallel distance
        k_per = - 1 / k
        b_per_s = point_js.y - k_per * point_js.x
        b_per_e = point_je.y - k_per * point_je.x
        l_par1, _ = calPoint2LineDistance(point_is, point_js, Point(point_js.x + 0.1,
                                                                    k_per * (point_js.x + 0.1) + b_per_s))
        l_par2, _ = calPoint2LineDistance(point_ie, point_je, Point(point_je.x + 0.1,
                                                                    k_per * (point_je.x + 0.1) + b_per_e))
        parallel_distance = min(l_par1, l_par2)

        # compute angle distance
        # FIXME: This part'naming doesn't follow the paper. Don't misunderstand d_theta!
        d_theta, _ = calPoint2LineDistance(point_js, point_je, Point(point_je.x + 0.1,
                                                                     k_per * (point_je.x + 0.1) + b_per_e))
        angle_distance = np.sqrt(length1 ** 2 - d_theta ** 2)
        return perpendicular_distance, parallel_distance, angle_distance

    # TODO: May optimize by vectoring computation!
    def calMdlPar(points_i2j):
        l_h = np.log2(calEuclideanDistance(points_i2j[0], points_i2j[-1]))
        point_c1 = points_i2j[0]
        point_c2 = points_i2j[-1]
        per_sum = 0
        ang_sum = 0
        for i in range(len(points_i2j) - 1):
            per, _, ang = calThreeDistance(point_c1, point_c2, points_i2j[i], points_i2j[i + 1])
            per_sum += per
            ang_sum += ang
        if per_sum == 0 or ang_sum == 0:
            l_dh = 0
        else:
            l_dh = np.log2(per_sum) + np.log2(ang_sum)
        return l_h + l_dh

    def calMdlNoPar(points_i2j):
        result = 0
        for i in range(len(points_i2j) - 1):
            result += calEuclideanDistance(points_i2j[i], points_i2j[i + 1])
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
            temp = [Point(point['x'], point['y']) for point in line]
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
                cost_par = calMdlPar(line[start_index:curr_index + 1])
                cost_no_par = calMdlNoPar(line[start_index:curr_index + 1])
                if cost_par > cost_no_par:
                    characteristic_points.append(line[curr_index - 1])
                    start_index = curr_index - 1
                    length = 1
                else:
                    length += 1
            characteristic_points.append(line[-1])
            line_cp.append(characteristic_points)
        for index, line in enumerate(line_cp):
            line_segments.append([Segment(index, line[i], line[i + 1]) for i in range(len(line) - 1)])

    def calENeighbor(segment: Segment, segments: np.ndarray):
        e_neighbor = []
        for index, s in np.ndenumerate(segments):
            if s != segment:
                d1, d2, d3 = calThreeDistance(segment.start, segment.end, s.start, s.end)
                if w_per * d1 + w_par * d2 + w_the * d3 <= epsilon:
                    e_neighbor.append(index[0])
            else:
                e_neighbor.append(index[0])
        return e_neighbor

    def ParameterSelectFunc(segments):
        NCountofSegs = []
        for index, seg in np.ndenumerate(segments):
            NCountofSegs.append(len(calENeighbor(seg, segments)))
        NCountofSegs = np.array(NCountofSegs)
        px2 = 0
        for count in NCountofSegs:
            if count == 0:
                px2 += 0
            else:
                px2 += (count * np.log2(count))
        return np.log2(NCountofSegs.sum()) - px2 / NCountofSegs.sum(), NCountofSegs.mean()

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

    def group():
        # ATTENTION: The cluster_id is accumulating from 1
        cluster_id = 1
        segs = []
        for line in line_segments:
            segs = segs + line
        segments = np.array(segs)
        segment_cluster = np.zeros(segments.shape)
        clusters = {1: []}
        for index, segment in np.ndenumerate(segments):
            if segment_cluster[index[0]] == 0:
                e_neighbor = calENeighbor(segment, segments)
                if len(e_neighbor) >= min_lns:
                    for i in e_neighbor:
                        clusters[cluster_id].append(segments[i])
                        segment_cluster[i] = cluster_id
                    queue = [l for l in e_neighbor if l != index[0]]
                    expandCluster(queue, segments, cluster_id, segment_cluster, clusters)
                    cluster_id += 1
                    clusters[cluster_id] = []
                else:
                    segment_cluster[index[0]] = -1
        for cluster_id, cluster in clusters.items():
            if len(cluster) >= min_lns:
                final_cluster.append(cluster)
                for s in cluster:
                    plt.plot((s.start.x,s.end.x), (s.start.y,s.end.y), c='purple')

    def calAverageDirectionVector(cluster):
        v = np.sum([[segment.end.x - segment.start.x, segment.end.y - segment.start.y] for segment in cluster], axis=0)
        return v / len(cluster)

    def represent():
        '''
        TODO: This for-loop should finally run only once!!!
        So I decide to expand all the cluster into one !!!
        '''

        all_segments = []
        for cluster in final_cluster:
            all_segments = all_segments + cluster

        aver_direct_v = calAverageDirectionVector(all_segments)
        plt.plot((0,0),(aver_direct_v[0],aver_direct_v[1]),c='red')
        v_len = np.sqrt(aver_direct_v[0] ** 2 + aver_direct_v[1] ** 2)
        trans_matrix = np.array([[aver_direct_v[0] / v_len, aver_direct_v[1] / v_len],
                                 [-aver_direct_v[1] / v_len, aver_direct_v[0] / v_len]])
        all_end_points = []
        for num, s in enumerate(all_segments):
            plt.scatter(s.start.x, s.start.y, c='yellow')
            plt.scatter(s.end.x, s.end.y, c='yellow')
            point1 = trans_matrix.dot(np.array([[s.start.x], [s.start.y]]))
            point2 = trans_matrix.dot(np.array([[s.end.x], [s.end.y]]))
            all_end_points.append(EndPoint(point1[0][0], point1[1][0], num))
            all_end_points.append(EndPoint(point2[0][0], point2[1][0], num))
            s.k = (point2[1][0] - point1[1][0]) / (point2[0][0] - point1[0][0])
            s.b = point1[1][0] - s.k * point1[0][0]
        all_end_points = sorted(all_end_points, key=lambda p: p.x)
        index = 0
        insert_list = []
        delete_list = []
        cur_active = [False] * len(all_segments)
        active_list = []
        out = []
        while index < len(all_end_points):
            insert_list[:] = []
            delete_list[:] = []
            pre_pos = all_end_points[index].x

            while index < len(all_end_points) and all_end_points[index].x - pre_pos <= gama:
                if not cur_active[all_end_points[index].segment_id]:
                    insert_list.append(all_end_points[index].segment_id)
                    cur_active[all_end_points[index].segment_id] = True
                else:
                    delete_list.append(all_end_points[index].segment_id)
                    cur_active[all_end_points[index].segment_id] = False
                index += 1

            for insert_p in insert_list:
                active_list.append(insert_p)
            if (len(out) == 0 or abs(pre_pos - out[-1].x) > gama) and \
                    len(active_list) >= min_lns:
                temp_y = 0
                temp_count = 0
                for num, active in enumerate(active_list):
                    temp_y += all_segments[active].k * pre_pos + all_segments[active].b
                    temp_count += 1
                temp_y = temp_y / temp_count
                aver_p = np.linalg.solve(trans_matrix, [[pre_pos], [temp_y]])
                out.append(Point(aver_p[0][0], aver_p[1][0]))
            for delete_p in delete_list:
                active_list.remove(delete_p)
        return out


    global line_count
    for i in range(1, 100):
        epsilon = i / 100
        partition(trajectorySet)
        segs = []
        for line in line_segments:
            segs = segs + line
        segments = np.array(segs)
        score_t, min_t = ParameterSelectFunc(segments)
        scoreList.append(score_t)
        mintlist.append(min_t)
        line_segments = []

    epsilon = (np.argsort(scoreList)[0] + 1) / 100
    gama = epsilon
    min_lns = int(mintlist[np.argsort(scoreList)[0]])

    print(str(line_count) + ": " + str(epsilon) + " " \
          + str(min_lns) + " " + str(datetime.datetime.now()))
    partition(trajectorySet)
    group()
    r = represent()

    resultPoints = []
    print("point number " + str(len(r)))
    for j in range(len(r) - 1):
        point_result = {'x': r[j].x, 'y': r[j].y}
        resultPoints.append(point_result)
        plt.plot([r[j].x, r[j + 1].x], [r[j].y, r[j + 1].y], c="blue")
    plt.show()
    resultPoints.append({'x':r[-1].x, 'y':r[-1].y})
    if len(resultPoints) == 0:
        print("empty")
    line_count += 1
    return resultPoints
