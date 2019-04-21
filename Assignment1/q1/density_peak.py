import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import queue

def calDenDis(X, dc):
    rho = np.empty(X.shape[0])
    all_distance = []
    for i in range(len(rho)):
        distance = np.sum((X[i,:]-X)**2, axis=1)
        distance = np.sqrt(distance)
        near_pt = distance[distance<dc]
        rho[i] = len(near_pt)-1
        all_distance.append(distance)
    return rho, all_distance

def calDelta(density, all_distance):
    delta = np.empty(len(density))
    nn = np.empty(len(density))
    density_index = np.argsort(-density)
    for i in range(1, len(density_index)):
        nneigh_index = 0
        nneigh_dis = 9999
        for j in range(0, i):
            if all_distance[density_index[i]][density_index[j]] < nneigh_dis:
                nneigh_index = density_index[j]
                nneigh_dis = all_distance[density_index[i]][density_index[j]]
        delta[density_index[i]] = nneigh_dis
        nn[density_index[i]] = nneigh_index
    delta[density_index[0]] = max(all_distance[density_index[0]])
    nn[density_index[0]] = -1
    return delta, nn

def setLabelHola(nn, centers, rho, hola):
    labels = np.empty(len(rho))
    for i in np.argsort(-rho):
        if i in centers:
            labels[i] = np.where(centers==i)[0][0]
        else:
            if not hola[i]:
                labels[i] = -1
            else:
                labels[i] = labels[int(nn[i])]
    return labels

def setLabel(nn, centers, rho):
    labels = np.empty(len(rho))
    for i in np.argsort(-rho):
        if i in centers:
            labels[i] = np.where(centers==i)[0][0]
        else:
            labels[i] = labels[int(nn[i])]
    return labels

def findHola(centers, all_distance, dc):
    next_queue = queue.Queue()
    hola = [0]*len(all_distance)
    for center in centers:
        next_queue.put(center)
    while not next_queue.empty():
        cur = next_queue.get()
        hola[cur] = 1
        for i, value in enumerate(all_distance[cur]):
            if hola[i] == 0 and value < dc:
                next_queue.put(i)
    return hola

def chooseDc(candidate, X):
    result = []
    num = X.shape[0]
    for dc in candidate:
        near_pt = 0
        for i in range(num):
            distance = np.sum((X[i, :] - X) ** 2, axis=1)
            distance = np.sqrt(distance)
            near_pt += len(distance[distance < dc])
        print(str(dc) + " " + str(near_pt/(num*num)))
        if 0.01*num < near_pt/num < 0.02*num:
            result.append(dc)
    return result

def predict(X, dc, NUM_OF_CENTER, plot=False, is_hola=False):
    rho, all_distance = calDenDis(X, dc)
    delta, nn = calDelta(rho, all_distance)

    centers = np.argsort(-np.multiply(rho, delta))[0:NUM_OF_CENTER]
    if is_hola:
        hola = findHola(centers, all_distance, dc)
        labels = setLabelHola(nn, centers, rho)
    else:
        labels = setLabel(nn, centers, rho)

    color = ['red','yellow','green','black','orchid','tomato','darkorchid','blue']
    if plot:
        plt.figure()
        plt.scatter(x=rho, y=delta)
        for i in range(NUM_OF_CENTER):
            plt.scatter(rho[centers[i]], delta[centers[i]], c=color[i])
        if X.shape[1] == 2:
            feature1, feature2 = X[:,0], X[:,1]

            plt.figure()
            plt.scatter(x=feature1, y=feature2)
            for i in range(NUM_OF_CENTER):
                plt.scatter(feature1[centers[i]], feature2[centers[i]], c=color[i])

            plt.figure()
            for i in range(len(labels)):
                plt.scatter(feature1[i], feature2[i], c=color[int(labels[i])])
        plt.show()
    return labels

def main():
    feature1, feature2 = np.array([]), np.array([])
    with open("Aggregation.txt") as file:
        for line in file:
            feature = line.split(",")
            feature1 = np.append(feature1, float(feature[0]))
            feature2 = np.append(feature2, float(feature[1]))

    feature1.reshape(feature1.shape[0],1)
    feature2.reshape(feature2.shape[0],1)
    X = np.array([feature1, feature2])
    X = X.T
    labels = predict(X, 1.8, 7, True)

    label_dict = {'x': feature1,
                  'y': feature2,
                  'label': labels}
    label_file = pd.DataFrame(label_dict)
    label_file.to_csv("cluster.csv")

if __name__ == '__main__':
    main()