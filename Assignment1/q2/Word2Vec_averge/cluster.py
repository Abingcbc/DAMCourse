import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.decomposition import PCA
from q1 import density_peak
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture

user_id = pd.read_csv("feature_word.csv").values[:,0]
X = pd.read_csv("feature_word.csv").values[:,1:]

dim = 0.01
pca = PCA(n_components=dim)
X_new = pca.fit_transform(X)
print("Silhouette Coefficient Calinski Harabaz Score")

# decide K ---- 6
sse = []
sil = []
for i in range(2,10):
    cluster_kmeans = KMeans(n_clusters=i)
    cluster_kmeans.fit(X_new)
    sse.append(cluster_kmeans.inertia_)
    sil.append(metrics.silhouette_score(X_new, cluster_kmeans.labels_))
plt.plot(range(2,10), sse, "o-")
plt.title("K-Means -- sse")
plt.show()
plt.plot(range(2,10), sil, "o-")
plt.title("K-Means -- Silhouette Coefficient")
plt.show()

# Density Peak 0.3021991689976956 14984.61603715683
print("----- Density Peak start -----")
start = time.time()
labels_dp = density_peak.predict(X_new, 0.01, 6)
class_count_dp = []
for i in range(6):
        class_count_dp.append(len(labels_dp[labels_dp==i]))
print("Time Used : " + str(time.time() - start))
sil_dp = metrics.silhouette_score(X_new, labels_dp)
cal_dp = metrics.calinski_harabaz_score(X_new,labels_dp)
print("Density Peak: " + str(sil_dp) + " " + str(cal_dp))

# K-Means 0.5222995572732575 35850.743230412285
print("----- K-Means start -----")
start = time.time()
cluster_km = KMeans(n_clusters=6)
cluster_km.fit(X_new)
labels_km = cluster_km.labels_
class_count_km = []
for i in range(6):
        class_count_km.append(len(labels_km[labels_km==i]))
print("Time Used : " + str(time.time() - start))
sil_km = metrics.silhouette_score(X_new, labels_km)
cal_km = metrics.calinski_harabaz_score(X_new, labels_km)
print("K-Means : " + str(sil_km) + " " + str(cal_km))

#DBSCAN -0.13515536027817404 207.29017961132627
print("----- DBSCAN start -----")
start = time.time()
cluster_db = DBSCAN(eps=0.009, min_samples=50)
labels_db = cluster_db.fit_predict(X_new)
class_count_db = []
for i in range(6):
        class_count_db.append(len(labels_db[labels_db==i]))
print("Time Used : " + str(time.time() - start))
sil_db = metrics.silhouette_score(X_new, labels_db)
cal_db = metrics.calinski_harabaz_score(X_new, labels_db)
print("DBSCAN : " + str(sil_db) + " " + str(cal_db))

# #Hierarchical Clustering 0.45622646692255475 30542.606776467514
print("----- Hierarchical Clustering start -----")
start = time.time()
cluster_hc = AgglomerativeClustering(n_clusters=6)
labels_hc = cluster_hc.fit_predict(X_new)
class_count_hc = []
for i in range(6):
        class_count_hc.append(len(labels_hc[labels_hc==i]))
print("Time Used : " + str(time.time() - start))
sil_hc = metrics.silhouette_score(X_new, labels_hc)
cal_hc = metrics.calinski_harabaz_score(X_new, labels_hc)
print("Hierarchical Clustering : " + str(sil_hc) + " " + str(cal_hc))

#Spectral Clustering 0.4845354412604292 16707.918523149187
print("----- Spectral Clustering start -----")
start = time.time()
cluster_sc = SpectralClustering(n_clusters=4, gamma=0.01)
labels_sc = cluster_sc.fit_predict(X_new)
class_count_sc = []
for i in range(6):
        class_count_sc.append(len(labels_sc[labels_sc==i]))
print("Time Used : " + str(time.time() - start))
sil_sc = metrics.silhouette_score(X_new, labels_sc)
cal_sc = metrics.calinski_harabaz_score(X_new, labels_sc)
print("Spectral Clustering : " + str(sil_sc) + " " + str(cal_sc))

#EM-GMM 0.5212429685773158 35723.72689810602
print("----- EM-GMM -----")
start = time.time()
cluster_em = GaussianMixture(n_components=6)
labels_em = cluster_em.fit_predict(X_new)
class_count_em = []
for i in range(6):
        class_count_em.append(len(labels_em[labels_em==i]))
print("Time Used : " + str(time.time() - start))
sil_em = metrics.silhouette_score(X_new, labels_em)
cal_em = metrics.calinski_harabaz_score(X_new, labels_em)
print("EM-GMM : " + str(sil_em) + " " + str(cal_em))

alg = ['Density Peak', 'K-Means', 'DBSCAN', 'Hierarchical Clustering',
       'Spectral Clustering', 'EM-GMM']
alg_score = [sil_dp, sil_km, sil_db, sil_hc, sil_sc, sil_em]
plt.title("Silhouette Coefficient")
plt.bar(x=alg, height=alg_score)
plt.xticks(rotation='vertical')
plt.show()

alg_score = [cal_dp, cal_km, cal_db, cal_hc, cal_sc, cal_em]
plt.title("Calinski Harabaz Score")
plt.bar(x=alg, height=alg_score)
plt.xticks(rotation='vertical')
plt.show()

print(class_count_dp)
print(class_count_km)
print(class_count_db)
print(class_count_hc)
print(class_count_sc)
print(class_count_em)

# compare_file = pd.DataFrame(data=np.array([labels_dp, labels_km,
#                                            labels_db, labels_hc,
#                                            labels_sc, labels_em
#                                            ]).T, index=user_id.astype(int),
#                             columns=alg)
# compare_file.to_csv("result.csv")
