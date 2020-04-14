from time import time
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


class Kmeans_Cluster():
    # The cluster method will loop from n_clusters = 2 to n_clusters = max_clusters to evaluate 
    # which n_clusters(k) is suitable best for the model
    # The initializor init need to be defined to determine the barycenter of each of k clusters， default is 'k-means++'. Method include:
    # 1. k-means++
    # 2. random
    # 3. pca: pca dimentionality reduction data
    def __init__(self, max_clusters, sample_size, init_list = ["k-means++"]):
        super(Kmeans_Cluster, self).__init__()
        self.__max_clusters = max_clusters
        self.__init_list = init_list
        self.__sample_size = sample_size
    
    # Only PCA method need to input data firstly to do dimentionality reduction 
    def estimator_builder(self, 
                          init, 
                          n_clusters, 
                          data = None):
        if init == "k-means++":
            estimator = KMeans(init="k-means++", n_clusters=n_clusters, n_init=10)
            return estimator
        elif init == "random":
            estimator = KMeans(init="random", n_clusters=n_clusters, n_init=10)
            return estimator
        elif init == "pca":
            pca = PCA(n_components=n_clusters).fit(data)
            estimator = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)
            return estimator
        else:
            print("ERROR: Initializer does not exist.")
            exit()
            
        

    def bench_k_means(self, 
                      init, 
                      data, 
                      n_clusters,
                      sample_size,
                      model_type = "cluster", 
                      labels = None):
        t0 = time()
        estimator = self.estimator_builder(init = init, 
                                           n_clusters = n_clusters, 
                                           data = data)
        estimator.fit(data)
        inertia = estimator.inertia_
        silhouette_score = metrics.silhouette_score(data, estimator.labels_,
                                                    metric='euclidean',
                                                    sample_size=sample_size)
        if model_type == 'cluster':
            print('%-9s\t%.2fs\t%i\t%i\t%.3f'% (init,
                                                (time() - t0),
                                                n_clusters,
                                                inertia,
                                                silhouette_score))
            
        elif model_type == 'classifier': 
            print('%-9s\t%.2fs\t%i\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'% (init,
                                                                              (time() - t0),
                                                                              n_clusters,
                                                                              inertia,
                                                                              metrics.homogeneity_score(labels, estimator.labels_),
                                                                              metrics.completeness_score(labels, estimator.labels_),
                                                                              metrics.v_measure_score(labels, estimator.labels_),
                                                                              metrics.adjusted_rand_score(labels, estimator.labels_),
                                                                              metrics.adjusted_mutual_info_score(labels,estimator.labels_),
                                                                              silhouette_score))
            print(82 * '_')
        return estimator, inertia, silhouette_score  

            
    def train_model(self, data):
                
        
        max_clusters = self.__max_clusters
        init_list = self.__init_list
        sample_size = self.__sample_size
        
        bench_result_dict = {} 
        bench_result_dict["init"] = []
        
        for init in init_list:
            n_clusters_list = []
            inertia_list = []
            silhouette_score_list = []
            predict_result_list = []
            print("INFO: Start model benchmark procedure with initializer:" + init)
            print(82 * '_')
            for n_clusters in range(2,max_clusters):
                estimator, inertia, silhouette_score = self.bench_k_means(data = data, 
                                                               init = init, 
                                                               n_clusters = n_clusters,
                                                               sample_size = sample_size,
                                                               model_type = "cluster")
                # Every key parameter should be seperately saved
                n_clusters_list.append(n_clusters)
                inertia_list.append(inertia)
                silhouette_score_list.append(silhouette_score)
                predict_result_list.append(estimator.predict(data))
                
            # Start to put the basic data into the dict
            # Model type info
            bench_result_dict["model_type"] = "kmeans"
            # Initializer Info
            bench_result_dict["init"].append(init)
            # Record sub-info under different initializer
            bench_result_dict[init] =  {}
            # Cluster number mapping info
            bench_result_dict[init]["n_clusters"] = []
            bench_result_dict[init]["n_clusters"].append(n_clusters_list)
            # Inertia Score(Major for Kmeans clustering)
            bench_result_dict[init]["inertia"] = []
            bench_result_dict[init]["inertia"].append(inertia_list)
            # Silhouette score(Reference for Kmeans clustering)
            bench_result_dict[init]["silhouette_score"] = []
            bench_result_dict[init]["silhouette_score"].append(silhouette_score_list)
            # Preduct Clustering Result for the all pictures
            bench_result_dict[init]["predict_result"] = []
            bench_result_dict[init]["predict_result"].append(predict_result_list)
            # Estimate inertia curvity to evaluate best n cluster, 1st/last point cannot calculate curvity, so append 0
            bench_result_dict[init]['inertia_curvity'] = []
            bench_result_dict[init]['inertia_curvity'].append(0)
            for i in range(0,len(bench_result_dict[init]["inertia"][0])-2):
                inertia_curvity = abs((bench_result_dict[init]["inertia"][0][i+2] -  
                                      bench_result_dict[init]["inertia"][0][i+1])/
                                     (bench_result_dict[init]["inertia"][0][i+1]-  
                                      bench_result_dict[init]["inertia"][0][i]))
                bench_result_dict[init]['inertia_curvity'].append(inertia_curvity)
            bench_result_dict[init]['inertia_curvity'].append(0)
            # Best cluster no is decided by the highest curvity position
            max_curvity_index = bench_result_dict[init]['inertia_curvity'].index(max(bench_result_dict[init]['inertia_curvity']))
            best_cluster_no = bench_result_dict[init]["n_clusters"][0][max_curvity_index]
            bench_result_dict[init]['best_cluster_no'] = []
            bench_result_dict[init]['best_cluster_no'] = best_cluster_no
            print(82 * '_')
        return bench_result_dict
        
                