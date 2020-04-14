from utils import data_pre_handler, data_post_handler
from model import kmeans
from config import SRC_IMAGE_PATH, RESIZE_FLAG, RESIZE_WIDTH, RESIZE_HEIGHT, NOISE_HANDLE_FLAG, FILTER_TYPE,\
MAX_CLUSTER_NO, SILHOUETTE_SAMPLE_SIZE, INITIALIZER_TYPE_LIST, TGT_ROOT_PATH, AUTO_MODEL_SELECTION

# Data Preparation
result_dict = data_pre_handler.main(path = SRC_IMAGE_PATH, 
                                    resize = RESIZE_FLAG,
                                    width = RESIZE_WIDTH,
                                    height = RESIZE_HEIGHT,
                                    noise_handle = NOISE_HANDLE_FLAG,
                                    filter_type = FILTER_TYPE)
data = result_dict["data"]
file_list = result_dict["file_list"]

# Model training
kmeans_cluster = kmeans.Kmeans_Cluster(max_clusters = MAX_CLUSTER_NO,
                                       sample_size = SILHOUETTE_SAMPLE_SIZE,
                                       init_list = INITIALIZER_TYPE_LIST
                                      )
model_kpi = kmeans_cluster.train_model(data)

if AUTO_MODEL_SELECTION:
    # Prediction(Auto select best model based on inertia)
    data_post_handler.image_data_post_handler(file_list = file_list,
                                              bench_result_dict = model_kpi)
else:
    for cluster_no in range(2,MAX_CLUSTER_NO):
        data_post_handler.image_data_post_handler(file_list = file_list,
                                                  bench_result_dict = model_kpi,
                                                  tgt_root_folder = TGT_ROOT_PATH,
                                                  auto_selection = AUTO_MODEL_SELECTION,
                                                  cluster_no = cluster_no)
