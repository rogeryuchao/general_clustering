# Data preparation parameters
SRC_IMAGE_PATH = "dataset/"

# Data preparation hyper-parameters
RESIZE_FLAG = True
RESIZE_WIDTH = 240
RESIZE_HEIGHT = 240
NOISE_HANDLE_FLAG = False
# 4 types of filters, "median"/"mean"/"gaussian"/"bilateral", or you can select "all" to implement all filters on the image
# This option is only when when NOISE_HANDLE_FLAG = True
FILTER_TYPE = "gaussian" 

# Model training hyper-parameters
MAX_CLUSTER_NO = 10 # The model will run from 2 cluster to 10 cluster
SILHOUETTE_SAMPLE_SIZE = 300 # Only used to evaluate the silhouette score, if you donot refer to it, no need to change
# Setup initializer of the Kmeans core, list type, if put three types, all types will be trained
INITIALIZER_TYPE_LIST = ["pca", "k-means++", "random"] 

# Data Post handling parameters
TGT_ROOT_PATH = "result/" # Root Folder to output the clustering result, script will create sub folder automatically
AUTO_MODEL_SELECTION = False





