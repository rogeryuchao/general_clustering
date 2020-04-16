import os
import shutil
import sys
from utils import data_pre_handler

def image_data_post_handler(file_list, 
                            bench_result_dict,
                            tgt_root_folder = "result/",
                            auto_selection = True, # model will auto-evluate the best cluster no and export the image into their cluster no
                            cluster_no = 2 # User should input the cluster_no when auto_selection is false
                           ):
    model_type =  bench_result_dict["model_type"]
    if auto_selection:
        for init in bench_result_dict["init"]:
            cluster_folder_class = bench_result_dict[init]["predict_result"][0][bench_result_dict[init]["best_cluster_no"]-2]
            cluster_folder = []
            for i in range(0,len(cluster_folder_class)):
                cluster_folder.append(tgt_root_folder  + model_type + "/" + init + "/" + "auto/" + str(cluster_folder_class[i]) + "/")
            mapping_table = zip(file_list, cluster_folder)
    
            for i in mapping_table:
                if not os.path.exists(i[1]):
                    os.makedirs(i[1])
                try:
                    shutil.copy(i[0], i[1])
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())

    else:
        for init in bench_result_dict["init"]:
            cluster_folder_class = bench_result_dict[init]["predict_result"][0][cluster_no - 2]
            cluster_folder = []
            for i in range(0, len(cluster_folder_class)):
                cluster_folder.append(tgt_root_folder  + 
                                      model_type + "/" + 
                                      init + "/" + 
                                      str(cluster_no) + "/" + 
                                      str(cluster_folder_class[i]) + "/")
            mapping_table = zip(file_list, cluster_folder)
    
            for i in mapping_table:
                if not os.path.exists(i[1]):
                    os.makedirs(i[1])
                try:
                    shutil.copy(i[0], i[1])
                except IOError as e:
                    print("Unable to copy file. %s" % e)
                except:
                    print("Unexpected error:", sys.exc_info())
    
    