import cv2
import os
import numpy as np

class Image_Data_Preprocess_Hanlder:

    def get_image_name(self, path):
        filenames = os.listdir(path)
        path_filenames = []
        filename_list = []
        for file in filenames:
            if not file.startswith('.'):
                path_filenames.append(os.path.join(path, file))
                filename_list.append(file)
 
        return path_filenames

    def load_image_data(self, 
                        file_list, 
                        resize = False, # If image need to resizing to shrink the training scope
                        width = 240,  # Only work when resize = True, default value = 240
                        height = 240, # Only work when resize = True, default value = 240
                        noise_handle = False, # If the image quality is bad, can do noise hanlding firstly
                        filter_type = "median" # What kind of filter to implement to screen out the noise, 
                                               # median/mean/gaussian/bilateral/all can be selected
                       ):
        features = []
        if noise_handle & (filter_type == "median"):
            print("INFO: Implement median filter to the image")
            for file in file_list:
                img = cv2.imread(file)
                img = cv2.medianBlur(img,5) 
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')    
            return np.array(features)
                
        elif noise_handle & (filter_type == "gaussian"):
            print("INFO: Implement gaussian filter to the image")
            for file in file_list:
                img = cv2.imread(file)
                img = cv2.GaussianBlur(img,(5,5),0)
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')    
            return np.array(features)
        
        elif noise_handle & (filter_type == "mean"):
            print("INFO: Implement mean filter to the image")
            for file in file_list:
                img = cv2.imread(file)
                img = cv2.blur(img,(5,5))
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')    
            return np.array(features)
           
        elif noise_handle & (filter_type == "bilateral"):
            print("INFO: Implement bilateral filter to the image")
            for file in file_list:
                img = cv2.imread(file)
                img = cv2.bilateralFilter(img,40,75,75)
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')    
            return np.array(features)
                
        elif noise_handle & (filter_type == "all"):
            print("INFO: Implement all filters to the image")
            for file in file_list:
                img = cv2.imread(file)
                img = cv2.medianBlur(img,5)
                img = cv2.GaussianBlur(img,(5,5),0)
                img = cv2.blur(img,(5,5))
                img = cv2.bilateralFilter(img,40,75,75)
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')    
            return np.array(features)    
                
        elif not noise_handle:
            print("INFO: No filter implemented to the image")
            for file in file_list:
                img = cv2.imread(file)
                if resize:
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                img=img.flatten()
                features.append(img)
            print(82 * '_')
            return np.array(features)
            
        else:
            print("ERROR: Filter type does not exist")
            exit()
                

        
        
    def image_data_prepare(self, 
                           path, 
                           resize, 
                           width, 
                           height, 
                           noise_handle, 
                           filter_type):
        result_dict = {}
        file_list = self.get_image_name(path)
        result_dict["file_list"] = file_list
        
        data = self.load_image_data(file_list, 
                                    resize = resize,
                                    width = width, 
                                    height = height, 
                                    noise_handle = noise_handle, 
                                    filter_type = filter_type)
        result_dict["data"] = data
        
    
        return result_dict

def main(path, 
         resize = False, 
         width = 240, 
         height = 240, 
         noise_handle = False, 
         filter_type = "median"):
    image_data_preprocess_hanlder = Image_Data_Preprocess_Hanlder()
    result_dict = image_data_preprocess_hanlder.image_data_prepare(path, 
                                                                   resize, 
                                                                   width, 
                                                                   height, 
                                                                   noise_handle, 
                                                                   filter_type)
    return result_dict

if __name__ == "__main__":
    main(path = "dataset/")
    