#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:01:44 2024

@author: worklab
"""



def _fit_MM_offset(self,space_dir):
    ae_x = []
    ae_y = []
    ae_id = []
    
    file = os.path.join(space_dir,"milemarker.csv")

    # load all points
    dataframe = pd.read_csv(file)
    dataframe = dataframe[dataframe['point_id'].notnull()]
    
    mm = dataframe["milemarker"].tolist()
    st_x = dataframe["st_x"].tolist()
    st_y = dataframe["st_y"].tolist()
    
    # convert each state plane  point into roadway
    mm_space = torch.tensor([st_x,st_y,[0 for _ in range(len(st_x))]]).transpose(1,0)
    mm_space = mm_space.unsqueeze(1).expand(mm_space.shape[0],8,3)
    mm_state = self.space_to_state(mm_space)[:,:2]
    
    
    # find the appropriate offest for each coordinate
    mm_in_feet = torch.tensor(mm)*5280
    
    offset = mm_in_feet - mm_state[:,0]
    
    # pick one mm as the benchmark mm
    
    offset = offset[6].item() # currently this is mm 60
    
    return offset
        
            

def _generate_extents_file(self,im_dir,output_path = "save/cam_extents.config", mode = "rectangle"):
    """
    Produce a text file as utilized by tracking with name=xmin,xmax,ymin,ymax for each camera
    im_dir         - str - path to directory with cpkl files of attributes labeled in image coordinates
    output_path    - str - desired output .config file, defaulting to current directory
    RETURN:     None
    
    """
    
    # 1. load all extent image points into a dictionary per side
    # 2. convert all extent points into state coordinates
    # 3. Find min enclosing extents for each camera
    # 4. Look for gaps
    # 5. write extents to output file
    
    data = {}
    
    # 1. load all extent image points into a dictionary per side

    # get all cameras
    cam_data_paths = glob.glob(os.path.join(im_dir,"*.cpkl"))
    
    for cam_data_path in cam_data_paths:
        # specify path to camera imagery file
        #cam_im_path   = cam_data_path.split(".cpkl")[0] + ".png"
        camera = cam_data_path.split(".cpkl")[0].split("/")[-1]
        
        # load all points
        with open(cam_data_path, "rb") as f:
            im_data = pickle.load(f)
            
        for direction in ["EB","WB"]:
            fov_data = im_data[direction]["FOV"]
            
            
            
            if len(fov_data) > 0:
                fov_data = torch.stack([torch.tensor([item[0],item[1]]) for item in fov_data])
                data[camera + "_" + direction] = fov_data
            
            
    # 2. convert all extent points into state coordinates
    for key in data.keys():
        if key not in self.correspondence.keys():
            continue
        key_data = data[key]
        name = [key.split("_")[0] for _ in key_data]
        data[key] = self.im_to_state(key_data.float().unsqueeze(1),name = name, heights = 0, refine_heights = False)
        
    # 3. Find min enclosing extents for each camera
    
    extents = {}
    if mode == "rectangle":
        for key in data.keys():
            key_data = data[key]
            minx = torch.min(key_data[:,0]).item()
            maxx = torch.max(key_data[:,0]).item()
            miny = torch.min(key_data[:,1]).item()
            maxy = torch.max(key_data[:,1]).item()
            
            extents[key] = [minx,maxx,miny,maxy]
    
    else:
        extents = data
    
       
    # 4. Look for gaps
    if False:
        minx_total = min([extents[key][0] for key in extents.keys()])
        maxx_total = max([extents[key][1] for key in extents.keys()])
        miny_total = min([extents[key][2] for key in extents.keys()])
        maxy_total = max([extents[key][3] for key in extents.keys()])
        extents_im = np.zeros([int(maxx_total - minx_total),int(maxy_total - miny_total)]).astype(np.uint8)
        for cam_fov in extents.values():
            cv2.rectangle(extents_im,(int(cam_fov[0]),int(cam_fov[1])),(int(cam_fov[2]),int(cam_fov[3])),(255,255,0),-1)
            
        scale = extents_im.shape[0]/2000
        res = (int(extents_im.shape[0]//scale), int(extents_im.shape[1]//scale))
        extents_im = cv2.resize(extents_im,res)
        cv2.imshow("Extents",extents_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 5. write extents to output file
    keys = list(extents.keys())
    keys.sort()
    
    if mode == "rectangle":
        with open(output_path,"w",encoding='utf-8') as f:
            for key in keys:
                key_data = extents[key]
                line = "{}={},{},{},{}\n".format(key,int(key_data[0]),int(key_data[1]),int(key_data[2]),int(key_data[3]))
                f.write(line)
    else:
        
        for key in keys:
            extents[key] = extents[key][:,:2]
            extents[key] = [[int(extents[key][i,0]+ self.MM_offset),int(extents[key][i,1])] for i in range(len(extents[key]))]
        
        with open(output_path,"w") as f:    
            json.dump(extents,f, sort_keys = True)
                

def _generate_mask_images(self,im_dir,mask_save_dir = "mask"):
    cam_data_paths = glob.glob(os.path.join(im_dir,"*.cpkl"))
    for cam_data_path in cam_data_paths:
        # specify path to camera imagery file
        #cam_im_path   = cam_data_path.split(".cpkl")[0] + ".png"
        camera = cam_data_path.split(".cpkl")[0].split("/")[-1]
        
        # load all points
        with open(cam_data_path, "rb") as f:
            im_data = pickle.load(f)
            
        for direction in ["EB","WB"]:
            try:
                mask = im_data[direction]["mask"]
                
                
                
                if len(mask) == 0:
                    continue
                
                mask_im = np.zeros([2160,3840])
                
                
                
                mask_poly = (np.array([pt for pt in mask]).reshape(
                    1, -1, 2) / self.downsample).astype(np.int32)
                
                
                
                mask_im= cv2.fillPoly(
                    mask_im, mask_poly,  255, lineType=cv2.LINE_AA)
                
                save_name = os.path.join(mask_save_dir,"{}_mask.png".format(camera))
                cv2.imwrite(save_name,mask_im)
                
                mask_im = cv2.resize(mask_im,(1920,1080))
                save_name2 = os.path.join(mask_save_dir,"{}_mask_1080.png".format(camera))
                cv2.imwrite(save_name2,mask_im)
                
            except:
                pass
            
            
def _convert_landmarks(self,space_dir):
    
     output_path = "save/landmarks.json"
    
     file = os.path.join(space_dir,"landmarks.csv")
    
     # load relevant data
     dataframe = pd.read_csv(os.path.join(space_dir,file))
     st_x = dataframe["X"].tolist()
     st_y = dataframe["Y"].tolist()
     st_type = dataframe["type"].tolist()
     st_location = dataframe["location"].tolist()

     # convert all points into roadway coords

     space_data = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
     road_data = self.space_to_state(space_data)[:,:2]
     names = [st_type[i] + "_" + st_location[i] for i in range(len(st_type))]
     
     file = os.path.join(space_dir,"poles.csv")
    
     # load relevant data
     dataframe = pd.read_csv(os.path.join(space_dir,file))
     st_x = dataframe["X"].tolist()
     st_y = dataframe["Y"].tolist()
     pole = dataframe["pole-number"].tolist()
     
     space_data_pole = torch.tensor([st_x,st_y,torch.zeros(len(st_x))]).permute(1,0).unsqueeze(1)
     road_data_pole = self.space_to_state(space_data_pole)[:,:2]
     
     
     
     
     underpasses = {}
     overpasses = {}
     poles = {}
     
     for p_idx in range(len(pole)):
         p_name = pole[p_idx]
         poles[p_name] = [road_data_pole[p_idx,0].item(), road_data_pole[p_idx,1].item()]
     
     for n_idx in range(len(names)):
         name = names[n_idx]
         
         if "under" in name:
             try:
                 underpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
             except:
                 underpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]
     
         if "over" in name:
             try:
                 overpasses[name.split("_")[2]].append(road_data[n_idx,0].item())
             except:
                 overpasses[name.split("_")[2]] = [road_data[n_idx,0].item()]   
     
     pass
     # store as JSON of points
     
     landmarks = {"overpass":overpasses,
                  "underpass":underpasses,
                  "poles":poles
                  }
     
     with open(output_path,"w") as f:    
         json.dump(landmarks,f, sort_keys = True)