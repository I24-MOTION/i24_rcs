"""
This file supercedes all older versions of homography and coordinate system. It defines an 
object for performing image to state plane conversions via homography and state plane
to roadway coordinate system conversions via spline curvilinear coordinate system conversion.

This object implements only the usage functions - fitting functions are contained in other files
"""


import os
import _pickle as pickle
import pandas as pd
import numpy as np
import torch
import glob
import cv2
import time
import string
import re
import copy
import sys
import json
import matplotlib.pyplot as plt

from scipy import interpolate


class RCS:
    def safe_name(func):
        """
        Wrapper function, catches camera names that aren't capitalized 
        """
        
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError:
                #print(args,kwargs)
                if type(kwargs["name"]) == list:
                    kwargs["name"] = [item.upper() for item in kwargs["name"]]
                elif type(kwargs["name"]) == str:
                    kwargs["name"] = kwargs["name"].upper()
                return func(*args, **kwargs)
        return new_func
    
    
    def __init__(self, 
                 homography_dir = None,
                 rcs_file = None,
                 cache_file = None,
                 downsample = 1
                 ):
        
        """
        Initializes RCS object. There are two options:
        
        
        homography_dir - None or str - if str, specifies directory with one file per camera eg hg_P40_C05.cpkl
        rcs_file       - None or str - path to file with stored parameters for curvilinear coordinate system spline
        
        or 
        
        cache_file     - None or str - a save file produced by an RCS object that directly caches all info in a single file
        """
        
        # intialize correspondence
        self.downsample = downsample 
        self.polarity = 1
        self.MM_offset = 0

        self.correspondence = {}
        if cache_file is not None and os.path.exists(save_file):
            with open(cache_file,"rb") as f:
                # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets = pickle.load(f)
            
        
                
        
        elif homography_dir is None or rcs_file is None:
            raise IOError("Either save_file or space_dir and im_dir must not be None")
        
        else:
            # load rcs 
            with open(rcs_file,"rb") as f:
                self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets = pickle.load(f)
            
            
        # load each correspondence


        # object class info doesn't really belong in homography but it's unclear
        # where else it should go, and this avoids having to pass it around 
        # for use in height estimation
        self.class_dims = {
                "sedan":[16,6,4],
                "midsize":[18,6.5,5],
                "van":[20,6,6.5],
                "pickup":[20,6,5],
                "semi":[55,9,14],
                "truck (other)":[25,9,14],
                "truck": [25,9,14],
                "motorcycle":[7,3,4],
                "trailer":[16,7,3],
                "other":[18,6.5,5]
            }
        
        self.class_heights = dict([(key,self.class_dims[key][2]) for key in self.class_dims.keys()])
        
        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer"
                    }
    
        
        
        
        
    def save(self,save_file):
        "Caches entire homography data in one file for easy reloading"
        with open(save_file,"wb") as f:
            pickle.dump([self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets],f)

#%% Utility functions
def line_to_point(line,point):
    """
    Given a line defined by two points, finds the distance from that line to the third point
    line - (x0,y0,x1,y1) as floats
    point - (x,y) as floats
    Returns
    -------
    distance - float >= 0
    """
    
    numerator = np.abs((line[2]-line[0])*(line[1]-point[1]) - (line[3]-line[1])*(line[0]-point[0]))
    denominator = np.sqrt((line[2]-line[0])**2 +(line[3]-line[1])**2)
    
    return numerator / (denominator + 1e-08)




class Curvilinear_Homography():
    
    def safe_name(func):
        """
        Wrapper function, catches camera names that aren't capitalized 
        """
        
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except KeyError:
                #print(args,kwargs)
                if type(kwargs["name"]) == list:
                    kwargs["name"] = [item.upper() for item in kwargs["name"]]
                elif type(kwargs["name"]) == str:
                    kwargs["name"] = kwargs["name"].upper()
                return func(*args, **kwargs)
        return new_func
    
    #%% Initialization and Setup Functions
    
    
    
    """
    
    3 coordinate systems are utilized in Curvilinear_Homography:
        -image coordinates
        - space coordinates (state plane coordinates) in feet
        - roadway coordianates / curvilinear coordinates in feet
    
    After fitting, each value of self.correspondence contains:
        H     - np array of size [3,3] used for image to space perspective transform
        H_inv - used for space to image perspective transform on ground plane
        P     - np array of size [3,4] used for space to image transform
        corr_pts - list of [x,y] points in image space that are fit for transform
        space_pts - corresponding list of [x,y] points in space (state plane coordinates in feet)
        state_plane_pts - same as space_pts but [x,y,id] (name is included)
        vps - z vanishing point [x,y] in image coordinates
        extents - xmin,xmax,ymin,ymax in roadway coordinates
        extents_space - list of array of [x,y] points defining boundary in state plane coordinates
        
    """
    
    
    def __init__(self, 
                 save_file = None,
                 space_dir = None,
                 im_dir = None,
                 downsample = 1,
                 fill_gaps = False):
        """
        Initializes homography object.
        
        save_file - None or str - if str, specifies path to cached homography object
        space_dir - None or str - path to directory with csv files of attributes labeled in space coordinates
        im_dir    - None or str - path to directory with cpkl files of attributes labeled in image coordinates
        """
        
        # intialize correspondence
        self.downsample = downsample 
        self.polarity = 1
        self.MM_offset = 0

        self.correspondence = {}
        if save_file is not None and os.path.exists(save_file):
            with open(save_file,"rb") as f:
                # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets = pickle.load(f)
            
            # reload  parameters of curvilinear axis spline
            # rather than the spline itself for better pickle reloading compatibility
                
        
        elif space_dir is None or im_dir is None:
            raise IOError("Either save_file or space_dir and im_dir must not be None")
        
        else:
            #  fit the axis spline once and collect extents
            self.generate(space_dir,im_dir)
            self.median_tck = None
            self.median_u = None
            self.guess_tck = None
            self.guess_tck2 = None
            self.all_splines = None
            self.yellow_offsets = None
            self.save(save_file)
            
        if False or self.median_tck is None:
            self._fit_spline(space_dir)
            self.save(save_file)
        self.save_file = save_file
        #self._fit_spline(space_dir)

        # object class info doesn't really belong in homography but it's unclear
        # where else it should go, and this avoids having to pass it around 
        # for use in height estimation
        self.class_dims = {
                "sedan":[16,6,4],
                "midsize":[18,6.5,5],
                "van":[20,6,6.5],
                "pickup":[20,6,5],
                "semi":[55,9,14],
                "truck (other)":[25,9,14],
                "truck": [25,9,14],
                "motorcycle":[7,3,4],
                "trailer":[16,7,3],
                "other":[18,6.5,5]
            }
        
        self.class_heights = dict([(key,self.class_dims[key][2]) for key in self.class_dims.keys()])
        
        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer"
                    }
        
        if fill_gaps:
            self.fill_gaps()
        
    def save(self,save_file):
        with open(save_file,"wb") as f:
            pickle.dump([self.correspondence,self.median_tck,self.median_u,self.guess_tck,self.guess_tck2,self.MM_offset,self.all_splines,self.yellow_offsets],f)
        
        
    def generate(self,
                 space_dir,
                 im_dir,
                 downsample     = 1,
                 max_proj_error = 0.25,
                 scale_factor   = 3,
                 ADD_PROJ       = False,
                 USE_SPLINES    = False):
        """
        Loads all available camera homographies from the specified paths.
        after running, self.correspondence is a dict with one key for each <camera>_<direction>
        
        space_dir      - str - path to directory with csv files of attributes labeled in space coordinates
        im_dir         - str - path to directory with cpkl files of attributes labeled in image coordinates
        downsample     - int - specifies downsampling ratio for image coordinates
        max_proj_error - float - max allowable positional error (ft) between point and selected corresponding point on spline, 
                                 lower will exclude more points from homography computation
        scale_factor   - float - sampling frequency (ft) along spline, lower is slower but more accurate
        ADD_PROJ       - bool - if true, compute points along yellow line to use in homography
        """
        
        print("Generating homography")
        
        ae_x = []
        ae_y = []
        ae_id = []
        for direction in ["EB","WB"]:
            ### State space, do once
            
    
            
            for file in os.listdir(space_dir):
                if direction.lower() not in file:
                    continue
                
                # load all points
                dataframe = pd.read_csv(os.path.join(space_dir,file))
                try:
                    dataframe = dataframe[dataframe['point_pos'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["point_id"].tolist()
                    st_id = [attribute_name + "_" + item for item in feature_idx]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
                except:
                    dataframe = dataframe[dataframe['side'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["id"].tolist()
                    side        = dataframe["side"].tolist()
                    st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
            
            
            # Find a-d end point of all d2 lane markers
            d2 = {}
            d3 = {}
            
            ae_spl_x = []
            ae_spl_y = []
            
            for i in range(len(ae_x)):
                if "d2" in ae_id[i]:
                    if ae_id[i].split("_")[-1] in ["a","d"]:
                        num = ae_id[i].split("_")[-2]
                        if num not in d2.keys():
                            d2[num] = [(ae_x[i],ae_y[i])]
                        else:
                            d2[num].append((ae_x[i],ae_y[i]))
                elif "d3" in ae_id[i]:
                    if ae_id[i].split("_")[-1] in ["a","d"]:
                        num = ae_id[i].split("_")[-2]
                        if num not in d3.keys():
                            d3[num] = [(ae_x[i],ae_y[i])]
                        else:
                            d3[num].append((ae_x[i],ae_y[i]))
                            
                elif "yeli" in ae_id[i]:
                    ae_spl_x.append(ae_x[i])
                    ae_spl_y.append(ae_y[i])
                
                    
            
            # stack d2 and d3 into arrays            
            d2_ids = []
            d2_values = []
            for key in d2.keys():
                val = d2[key]
                d2_ids.append(key)
                d2_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
            
            d3_ids = []
            d3_values = []
            for key in d3.keys():
                val = d3[key]
                d3_ids.append(key)
                d3_values.append(   [(val[0][0] + val[1][0])/2.0   ,   (val[0][1] + val[1][1])/2.0     ])
            
            d2_values = torch.from_numpy(np.stack([np.array(item) for item in d2_values]))
            d3_values = torch.from_numpy(np.stack([np.array(item) for item in d3_values]))
            
            d2_exp = d2_values.unsqueeze(1).expand(d2_values.shape[0],d3_values.shape[0],2)
            d3_exp = d3_values.unsqueeze(0).expand(d2_values.shape[0],d3_values.shape[0],2)
            
            dist = torch.sqrt(torch.pow(d2_exp - d3_exp , 2).sum(dim = -1))
            
            min_matches = torch.min(dist, dim = 1)[1]
            
            # if ADD_PROJ:
                
            #     try:
            #         with open("ae_cache_{}.cpkl".format(direction),"rb") as f:
            #             additional_points = pickle.load(f)
            #     except:
            #         # For each d2 lane marker, find the closest d3 lane marker
            #         proj_lines = []
                    
            #         for i in range(len(min_matches)):
            #             j = min_matches[i]
            #             pline = [d3_values[j],d2_values[i],d3_ids[j],d2_ids[i]]
            #             proj_lines.append(pline)
                    
                    
                    
            #         # compute the yellow line spline in state plane coordinates
                    
            #         ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
            #         ae_data = ae_data[:,np.argsort(ae_data[1,:])]
                    
            #         ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
                    
            #         span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
            #         ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*scale_factor)), ae_tck)
                
            #         additional_points = []
            #         # for each d2 lane marker, find the intersection between the d2-d3 line and the yellow line spline
            #         for p_idx, proj_line in enumerate(proj_lines):
            #             print("On proj line {} of {}".format(p_idx,len(proj_lines)))
            #             min_dist = np.inf
            #             min_point = None
            #             line = [proj_line[0][0],proj_line[0][1],proj_line[1][0],proj_line[1][1]]
            #             for i in range(len(ae_x_prime)):
            #                 point = [ae_x_prime[i],ae_y_prime[i]]
                            
            #                 dist = line_to_point(line, point)
            #                 if dist < min_dist:
            #                     min_dist = dist
            #                     min_point = point
            #             if min_dist > max_proj_error:
            #                 print("Issue")
            #             else:
            #                 name = "{}_{}".format(proj_line[2],proj_line[3])
            #                 min_point.append(name)
            #                 additional_points.append(min_point)
                            
            #         with open("ae_cache_{}.cpkl".format(direction),"wb") as f:
            #             pickle.dump(additional_points,f)
                        
                
            #     for point in additional_points:
            #         ae_x.append(point[0])
            #         ae_y.append(point[1])
            #         ae_id.append(point[2])
    
        ### Shift aerial points using splines
        if USE_SPLINES: 
            ae_x,ae_y,ae_id = self.shift_aerial_points(ae_x,ae_y,ae_id)
    
        # get all cameras
        cam_data_paths = glob.glob(os.path.join(im_dir,"*.cpkl"))
        
        for cam_data_path in cam_data_paths:
            
            
            # specify path to camera imagery file
            cam_im_path   = cam_data_path.split(".cpkl")[0] + ".png"
            camera = cam_data_path.split(".cpkl")[0].split("/")[-1]
            
            # if "46" in camera or "47" in camera or "48" in camera:
            #     print("excluded validation system pole")
            #     continue
            
            # load all points
            with open(cam_data_path, "rb") as f:
                im_data = pickle.load(f)
                
            for direction in ["EB","WB"]:
                # get all non-curve matching points
                try:
                    point_data = im_data[direction]["points"]
                except KeyError:
                    continue
                filtered = filter(lambda x: x[2].split("_")[1] not in ["yeli","yelo","whli","whlo"],point_data)
                im_x  = []
                im_y  = []
                im_id = []
                for item in filtered:
                    im_x.append(item[0])
                    im_y.append(item[1])
                    im_id.append(item[2])
                
                if len(im_x) == 0:
                    continue
                
                if ADD_PROJ:
                
                    # compute the yellow line spline in image coordinates
                    curve_data = im_data[direction]["curves"]
                    filtered = filter(lambda x: "yeli" in x[2], curve_data)
                    x = []
                    y = []
                    for item in filtered:
                        x.append(item[0])
                        y.append(item[1])
                    data = np.stack([np.array(x),np.array(y)])
                    data = data[:,np.argsort(data[0,:])]
                    
                    tck, u = interpolate.splprep(data, s=0, per=False)
                    x_prime, y_prime = interpolate.splev(np.linspace(0, 1, 4000), tck)
                    
                    if False:
                        im = cv2.imread(cam_im_path)
                        for i in range(len(x_prime)):
                            cv2.circle(im,(int(x_prime[i]),int(y_prime[i])), 2, (255,0,0,),-1)
                            
                        cv2.imshow("frame",im)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                    # find all d2 and d3 points
                
                    # find the intersection of each d2d3 line and the yellow line spline for d2-d3 pairs in image
                    d2 = {}
                    d3 = {}
                    for i in range(len(im_x)):
                        if "d2" in im_id[i]:
                            if im_id[i].split("_")[-1] in ["a","d"]:
                                num = im_id[i].split("_")[-2]
                                if num not in d2.keys():
                                    d2[num] = [(im_x[i],im_y[i])]
                                else:
                                    d2[num].append((im_x[i],im_y[i]))
                        elif "d3" in im_id[i]:
                            if im_id[i].split("_")[-1] in ["a","d"]:
                                num = im_id[i].split("_")[-2]
                                if num not in d3.keys():
                                    d3[num] = [(im_x[i],im_y[i])]
                                else:
                                    d3[num].append((im_x[i],im_y[i]))
                        
                
                    # stack d2 and d3 into arrays            
                    d2_ids = []
                    d2_values = []
                    for key in d2.keys():
                        d2[key] = [(d2[key][0][0] + d2[key][1][0])/2.0  , (d2[key][0][1] + d2[key][1][1])/2.0 ] 
                    
                    d3_ids = []
                    d3_values = []
                    for key in d3.keys():
                        d3[key] = [(d3[key][0][0] + d3[key][1][0])/2.0  , (d3[key][0][1] + d3[key][1][1])/2.0 ] 
                    
                    additional_im_points = []
                    for proj_point in additional_points:
                        
                        
                        d3_id = proj_point[2].split("_")[0]
                        d2_id = proj_point[2].split("_")[1]
                        
                        if d3_id not in d3.keys() or d2_id not in d2.keys():
                            continue
                        
                        im_line = [d3[d3_id][0], d3[d3_id][1], d2[d2_id][0], d2[d2_id][1]]
                        
                        min_dist = np.inf
                        min_point = None
                        for i in range(len(x_prime)):
                            point = [x_prime[i],y_prime[i]]
                
                            dist = line_to_point(im_line, point)
                            if dist < min_dist:
                                min_dist = dist
                                min_point = point
                        if min_dist > 2:
                            print("Issue")
                        else:
                            name = proj_point[2]
                            min_point.append(name)
                            additional_im_points.append(min_point)
                            
                    for point in additional_im_points:
                        im_x.append(point[0])
                        im_y.append(point[1])
                        im_id.append(point[2])
                    
                        
                
                    
                ### Joint
                
                # assemble ordered list of all points visible in both image and space
                
                include_im_x  = []
                include_im_y  = []
                include_im_id = []
                
                include_ae_x  = []
                include_ae_y  = []
                include_ae_id = []
                
                for i in range(len(ae_id)):
                    for j in range(len(im_id)):
                        
                        if ae_id[i] == im_id[j]:
                            include_im_x.append(  im_x[j])
                            include_im_y.append(  im_y[j])
                            include_im_id.append(im_id[j])
                            
                            include_ae_x.append(  ae_x[i])
                            include_ae_y.append(  ae_y[i])
                            include_ae_id.append(ae_id[i])
                
                
                
                # compute homography
                vp = im_data[direction]["z_vp"]
                corr_pts = np.stack([np.array(include_im_x),np.array(include_im_y)]).transpose(1,0)
                space_pts = np.stack([np.array(include_ae_x),np.array(include_ae_y)]).transpose(1,0)
                
                if len(corr_pts) < 4 or len(space_pts) < 4:
                    continue
                
                cor = {}
                #cor["vps"] = vp
                cor["corr_pts"] = corr_pts
                cor["space_pts"] = space_pts
                
                cor["H"],_     = cv2.findHomography(corr_pts,space_pts)
                cor["H_inv"],_ = cv2.findHomography(space_pts,corr_pts)
                
                
                # P is a [3,4] matrix 
                #  column 0 - vanishing point for space x-axis (axis 0) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 1 - vanishing point for space y-axis (axis 1) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 2 - vanishing point for space z-axis (axis 2) in image coordinates (im_x,im_y,im_scale_factor)
                #  column 3 - space origin in image coordinates (im_x,im_y,scale_factor)
                #  columns 0,1 and 3 are identical to the columns of H, 
                #  We simply insert the z-axis column (im_x,im_y,1) as the new column 2
                
                P = np.zeros([3,4])
                P[:,0] = cor["H_inv"][:,0]
                P[:,1] = cor["H_inv"][:,1]
                P[:,3] = cor["H_inv"][:,2] 
                P[:,2] = np.array([vp[0],vp[1],1])  * 10e-09
                cor["P"] = P
        
                self._fit_z_vp(cor,im_data,direction)
                
                cor["state_plane_pts"] = [include_ae_x,include_ae_y,include_ae_id]
                cor_name = "{}_{}".format(camera,direction)
                self.correspondence[cor_name] = cor
        
            # use other side if no homography defined
            if True:
                if "{}_{}".format(camera,"EB") not in self.correspondence.keys():
                    if "{}_{}".format(camera,"WB") in self.correspondence.keys():
                        self.correspondence["{}_{}".format(camera,"EB")] = self.correspondence["{}_{}".format(camera,"WB")]
                if "{}_{}".format(camera,"WB") not in self.correspondence.keys():
                    if "{}_{}".format(camera,"EB") in self.correspondence.keys():
                        self.correspondence["{}_{}".format(camera,"WB")] = self.correspondence["{}_{}".format(camera,"EB")]
    
    def shift_aerial_points(self,ae_x,ae_y,ae_id):
        """
        Moves each aerial imagery point such that it is within 1/2 lane width distance of the center-of-line spline.
        This is done to account for labeling noise and remove jump discontinuities in aerial image points.
        
        
        1. Sample each center line spline at fine (0.02 ft intervals)
        2. Group all points into sets (i.e. corresponding to the same tick)
        3. Find the midpoint for each tick
        4. Find the closest sampled point and compute the offset
        5. Shift each point by that amount
        - Ignore yellow lines, only compute for lanes 1-3 ticks
        
        """
    
        new_ae_x = []
        new_ae_y = []
        new_ae_id = []
        
        #1. Sample each spline
        sampled_splines = {}
        for key in self.all_splines.keys():
            if "center" in key:
                spl = self.all_splines[key]
                
                umin,umax = np.min(spl[1]),np.max(spl[1])
                diff = umax-umin
                # 50 means 50 samples per foot
                u_range = np.linspace(umin,umax,int(diff*50))
            
                sample_x,sample_y = interpolate.splev(u_range,spl[0])
                
                data = np.stack([sample_x,sample_y])
                sampled_splines[key] = data
        
        #2. Group all points
        points = {}
        for i in range(len(ae_id)):
            base_id = ae_id[i].split("_")[0] + ae_id[i].split("_")[1] + ae_id[i].split("_")[2]
            
            try:
                points[base_id].append([ae_x[i],ae_y[i],ae_id[i]])
            except:
                points[base_id] = [[ae_x[i],ae_y[i],ae_id[i]]]
        
        ## Iterate through all sets of points
        for key in points.keys():
            if "d1" not in key and "d2" not in key and "d3" not in key or len(points[key]) != 4:
                pt = points[key]
                for i in range(len(pt)):
                    new_ae_x.append(pt[i][0])
                    new_ae_y.append(pt[i][1])
                    new_ae_id.append(pt[i][2])
                continue
            
            
            #3. get midpoint
            x_list  = [points[key][i][0] for i in range(4)]
            y_list  = [points[key][i][1] for i in range(4)]
            id_list = [points[key][i][2] for i in range(4)]

                
            x_mean = sum(x_list)/4.0
            y_mean = sum(y_list)/4.0
            
            #find closest point on spline
            
            # match to spline by direction and dash index
            direction = id_list[0].split("_")[0].upper()
            dash      = id_list[0].split("_")[1]
            spline_key = "{}_{}_center".format(direction,dash)
            
            spl = sampled_splines[spline_key]
            
            point = np.array([[x_mean],[y_mean]]).repeat(spl.shape[1],axis = 1)
            dist = ((spl - point)**2).sum(axis = 0)
            min_idx = np.argmin(dist)
            
            min_spl_x, min_spl_y = spl[0,min_idx],spl[1,min_idx]
            
            x_disp = min_spl_x - x_mean
            y_disp = min_spl_y - y_mean
            
            # 5. Shift points so that center of dash lies on spline
            pt = points[key]
            for i in range(4):
                new_ae_x.append(pt[i][0]+x_disp)
                new_ae_y.append(pt[i][1]+y_disp)
                new_ae_id.append(pt[i][2])
        
        print(len(new_ae_x),len(ae_x))
        
        return new_ae_x,new_ae_y,new_ae_id
        
    
    
    def shift_aerial_points2(self,ae_x,ae_y,ae_id):
        """
        So, as it turns out, it is much easier to smooth a spline in roadway coordinates than in state plane coordinates. 
        We want to use the smoothed splines to shift points, so we adopt the following logic
        1. For each spline (inside and outside, y-d3)
        2. Get all points and sort by increasing x (state plane)
        3. convert to roadway coordinates (spline and points)
        4. Smooth spline
        5. For each point, find closest spline point 
        5. Convert to state plane
        6. Reassign each point the smoothed value
        
        7. At the end, output smoothed values
        
        This is written is such a way that we can call it in fit_spline as:
            
            if self.median_tck is not None:
                ae_x,ae_y,ae_id = shift_aerial_points2(ae_x,ae_y_ae_id)
                
                ... fit splines
                
        and then simply call self.fit_spline() twice
         
        NOTE - we smooth all points according to the inside spline fit, ignoring the outside spline fit
      
        
        """
        spline_sample_frequency = 1
    
        new_ae_x = []
        new_ae_y = []
        new_ae_id = []
        
        # get key from ae_id
        indicator = ae_id[0].split("_")[1]
        direction = ae_id[0].split("_")[0].upper()
        
        
        if "yel" in indicator:
            indicator = "yel"
            
        for side in ["i","o"]:
        
            #1. Sample appropriate spline
            for key in self.all_splines.keys():
                if indicator in key and direction in key and side in key:
                    
                    spl = self.all_splines[key]
                    
                    umin,umax = np.min(spl[1]),np.max(spl[1])
                    diff = umax-umin
                    # 50 means 50 samples per foot
                    u_range = np.linspace(umin,umax,int(diff*spline_sample_frequency))
                
                    sample_x,sample_y = interpolate.splev(u_range,spl[0])
                    
                    spline_data = np.stack([sample_x,sample_y]).transpose(1,0)
                    spline_data = torch.from_numpy(spline_data)
                    zeros = torch.zeros([spline_data.shape[0],1])
                    spline_data = torch.cat([spline_data,zeros],dim = 1)
                    break
            
            try: 
                spline_data
            except: 
                print("error getting correct spline key for {} {} {}".format(indicator,direction,side))
                return ae_x,ae_y,ae_id
            
            
        
            #2. Get the subset of relevant points
            include_idx = []
            include_id = []
            include_x = []
            include_y = []
            
            side_list = ["c","d"] if ((side == "i" and direction == "WB") or (side == "o" and direction == "EB" )) else ["a","b"]
            side_list.append(side)
            
            for i in range(len(ae_id)):
                id = ae_id[i] 
                if direction.lower() in id and indicator in id and (id[-1] in side_list or indicator == "yel"):
                
                    include_idx.append(i)
                    include_id.append(id)
                    include_x.append(ae_x[i])
                    include_y.append(ae_y[i])
    
        
            # sort points according to increasing X
            order = np.array(include_x).argsort()
            include_idx = [include_idx[o] for o in order]
            include_id = [include_id[o] for o in order]
            include_x  = [include_x[o] for o in order]
            include_y  = [include_y[o] for o in order]
            
            
        
            # convert points to roadway coordinates
            points = torch.stack([torch.tensor(include_x),torch.tensor(include_y),torch.zeros(len(include_x))]).transpose(1,0).unsqueeze(1)
            points_rc = self.space_to_state(points)
        
            # convert spline sample points to roadway coordiantes
            spline_data_rc = self.space_to_state(spline_data.unsqueeze(1))
            
            # Smooth spline points
            width = 1200*spline_sample_frequency + 5
            
            extend1 = torch.ones((width-1)//2) * spline_data_rc[0,1]
            extend2 = torch.ones((width-1)//2) * spline_data_rc[-1,1]
            splrc_extended = torch.cat([extend1,spline_data_rc[:,1],extend2])

            smoother = np.hamming(width)
            smoother = smoother/ sum(smoother)
            splrc = np.convolve(splrc_extended,smoother,mode = "valid")
            spline_data_rc[:,1] = torch.from_numpy(splrc)
        
            # 5. for each point, find closest smoothed spline point
            
            # dist will be [m,n], where n = number of points an n = number of sample points        
            m = points_rc.shape[0] 
            n = spline_data_rc.shape[0]
            
            # points_rc is of shape [m,6]
            # splrc is of shape [n,6]
            
            points_exp = points_rc.unsqueeze(1).expand(m,n,6)[:,:,:2]
            splrc_exp  = spline_data_rc.unsqueeze(0).expand(m,n,6)[:,:,:2]
            
            dist = ((points_exp - splrc_exp)**2).sum(dim = -1).sqrt()
            
            min_dist,min_idx = torch.min(dist,dim = 1) # of size m
            
            points_rc[:,:2] = spline_data_rc[min_idx][:,:2]
            
            # 6. convert set of selected spline points back to state plane coordinates
            points_new = self.state_to_space(points_rc)[:,0,:2]
            
            # 7. Parse resulting points
            for p_idx,point in enumerate(points_new):
                new_ae_x.append(points_new[p_idx,0])
                new_ae_y.append(points_new[p_idx,1])
                new_ae_id.append(include_id[p_idx])
        
        
        
        
        return new_ae_x,new_ae_y,new_ae_id
        
    
    
    
    def _fit_z_vp(self,cor,im_data,direction):
        
        #print("fitting Z coordinate scale")
        
        P_orig = cor["P"].copy()
        
        max_scale = 10000
        granularity = 1e-12
        upper_bound = max_scale
        lower_bound = -max_scale
        
        # create a grid of 100 evenly spaced entries between upper and lower bound
        C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
        step_size = C_grid[1] - C_grid[0]
        iteration = 1
        
        while step_size > granularity:
            
            best_error = np.inf
            best_C = None
            # for each value of P, get average reprojection error
            for C in C_grid:
                
                # scale P
                P = P_orig.copy()
                P[:,2] *= C
                
                
                # search for optimal scaling of z-axis row
                vp_lines = im_data[direction]["z_vp_lines"]
                
                # get bottom point (point # 2)
                points = torch.stack([ torch.tensor([vpl[2] for vpl in vp_lines]), 
                                          torch.tensor([vpl[3] for vpl in vp_lines]) ]).transpose(1,0)
                t_points = torch.stack([ torch.tensor([vpl[0] for vpl in vp_lines]), 
                                          torch.tensor([vpl[1] for vpl in vp_lines]) ]).transpose(1,0)
                heights =  torch.tensor([vpl[4] for vpl in vp_lines]).unsqueeze(1)
    
                
                # project to space
                
                d = points.shape[0]
                
                # convert points into size [dm,3]
                points = points.view(-1,2).double()
                points = torch.cat((points,torch.ones([points.shape[0],1],device=points.device).double()),1) # add 3rd row
                
                H = torch.from_numpy(cor["H"]).transpose(0,1).to(points.device)
                new_pts = torch.matmul(points,H)
                    
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                new_pts = new_pts.view(d,2)
                
                # add third column for height
                new_pts_shifted  = torch.cat((new_pts,heights.double()),1)
                
    
                # add fourth column for scale factor
                new_pts_shifted  = torch.cat((new_pts_shifted,torch.ones(heights.shape)),1)
                new_pts_shifted = torch.transpose(new_pts_shifted,0,1).double()
    
                # project to image
                P = torch.from_numpy(P).double().to(points.device)
                
    
                new_pts = torch.matmul(P,new_pts_shifted).transpose(0,1)
                
                # divide each point 0th and 1st column by the 2nd column
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                # drop scale factor column
                new_pts = new_pts[:,:2] 
                
                # reshape to [d,m,2]
                repro_top = new_pts.view(d,-1,2).squeeze()
                
                # get error
                
                error = torch.pow((repro_top - t_points),2).sum(dim = 1).sqrt().mean()
                
                # if this is the best so far, store it
                if error < best_error:
                    best_error = error
                    best_C = C
                    
            
            # define new upper, lower  with width 2*step_size centered on best value
            #print("On loop {}: best C so far: {} avg error {}".format(iteration,best_C,best_error))
            lower_bound = best_C - 2*step_size
            upper_bound = best_C + 2*step_size
            C_grid = np.linspace(lower_bound,upper_bound,num = 100,dtype = np.float64)
            step_size = C_grid[1] - C_grid[0]
    
            #print("New C_grid: {}".format(C_grid.round(4)))
            iteration += 1
        
        
        
        P_new = P_orig.copy()
        P_new[:,2] *= best_C
        cor["P"] = P_new
            
        
        #print("Best Error: {}".format(best_error))
        
    def _fit_spline(self,space_dir,use_MM_offset = True):
        """
        Spline fitting is done by:
            1. Assemble all points labeled along a yellow line in either direction
            2. Fit a spline to each side of each line
            3. Sample each spline at fine intervals
            4. Use finite difference method to determine the distance along the spline for each fit point
            5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
            5b. For each line, sample and define the set of points midway between
            5c. For each line, define a separate spline fit to these points that defines the position of that lane as a function of x (along roadway)
            6. Sample each yellow-line spline at fine intervals
            7. Move along one spline and at each point, find the closest point on each other spline
            8. Define a point on the median/ midpoint axis as the average on these 4 splines
            9. Use the set of median points to define a new spline
            10. Use the finite difference method to reparameterize this spline according to distance along it
            11. Optionally, compute a median spline distance offset from mile markers
            12. Optionally, recompute the same spline, this time accounting for the MM offset
            
        
            
            space_dir - str - path to directory with csv files of attributes labeled in space coordinates
            use_MM_offset - bool - if True, offset according to I-24 highway mile markers
        """
        
        print("Fitting median spline..")
        
        samples_per_foot = 12
        splines = {}
       
        # First, load all annotations
        for direction in ["EB","WB"]:
            for line_side in ["i","o"]:
                ### State space, do once
                
                ae_x = []
                ae_y = []
                ae_id = []
                
                # 1. Assemble all points labeled along a yellow line in either direction
                for file in os.listdir(space_dir):
                    if direction.lower() not in file:
                        continue
                    
                
                    
                    # load all points
                    dataframe = pd.read_csv(os.path.join(space_dir,file))
                    try:
                        dataframe = dataframe[dataframe['point_pos'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["point_id"].tolist()
                        st_id = [attribute_name + "_" + item for item in feature_idx]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                    except:
                        dataframe = dataframe[dataframe['side'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["id"].tolist()
                        side        = dataframe["side"].tolist()
                        st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                
                
                for line in ["yel{}".format(line_side), "d1{}".format(line_side),"d2{}".format(line_side),"d3{}".format(line_side)]:
                    #print("On line {} {}".format(line,direction))
                    ae_spl_x = []
                    ae_spl_y = []
                    ae_spl_u = []  # u parameterizes distance along spline 
                    ae_spl_id = []                
                    
                    letter_to_side = {"a":"i",
                                      "b":"i",
                                      "c":"o",
                                      "d":"o"
                                      }
                    
                    for i in range(len(ae_x)):
                        try:
                            if line in ae_id[i] or ( len(ae_id[i].split("_")) == 4  and line == ae_id[i].split("_")[1] + letter_to_side[ae_id[i].split("_")[3]]):
                            #if "yel{}".format(line_side) in ae_id[i]:
                                ae_spl_x.append(ae_x[i])
                                ae_spl_y.append(ae_y[i])
                                ae_spl_id.append(ae_id[i])
                        except KeyError:
                            pass
        
                    # if possible, use spline to smooth points
                    # if self.median_tck is not None:
                    #     ae_spl_x,ae_spl_y,ae_spl_id = self.shift_aerial_points2( ae_spl_x,ae_spl_y,ae_spl_id)
        
                    # 2. Fit a spline to each of EB, WB inside and outside
                     
                    # find a sensible smoothness parameter
                    
                    # compute the yellow line spline in state plane coordinates (sort points by y value since road is mostly north-south)
                    ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                    ae_data,idx = np.unique(ae_data,axis = 1,return_index = True)
                    order = np.argsort(ae_data[0,:])
                    ae_data2 = ae_data[:,order]#[::-1]]
                    order2 = np.argsort(ae_data[0,:])
                    ae_data = ae_data2.copy()
                    # 3. Sample the spline at fine intervals
                    # get spline and sample points on spline
                    w = np.ones(ae_data.shape[1])
                    
                    ae_spl_x  = [ae_spl_x[i]  for i in idx]
                    ae_spl_y  = [ae_spl_y[i]  for i in idx]
                    ae_spl_id = [ae_spl_id[i] for i in idx]
                    # for dim in [0,1]:
                    #     width = 13
                    #     extend1 = torch.ones((width-1)//2) * ae_data[dim,0]
                    #     extend2 = torch.ones((width-1)//2) * ae_data[dim,-1]
                    #     ys_extended = torch.cat([extend1,torch.from_numpy(ae_data[dim,:]),extend2])
                
                    #     smoother = np.hamming(width)
                    #     smoother = smoother/ sum(smoother)
                    #     ys = np.convolve(ys_extended,smoother,mode = "valid")
                    #     ae_data[dim,:] = ys
                    
                    knot_spacing = 0
                    s0 = 0.1
                    try:
                        ae_tck, ae_u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, per=False) 
                    except ValueError as e:
                        print(e)
                   
                    span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                    ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), ae_tck)

                    # TEMP
                    # plt.plot(ae_data[0,:],ae_data[1,:], "o-")
                    # legend.append(direction + "_" + line_side)
        
                    # 4. Use finite difference method to determine the distance along the spline for each fit point
                    fd_dist = np.concatenate(  (np.array([0]),  ((ae_x_prime[1:] - ae_x_prime[:-1])**2 + (ae_y_prime[1:] - ae_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
                    integral_dist = np.cumsum(fd_dist)
                    
                    # for each fit point, find closest point on spline, and assign it the corresponding integral distance
                    for p_idx in range(len(ae_spl_x)):
                        # I think these should instead reference ae_data because that one has been sorted
                        #px = ae_spl_x[p_idx]
                        #py = ae_spl_y[p_idx]
                        px = ae_data[0,p_idx]
                        py = ae_data[1,p_idx]
                        
                        dist = ((ae_x_prime - px)**2 + (ae_y_prime - py)**2)**0.5
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        ae_spl_u.append(integral_dist[min_idx])
                    
                    # 5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
                    #ae_spl_u.reverse()
                    
                    # sort by increasing u
                    ae_spl_u = np.array(ae_spl_u)
                    sorted_idxs = np.argsort(ae_spl_u)
                    ae_spl_u = ae_spl_u[sorted_idxs]
                    ae_data  = ae_data[:,sorted_idxs]
                    
                    while True and  knot_spacing < 10:
                        s0 *= 1.2
                        tck, u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, u = ae_spl_u)
                        knots = tck[0]
                        knot_spacing = np.min(np.abs(knots[5:-4] - knots[4:-5]))
                        #print(knot_spacing,s0)
                    tck, u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, u = ae_spl_u)
                    splines["{}_{}_{}".format(line,direction,line_side)] = [tck,u]
                

            
        # to prevent any bleedover
        del dist, min_dist, min_idx, ae_spl_y,ae_spl_x, ae_spl_u, ae_data
           
        import matplotlib.pyplot as plt
        plt.figure()
        legend = []
        # 6. Sample each of the 4 splines at fine intervals 
        for key in splines:
            tck,u = splines[key]
    
            span_dist = np.abs(u[0] - u[-1])
            x_prime, y_prime = interpolate.splev(np.linspace(u[0], u[-1], int(span_dist)), tck)
            splines[key].append(x_prime)
            splines[key].append(y_prime)
            
            plt.plot(x_prime,y_prime)
            legend.append(key)
        
        ###### Now, for each pair of splines, sample each and get a midway spline. These will be the splines we use
        
        for direction in ["EB","WB"]:
            for line in ["yel","d1","d2","d3"]:
                new_key = "{}_{}_center".format(direction,line)
                print("Getting smooth centered spline for {}".format(new_key))
                
                for key in splines.keys():
                    if direction in key and line in key and "i" in key:
                        i_spline = splines[key][0] # just tck
                    elif direction in key and line in key and "o" in key:
                        o_spline = splines[key][0] # just tck
                        u_range =  np.linspace(np.min(splines[key][1]), np.max(splines[key][1]), 50)
                
                # sample each spline at fine interval 
                #u_range = np.array(med_spl_u )
                
                
                # sample each spline at the same points
                x_in,y_in  = np.array(interpolate.splev(u_range,i_spline))
                x_out, y_out = np.array(interpolate.splev(u_range,o_spline))
                
                # average the two points
                y_mid = (y_in + y_out)/2
                x_mid = (x_in + x_out )/2
                
                data = np.stack([x_mid,y_mid])
                # fit spline y(u)
                s0 = 0.1
                knot_spacing = 0
                while knot_spacing < 100: # adjust spline smoothness for each center line
                    s0 *= 1.25
                    mid_line_tck,mid_line_u = interpolate.splprep(data, s=s0, u = u_range)
                    knots = mid_line_tck[0]
                    knot_spacing = np.min(np.abs(knots[5:-4] - knots[4:-5]))
                    #print(knot_spacing,s0)
                # store
            
                splines[new_key] = [mid_line_tck,mid_line_u]
                
                # plot
                x_prime, y_prime = interpolate.splev(np.linspace(u_range[0], u_range[-1], 5000), mid_line_tck)
                splines[new_key].append(x_prime)
                splines[new_key].append(y_prime)
                plt.plot(x_prime,y_prime)
                legend.append(new_key)

        # plt.legend(legend)
        # plt.show()        

        # # cache spline for each lane for plotting purposes
        # self.all_splines = splines 
        # return

        med_spl_x = []
        med_spl_y = []
        
        
        
        print("sampling yellow line splines")
        
        # 7. Move along one spline and at each point, find the closest point on each other spline
        # by default, we'll use EB_o as the base spline
        for main_key in ["yelo_EB_o","yeli_EB_i","yelo_WB_o","yeli_WB_i"]:
        #for main_key in ["yelo_WB_o","yeli_WB_i"]:

            main_spl = splines[main_key]
            main_x = main_spl[2]
            main_y = main_spl[3]
            
            
            for p_idx in range(len(main_x)):
                px,py = main_x[p_idx],main_y[p_idx]
                points_to_average = [np.array([px,py])]
                
                for key in splines:
                    if key != main_key:
                        if key not in ["yelo_WB_o","yeli_WB_i","yelo_EB_o","yeli_EB_i"]: continue
                        arr_x,arr_y = splines[key][2], splines[key][3]
                        
                        dist = np.sqrt((arr_x - px)**2 + (arr_y - py)**2)
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        
                        points_to_average.append( np.array([arr_x[min_idx],arr_y[min_idx]]))
                
                if len(points_to_average) < 4:
                    print("Outlier removed")
                    continue
                
                med_point = sum(points_to_average)/len(points_to_average)
                
                
                
                # 8. Define a point on the median/ midpoint axis as the average on these 4 splines
                med_spl_x.append(med_point[0])
                med_spl_y.append(med_point[1])
            
        print("Done sampling")
        
        # 9. Use the set of median points to define a new spline
        # sort by increasing x
        med_data = np.stack([np.array(med_spl_x),np.array(med_spl_y)])
        med_data = med_data[:,np.argsort(med_data[0])]
        
        # remove weirdness (i.e. outlying points) s.t. both x and y are monotonic and strictly increasing
        keep = (med_data[1,1:] <  med_data[1,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        
        keep = (med_data[0,1:] >  med_data[0,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        #med_data = np.ascontiguousarray(med_data)
        

        
        s = 10
        n_knots = len(med_data[0])
        while n_knots > 300:
            med_tck,med_u = interpolate.splprep(med_data, s=s, per=False)
            n_knots = len(med_tck[0])
            s = s**1.2
            print("Fitting median spline, n_knots = {}".format(n_knots))
        
        # 10. Use the finite difference method to reparameterize this spline according to distance along it
        med_spl_x = med_data[0]
        med_spl_y = med_data[1]
        span_dist = np.sqrt((med_spl_x[0] - med_spl_x[-1])**2 + (med_spl_y[0] - med_spl_y[-1])**2)
        med_x_prime, med_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), med_tck)
        
        
        med_fd_dist = np.concatenate(  (np.array([0]),  ((med_x_prime[1:] - med_x_prime[:-1])**2 + (med_y_prime[1:] - med_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
        med_integral_dist = np.cumsum(med_fd_dist)
        
        # for each fit point, find closest point on spline, and assign it the corresponding integral distance
        med_spl_u = []
        print("Getting integral distance along median spline")
        for p_idx in range(len(med_data[0])):
            px,py = med_data[0,p_idx], med_data[1,p_idx]
            
            dist = ((med_x_prime - px)**2 + (med_y_prime - py)**2)**0.5
            min_dist,min_idx= np.min(dist),np.argmin(dist)
            med_spl_u.append(med_integral_dist[min_idx])
        
        
        # sort by increasing u I guess
        med_spl_u = np.array(med_spl_u)
        sorted_idxs = np.argsort(med_spl_u)
        med_spl_u = med_spl_u[sorted_idxs]
        med_data  = med_data[:,sorted_idxs]
        
        # sort by strictly increasing u
        keep = (med_spl_u[1:] >  med_spl_u[:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        med_spl_u = med_spl_u[keep.nonzero()[0]]
        
        import matplotlib.pyplot as plt
        #plt.figure(figsize = (20,20))
        plt.plot(med_data[0],med_data[1])
        legend.append("Median")
        
       
        
        
        smoothing_dist = 500
        max_allowable_dev = 1
        # at this point, we have the median data and the integrated median distances (med_spl_u) and med_data
        # Let's try simply finding a single spline with <smoothing_dist> spaced fit-points and high-weighted edges
        
        
        
  
        
        
        # plt.figure()
        # plt.plot(med_spl_u)
        # plt.plot(np.array(med_spl_u)[np.argsort(med_spl_u)])
        # plt.legend(["Unsorted","Sorted"])
        
        
        s = 8
        min_dist = 0
        max_dev = 0

        while min_dist < smoothing_dist and max_dev < max_allowable_dev:
            final_tck,final_u = interpolate.splprep(med_data.astype(float), s=s, u=med_spl_u)
            knots = final_tck[0]
            min_dist = np.min(np.abs(knots[4:] - knots[:-4]))
            
            current_x,current_y = interpolate.splev(med_spl_u,final_tck)
            
            dist = ((current_x - med_data[0,:])**2 + (current_y - med_data[1,:])**2)**0.5
            max_dev = np.max(dist)
            
            print("With s = {}, {} knots, and min knot spacing {}, max spline - median point deviation = {}".format(s,len(knots),min_dist,max_dev))
            s = s**1.1
        
        #final_tck, final_u = interpolate.splprep(med_data, u = med_spl_u)
        self.median_tck = final_tck
        self.median_u = final_u
        
        
        # sample for final plotting
        final_plot_x,final_plot_y = interpolate.splev(np.linspace(min(med_spl_u), max(med_spl_u), 2000), final_tck)
        plt.plot(final_plot_x,final_plot_y)
        legend.append("Final Spline")
        
        
    
        
        
        # cache spline for each lane for plotting purposes
        self.all_splines = splines 

        
        
        
        
        
        ### get the inverse spline g(x) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        print(med_data.shape,med_spl_u.shape)
    
    
        # get guess_tck from all sparse 
        if True:
            med_data = np.array([final_plot_x,final_plot_y])
            med_spl_u = np.linspace(min(med_spl_u), max(med_spl_u), 2000)

            
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[0])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
    
        self.guess_tck = interpolate.splrep(med_data[0].astype(float),med_spl_u.astype(float))
        
        # get the secondary inverse spline g(y) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        
        # a second copy for later which won't be resorted
        u_range = np.array(med_spl_u )
        
        print(med_data.shape,med_spl_u.shape)
    
    
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[1])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
        self.guess_tck2 = interpolate.splrep(med_data[1].astype(float),med_spl_u.astype(float))
            

        plt.legend(legend)
        plt.title("Individual splines")
        plt.show()    
            
        
        ### Offset by MM
        if use_MM_offset:
            # 11. Optionally, compute a median spline distance offset from mile markers
            self.MM_offset = self._fit_MM_offset(space_dir)
        
        
        ### get y(u) splines for eastbound and westbound side yellow lines
        
        if False:
            self.yellow_splines = {}
            
            for direction in ["EB", "WB"]:
                i_spline = splines["yeli_{}_i".format(direction)][0] # just tck
                
            
                # sample each spline at fine interval 
                u_range = np.array(med_spl_u )
                sorted_idxs = np.argsort(u_range)
                u_range = u_range[sorted_idxs]
                
                # sample each spline at the same points
                y_in  = np.array(interpolate.splev(u_range,i_spline))
                med   = np.array(interpolate.splev(u_range,self.median_tck))
                
                dist = np.sum(((y_in-med)**2),axis = 0)**0.5
                if direction == "WB": dist *= -1
        
                self.yellow_splines[direction] = interpolate.splrep(u_range,dist,s = 1)
                

            
            
    def _fit_spline_old(self,space_dir,use_MM_offset = False):
        """
        Spline fitting is done by:
            1. Assemble all points labeled along a yellow line in either direction
            2. Fit a spline to each of EB, WB inside and outside
            3. Sample the spline at fine intervals
            4. Use finite difference method to determine the distance along the spline for each fit point
            5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
            6. Sample each spline at fine intervals
            7. Move along one spline and at each point, find the closest point on each other spline
            8. Define a point on the median/ midpoint axis as the average on these 4 splines
            9. Use the set of median points to define a new spline
            10. Use the finite difference method to reparameterize this spline according to distance along it
            11. Optionally, compute a median spline distance offset from mile markers
            12. Optionally, recompute the same spline, this time accounting for the MM offset
            
            space_dir - str - path to directory with csv files of attributes labeled in space coordinates
            use_MM_offset - bool - if True, offset according to I-24 highway mile markers
        """
        
        print("Fitting median spline..")
        
        samples_per_foot = 10
        splines = {}
        
        for direction in ["EB","WB"]:
            for line_side in ["i","o"]:
                ### State space, do once
                
                ae_x = []
                ae_y = []
                ae_id = []
                
                # 1. Assemble all points labeled along a yellow line in either direction
                for file in os.listdir(space_dir):
                    if direction.lower() not in file:
                        continue
                    
                
                    
                    # load all points
                    dataframe = pd.read_csv(os.path.join(space_dir,file))
                    try:
                        dataframe = dataframe[dataframe['point_pos'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["point_id"].tolist()
                        st_id = [attribute_name + "_" + item for item in feature_idx]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                    except:
                        dataframe = dataframe[dataframe['side'].notnull()]
                        attribute_name = file.split(".csv")[0]
                        feature_idx = dataframe["id"].tolist()
                        side        = dataframe["side"].tolist()
                        st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                        
                        st_x = dataframe["st_x"].tolist()
                        st_y = dataframe["st_y"].tolist()
                    
                        ae_x  += st_x
                        ae_y  += st_y
                        ae_id += st_id
                
                
                for line in ["yel{}".format(line_side), "d1{}".format(line_side),"d2{}".format(line_side),"d3{}".format(line_side)]:
                
                    ae_spl_x = []
                    ae_spl_y = []
                    ae_spl_u = []  # u parameterizes distance along spline 
                                    
                    
                    letter_to_side = {"a":"i",
                                      "b":"i",
                                      "c":"o",
                                      "d":"o"
                                      }
                    
                    for i in range(len(ae_x)):
                        try:
                            if line in ae_id[i] or ( len(ae_id[i].split("_")) == 4  and line == ae_id[i].split("_")[1] + letter_to_side[ae_id[i].split("_")[3]]):
                            #if "yel{}".format(line_side) in ae_id[i]:
                                ae_spl_x.append(ae_x[i])
                                ae_spl_y.append(ae_y[i])
                        except KeyError:
                            pass
        
                    # 2. Fit a spline to each of EB, WB inside and outside
                     
                    # find a sensible smoothness parameter
                    s0 = 100
                    n_knots = np.inf
                    
                    # compute the yellow line spline in state plane coordinates (sort points by y value since road is mostly north-south)
                    ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                    order = np.argsort(ae_data[0,:])
                    ae_data2 = ae_data[:,order]#[::-1]]
                    order2 = np.argsort(ae_data[0,:])
                    ae_data = ae_data2.copy()
                    # 3. Sample the spline at fine intervals
                    # get spline and sample points on spline
                    w = np.ones(ae_data.shape[1])
                    # w[:10 ] = 1000
                    # w[-10:] = 1000
                    
                    while n_knots > 200:
                        s0 = s0**1.05
                        print(s0,n_knots,line)
                        
                        try:
                            ae_tck, ae_u = interpolate.splprep(ae_data, s=s0, w = w, per=False)
                            n_knots = len(ae_tck[0])
                        except:
                            s0 = s0 **(1/1.05)
                    
                    ae_tck, ae_u = interpolate.splprep(ae_data, s=s0, w = w, per=False) 
                    
                    span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                    ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), ae_tck)
                
                    # TEMP
                    # plt.plot(ae_data[0,:],ae_data[1,:], "o-")
                    # legend.append(direction + "_" + line_side)
        
                    # 4. Use finite difference method to determine the distance along the spline for each fit point
                    fd_dist = np.concatenate(  (np.array([0]),  ((ae_x_prime[1:] - ae_x_prime[:-1])**2 + (ae_y_prime[1:] - ae_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
                    integral_dist = np.cumsum(fd_dist)
                    
                    # for each fit point, find closest point on spline, and assign it the corresponding integral distance
                    for p_idx in range(len(ae_spl_x)):
                        # I think these should instead reference ae_data because that one has been sorted
                        #px = ae_spl_x[p_idx]
                        #py = ae_spl_y[p_idx]
                        px = ae_data[0,p_idx]
                        py = ae_data[1,p_idx]
                        
                        dist = ((ae_x_prime - px)**2 + (ae_y_prime - py)**2)**0.5
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        ae_spl_u.append(integral_dist[min_idx])
                    
                    # 5. Refit the splines, this time parameterizing the spline by these distances (u parameter in scipy.splprep)
                    #ae_spl_u.reverse()
                    
                    # sort by increasing u
                    ae_spl_u = np.array(ae_spl_u)
                    sorted_idxs = np.argsort(ae_spl_u)
                    ae_spl_u = ae_spl_u[sorted_idxs]
                    ae_data  = ae_data[:,sorted_idxs]
                    
                    tck, u = interpolate.splprep(ae_data.astype(float), s=s0, w = w, u = ae_spl_u)
                    splines["{}_{}_{}".format(line,direction,line_side)] = [tck,u]
                

            
        # to prevent any bleedover
        del dist, min_dist, min_idx, ae_spl_y,ae_spl_x, ae_spl_u, ae_data
           
        import matplotlib.pyplot as plt
        plt.figure()
        legend = []
        # 6. Sample each of the 4 splines at fine intervals (every 2 feet)
        for key in splines:
            tck,u = splines[key]
    
            span_dist = np.abs(u[0] - u[-1])
            x_prime, y_prime = interpolate.splev(np.linspace(u[0], u[-1], int(span_dist/2)), tck)
            splines[key].append(x_prime)
            splines[key].append(y_prime)
            
            plt.plot(x_prime,y_prime)
            legend.append(key)
        

        # plt.legend(legend)
        # plt.show()        

        med_spl_x = []
        med_spl_y = []
        
        
        
        
        # 7. Move along one spline and at each point, find the closest point on each other spline
        # by default, we'll use EB_o as the base spline
        for main_key in ["yelo_WB_o","yeli_WB_i"]:#,"yelo_WB_o","yeli_WB_i"]:
            main_spl = splines[main_key]
            main_x = main_spl[2]
            main_y = main_spl[3]
            
            
            for p_idx in range(len(main_x)):
                px,py = main_x[p_idx],main_y[p_idx]
                points_to_average = [np.array([px,py])]
                
                for key in ["yelo_WB_o","yeli_WB_i"]:# splines:
                    if key != main_key:
                        arr_x,arr_y = splines[key][2], splines[key][3]
                        
                        dist = np.sqrt((arr_x - px)**2 + (arr_y - py)**2)
                        min_dist,min_idx= np.min(dist),np.argmin(dist)
                        
                        points_to_average.append( np.array([arr_x[min_idx],arr_y[min_idx]]))
                
                # if len(points_to_average) < 3:
                #     print("Outlier removed")
                #     continue
                
                med_point = sum(points_to_average)/len(points_to_average)
                
                
                
                # 8. Define a point on the median/ midpoint axis as the average on these 4 splines
                med_spl_x.append(med_point[0])
                med_spl_y.append(med_point[1])
            
    
        
        # 9. Use the set of median points to define a new spline
        # sort by increasing x
        med_data = np.stack([np.array(med_spl_x),np.array(med_spl_y)])
        med_data = med_data[:,np.argsort(med_data[0])]
        
        # remove weirdness (i.e. outlying points) s.t. both x and y are monotonic and strictly increasing
        keep = (med_data[1,1:] <  med_data[1,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        
        keep = (med_data[0,1:] >  med_data[0,:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        #med_data = np.ascontiguousarray(med_data)
        
        s = 2
        n_knots = len(med_data[0])
        while n_knots > 5000:
            med_tck,med_u = interpolate.splprep(med_data, s=s, per=False)
            n_knots = len(med_tck[0])
            s = s**1.2
        
        # 10. Use the finite difference method to reparameterize this spline according to distance along it
        med_spl_x = med_data[0]
        med_spl_y = med_data[1]
        span_dist = np.sqrt((med_spl_x[0] - med_spl_x[-1])**2 + (med_spl_y[0] - med_spl_y[-1])**2)
        med_x_prime, med_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), med_tck)
        
        
        med_fd_dist = np.concatenate(  (np.array([0]),  ((med_x_prime[1:] - med_x_prime[:-1])**2 + (med_y_prime[1:] - med_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
        med_integral_dist = np.cumsum(med_fd_dist)
        
        # for each fit point, find closest point on spline, and assign it the corresponding integral distance
        med_spl_u = []
        for p_idx in range(len(med_data[0])):
            px,py = med_data[0,p_idx], med_data[1,p_idx]
            
            dist = ((med_x_prime - px)**2 + (med_y_prime - py)**2)**0.5
            min_dist,min_idx= np.min(dist),np.argmin(dist)
            med_spl_u.append(med_integral_dist[min_idx])
        
        # sort by increasing u I guess
        med_spl_u = np.array(med_spl_u)
        sorted_idxs = np.argsort(med_spl_u)
        med_spl_u = med_spl_u[sorted_idxs]
        med_data  = med_data[:,sorted_idxs]
        
        # sort by strictly increasing u
        keep = (med_spl_u[1:] >  med_spl_u[:-1]).astype(int).tolist()
        keep = [1] + keep
        keep = np.array(keep)
        med_data = med_data[:,keep.nonzero()[0]]
        med_spl_u = med_spl_u[keep.nonzero()[0]]
        
        import matplotlib.pyplot as plt
        #plt.figure(figsize = (20,20))
        plt.plot(med_data[0],med_data[1])
        legend.append("Median")
        
        
        
        
        smoothing_dist = 20
        max_allowable_dev = 0.25
        # at this point, we have the median data and the integrated median distances (med_spl_u) and med_data
        # Let's try simply finding a single spline with <smoothing_dist> spaced fit-points and high-weighted edges
        
        
        
  
        
        
        # plt.figure()
        # plt.plot(med_spl_u)
        # plt.plot(np.array(med_spl_u)[np.argsort(med_spl_u)])
        # plt.legend(["Unsorted","Sorted"])
        
        
        s = 8
        min_dist = 0
        max_dev = 0

        while min_dist < smoothing_dist and max_dev < max_allowable_dev:
            final_tck,final_u = interpolate.splprep(med_data.astype(float), s=s, u=med_spl_u)
            knots = final_tck[0]
            min_dist = np.min(np.abs(knots[4:] - knots[:-4]))
            
            current_x,current_y = interpolate.splev(med_spl_u,final_tck)
            
            dist = ((current_x - med_data[0,:])**2 + (current_y - med_data[1,:])**2)**0.5
            max_dev = np.max(dist)
            
            print("With s = {}, {} knots, and min knot spacing {}, max spline - median point deviation = {}".format(s,len(knots),min_dist,max_dev))
            s = s**1.1
        
        #final_tck, final_u = interpolate.splprep(med_data, u = med_spl_u)
        self.median_tck = final_tck
        self.median_u = final_u
        
        
        # sample for final plotting
        final_plot_x,final_plot_y = interpolate.splev(np.linspace(min(med_spl_u), max(med_spl_u), 2000), final_tck)
        plt.plot(final_plot_x,final_plot_y)
        legend.append("Final Spline")
        
        
        
        # Finally, resample the spline at <smoothing_dist> foot intervals, and use this to fit a final spline
        ulist = []
        xlist = []
        ylist = []
        for u in range(int(min(final_u)),int(max(final_u)), smoothing_dist):
            ulist.append(u)
            x,y = interpolate.splev(u,final_tck)
            xlist.append(x)
            ylist.append(y)
         
        legend.append("Smoothed median respampled data")    
        plt.plot(xlist,ylist)
        
        data = np.array([xlist,ylist])
        smooth_tck,smooth_u = interpolate.splprep(data,u= ulist)
        
        # self.median_tck = smooth_tck
        # self.median_u = smooth_u
        
        
        # cache spline for each lane for plotting purposes
        self.all_splines = splines 

        
        
        
        
        
        # get the inverse spline g(x) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        print(med_data.shape,med_spl_u.shape)
    
    
        # get guess_tck from all sparse 
        if True:
            med_data = data
            med_spl_u = np.array(ulist)

            
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[0])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
    
        self.guess_tck = interpolate.splrep(med_data[0].astype(float),med_spl_u.astype(float))
        
        # get the secondary inverse spline g(y) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        print(med_data.shape,med_spl_u.shape)
    
    
        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[1])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]
        self.guess_tck2 = interpolate.splrep(med_data[1].astype(float),med_spl_u.astype(float))
            

        plt.legend(legend)
        plt.title("Individual splines")
        plt.show()    
            
        
        if use_MM_offset:
            # 11. Optionally, compute a median spline distance offset from mile markers
            self.MM_offset = self._fit_MM_offset(space_dir)
        
            # 12. Optionally, recompute the same spline, this time accounting for the MM offset
            med_spl_u += self.MM_offset
            final_tck, final_u = interpolate.splprep(med_data.astype(float), s=s, u = med_spl_u)
            self.median_tck = final_tck
            self.median_u = final_u
    
    
    def closest_spline_point(self,points, epsilon = 0.01, max_iterations = 10):
        """
        Given a tensor of points in 3D space, find the closest point on the median spline
        for each point as follows:
            1. Query self.guess_tck spline to get f(x) = u initial guess
            2. Use Newton's method to find the point where dist = min
      
        points         - [d,3] tensor of points in state plane coordinates
        epsilon        - float - keep iterating while max change > epsilon or ii.
        max_iterations - int - ii. keep iterating until n_iterations = max_iterations
        RETURNS:         [d] tensor of coordinates along spline axis
        """
        PLOT = False
        leg = []
        start = time.time()
        # intial guess at closest u values
        points = points.cpu().data.numpy()
        #points = points[:1,:]
        guess_u = interpolate.splev(points[:,0],self.guess_tck)
        guess_u2 = interpolate.splev(points[:,1],self.guess_tck2)
        guess_u = (guess_u + guess_u2)/2.0
        
        guess_u *= 0
        
        
        
        # guesses = np.linspace(guess_u-500,guess_u+500,103)
        # guess_x,guess_y = interpolate.splev(guesses, self.median_tck)
        
        # plt.scatter(guess_x[:,0],guess_y[:,0])
        # leg.append("Initial samples")
        
        # dist = np.sqrt((guess_x - points[:,0][np.newaxis,:].repeat(guess_x.shape[0],axis = 0))**2 + (guess_y - points[:,1][np.newaxis,:].repeat(guess_y.shape[0],axis = 0))**2)
        # min_idx = np.argmin(dist,axis = 0)
        
        # guess = guesses[min_idx,[i for i in range(guesses.shape[1])]]
        # guesses = np.linspace(guess-10,guess+10,200)
        # guess_x,guess_y = interpolate.splev(guesses, self.median_tck)
        
        # dist = np.sqrt((guess_x - points[:,0])**2 + (guess_y - points[:,1])**2)
        # min_idx = np.argmin(dist,axis = 0)
        
        #guess_u = guesses[min_idx,[i for i in range(guesses.shape[1])]]
        
        
        it = 0
        max_change = np.inf
        while it < max_iterations and max_change > epsilon:
            spl_x,spl_y             = interpolate.splev(guess_u,self.median_tck)
            spl_xx,spl_yy = interpolate.splev(guess_u,self.median_tck, der = 1)
            spl_xxx,spl_yyy = interpolate.splev(guess_u,self.median_tck, der = 2)

            
            dist_proxy = (spl_x - points[:,0])**2 + (spl_y - points[:,1])**2
            dist_proxy_deriv = (spl_x-points[:,0])*spl_xx + (spl_y-points[:,1])*spl_yy
            #dist_proxy_deriv2 = (2*spl_xx**2)+2*(spl_x-points[:,0])*spl_xxx + (2*spl_yy**2)+2*(spl_y-points[:,1])*spl_yyy
            dist_proxy_deriv2 = (spl_xx**2)+(spl_x-points[:,0])*spl_xxx + (spl_yy**2)+(spl_y-points[:,1])*spl_yyy

            
            new_u = guess_u - dist_proxy_deriv/dist_proxy_deriv2
            
            max_change = np.max(np.abs(new_u-guess_u))
            it += 1
            
            guess_u = new_u
            
            if PLOT:
                plt.scatter(spl_x[:1],spl_y[:1])
                leg.append(it)
                
            #print("Max step: {}".format(max_change))
         
        #print("Newton method took {}s for {} points".format(time.time() - start,points.shape[0]))
        
        if PLOT:
            
            # plt.scatter(guess_x[min_idx[0],0],guess_y[min_idx[0],0])
            # leg.append("Closest")
            
            plt.scatter(points[0,0],points[0,1])
            leg.append("Point")
            
            splx,sply = interpolate.splev(np.linspace(0,30000,100000),self.median_tck)
            plt.plot(splx,sply)
            leg.append("Median Spline")
            
            plt.legend(leg)
            plt.axis("equal")
            plt.show()
            raise Exception
        
        return guess_u
            
    
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
        
    def _generate_lane_offset(self,space_dir,SPLINE_OFFSET = False,SHIFT = False):
        """
        0.) Load all points into line_pts
        1.) Get mean for yellow line on each side
        2.) For each point, subtract yellow line and add back in mean yellow line
        3.) Find best fit line (mean/ std) for each line
        4.) Plot each
        5.) Save lane offsets?
        """
    
        def safe(x):
            try:
                return x.item()
            except:
                return x
            
        # 0. assemble all lane marking coordinates across all correspondences

        line_pts = {}
        
        for direction in ["eb","wb"]:
            ### State space, do once
            

            
            # 1. Assemble all points labeled along a yellow line in either direction
            for file in os.listdir(space_dir):
                if direction.lower() not in file or ("yel" not in file and "d1" not in file and "d2" not in file and "d3" not in file and "d4" not in file):
                    continue
                
                print(file)
                
                
                ae_x = []
                ae_y = []
                ae_id = []
                # load all points
                dataframe = pd.read_csv(os.path.join(space_dir,file))
                try:
                    dataframe = dataframe[dataframe['point_pos'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["point_id"].tolist()
                    st_id = [attribute_name + "_" + item for item in feature_idx]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
                except:
                    dataframe = dataframe[dataframe['side'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["id"].tolist()
                    side        = dataframe["side"].tolist()
                    st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
            
                # if direction == "wb":
                if SHIFT:
                    ae_x,ae_y,ae_id = self.shift_aerial_points2(ae_x,ae_y,ae_id)    
            
                    
            
                    if False:
                        # save updated spline_points here
                        if "yel" not in file:
                            id_trunc = [item.split("_")[-2] + "_" + item.split("_")[-1] for item in ae_id]
                        else:
                            id_trunc = [int(item.split("_")[-1]) for item in ae_id]
                            
                        update_dict = dict([(id_trunc[i], [safe(ae_x[i]),safe(ae_y[i])]) for i in range(len(ae_id))])     
                        orig_ids = feature_idx
                        
                        x_update = []
                        y_update = []
                        for item in orig_ids:
                            new_item = update_dict[item]
                            x_update.append(new_item[0])
                            y_update.append(new_item[1])
                            
                        dataframe["st_x"] = x_update
                        dataframe["st_y"] = y_update
                        dataframe.to_csv(os.path.join("shifted_aerial_points",file))    
            
                try:
                    line_pts[attribute_name][0].append(ae_x)
                    line_pts[attribute_name][1].append(ae_y)
                except KeyError:
                    line_pts[attribute_name] = [ae_x,ae_y]
                    
        if self.yellow_offsets is None:
            yellow_offsets = {}
            for direction in ["EB","WB"]:
                # 1. Get mean yellow line value
                spl = self.all_splines["yeli_{}_i".format(direction.upper())]
                ran = max(spl[1]) - min(spl[1])
                
                u = np.linspace(min(spl[1]),max(spl[1]),int(ran))
                
               
                tck = spl[0]
                yelx,yely = interpolate.splev(u,tck)
                yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                yel_state = self.space_to_state(yel_space)
                ys  = yel_state[:,1]
                
                width = 1205
                extend1 = torch.ones((width-1)//2) * ys[0]
                extend2 = torch.ones((width-1)//2) * ys[-1]
                ys_extended = torch.cat([extend1,ys,extend2])
    
                smoother = np.hamming(width)
                smoother = smoother/ sum(smoother)
                ys = np.convolve(ys_extended,smoother,mode = "valid")
                
                bin_width = 10
                offsets = np.zeros(3000)
                counts = np.zeros(3000)
                
                for idx in range(len(u)):
                    binidx = int(u[idx]//bin_width)
                    offsets[binidx] += ys[idx]
                    counts[binidx]  += 1
                
                offsets = offsets / (counts + 1e-04)
                
                idx = 0
                while offsets[idx] == 0:
                    idx += 1
                offsets[:idx] = offsets[idx]
                
                idx = -1
                while offsets[idx] == 0:
                    idx-= 1
                    offsets[idx:] = offsets[idx]
                    
                yellow_offsets[direction] = offsets
            
            
            self.yellow_offsets = yellow_offsets
            self.save(self.save_file)
        
        # 2. for each lane, convert all points to roadway coordinates
        
        roadway_data = {}
        for key in line_pts.keys():
            
            if "eb" in key or "EB" in key:
                ymean = 12
                direction = "EB"
            else:
                ymean = -12
                direction = "WB"
                
                
            data = line_pts[key]
            space_data = torch.tensor([data[0],data[1],torch.zeros(len(data[0]))]).permute(1,0).unsqueeze(1)
            
            road_data = self.space_to_state(space_data)
            order = torch.argsort(road_data[:,0])
            
            
            #tck = self.all_splines[direction.upper()+ "_yel_center"][0]
            
            # TODO - let's use the black lines we plot as the yellow offset spline since they are obviously right
            # if direction == "WB":
            #     road_data[:,1] = road_data[:,1] - yel_state[:,1] + ymean
            # else:
            if SPLINE_OFFSET:
                
                # tck = self.all_splines["yeli_{}_i".format(direction.upper())][0]
                # yelx,yely = interpolate.splev(road_data[:,0],tck)
                # yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                # yel_state = self.space_to_state(yel_space)
                #                                         # TODO - let's use the black lines we plot as the yellow offset spline since they are obviously right
                # road_data[:,1] = road_data[:,1] - yel_state[:,1] + ymean
                
                # new offset style
                road_data_offset_bins = (road_data[:,0] /10).int()
                offsets = yellow_offsets[direction][road_data_offset_bins]
                road_data[:,1] = road_data[:,1] - offsets + ymean
                
            roadway_data[key] = road_data[order,:2]
            
        # 3. Get mean and stddev for line
        for key in roadway_data.keys():
            mean = torch.mean(roadway_data[key][:,1])
            std =  torch.std(roadway_data[key][:,1])
            print("Stats for {}: mean {}, stddev {}".format(key,mean,std))
        
        
        
        
        
        import matplotlib.pyplot as plt
        plt.figure(figsize = (4,4))
        leg= []
        leg2 = []
        # 5. Plot
        
        for key in roadway_data.keys():
            if "yel" in key:
                color = (1,1,0)
            else:
                color = (0,0,0)
            legentry1 = plt.plot(roadway_data[key][:,0],roadway_data[key][:,1],color = color)
            leg.append(key)
        
        if False and self.all_splines:
            for key in self.all_splines.keys():
                if True and "center" not in key:
                    
                    if "eb" in key or "EB" in key:
                        ymean = 12
                        direction = "EB"
                    else:
                        ymean = -12
                        direction = "WB"
                    
                    spline = self.all_splines[key]
                    xs,ys = spline[2],spline[3]
                    
                    space_data = torch.tensor([xs,ys,torch.zeros(len(xs))]).permute(1,0).unsqueeze(1)
                    state_data = self.space_to_state(space_data)
                    
                    #tck = self.all_splines[direction.upper()+ "_yel_center"][0]
                    
                    if SPLINE_OFFSET:
                        # tck = self.all_splines["yeli_{}_i".format(direction.upper())][0]
                        # yelx,yely = interpolate.splev(state_data[:,0],tck)
                        # yel_space = torch.tensor([yelx,yely,torch.zeros(len(yelx))]).permute(1,0).unsqueeze(1)
                        # yel_state = self.space_to_state(yel_space)
                        # state_data[:,1] = state_data[:,1] - yel_state[:,1] + ymean
                        
                        # new offset style
                        road_data_offset_bins = (state_data[:,0]/10).int()
                        offsets = yellow_offsets[direction][road_data_offset_bins]
                        state_data[:,1] = state_data[:,1] - offsets + ymean
                    
                    
                    xs,ys = state_data[:,0],state_data[:,1]
                    
                    # try smoothing
                    if True:
                        width = 1205
                        extend1 = torch.ones((width-1)//2) * ys[0]
                        extend2 = torch.ones((width-1)//2) * ys[-1]
                        ys_extended = torch.cat([extend1,ys,extend2])
    
                        smoother = np.hamming(1205)
                        smoother = smoother/ sum(smoother)
                        ys = np.convolve(ys_extended,smoother,mode = "valid")
                        
                        
                    #xs,ys = interpolate.splev(np.linspace(0,25000,1000),tck)
    
                    # sample at 1000 points
                    plt.plot(xs,ys,color = (0,0,0))
                    leg.append(key)
        
        if True:
            # for each correspondence, plot all of the image points in roadway coordinates
            for corr in self.correspondence:
                direction = corr.split("_")[-1]
                if direction == "WB":  ymean = -12
                else: ymean = 12
                im_pts = self.correspondence[corr]["corr_pts"]
                im_pts = torch.from_numpy(im_pts).unsqueeze(1)
                rcs_pts = self.im_to_state(im_pts, heights = torch.zeros(im_pts.shape[0]),name = [corr for _ in range(im_pts.shape[0])],refine_heights = False)
                
                if SPLINE_OFFSET:
                    road_data_offset_bins = (rcs_pts[:,0]/10).int()
                    offsets = yellow_offsets[direction][road_data_offset_bins]
                    rcs_pts[:,1] = rcs_pts[:,1] - offsets + ymean
                    
                legentry2 = plt.scatter(rcs_pts[:,0],rcs_pts[:,1], color = (0,0,1), marker = ".")
        
        plt.xlabel("Roadway X (ft)")
        plt.ylabel("Roadway Y (ft)")
        plt.legend([legentry2],["Image Correspondence Points, Transformed"])

        
        
    def _generate_lane_offset_old(self,space_dir):
        """
        For each lane in d1,d2,d3,d4,yellow for each side, compute mean and standard deviation
        """
    
    
        # 1. assemble all lane marking coordinates across all correspondences

        line_pts = {}
        
        for direction in ["eb","wb"]:
            ### State space, do once
            

            
            # 1. Assemble all points labeled along a yellow line in either direction
            for file in os.listdir(space_dir):
                if direction.lower() not in file or ("yel" not in file and "d1" not in file and "d2" not in file and "d3" not in file and "d4" not in file):
                    continue
                
                
                
                ae_x = []
                ae_y = []
                ae_id = []
                # load all points
                dataframe = pd.read_csv(os.path.join(space_dir,file))
                try:
                    dataframe = dataframe[dataframe['point_pos'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["point_id"].tolist()
                    st_id = [attribute_name + "_" + item for item in feature_idx]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
                except:
                    dataframe = dataframe[dataframe['side'].notnull()]
                    attribute_name = file.split(".csv")[0]
                    feature_idx = dataframe["id"].tolist()
                    side        = dataframe["side"].tolist()
                    st_id = [attribute_name + str(side[i]) + "_" + str(feature_idx[i]) for i in range(len(feature_idx))]
                    
                    st_x = dataframe["st_x"].tolist()
                    st_y = dataframe["st_y"].tolist()
                
                    ae_x  += st_x
                    ae_y  += st_y
                    ae_id += st_id
            
                try:
                    line_pts[attribute_name][0].append(ae_x)
                    line_pts[attribute_name][1].append(ae_y)
                except KeyError:
                    line_pts[attribute_name] = [ae_x,ae_y]
                    
                

            
    
        # 2. for each lane, convert all points to roadway coordinates
        
        roadway_data = {}
        for key in line_pts.keys():
            data = line_pts[key]
            space_data = torch.tensor([data[0],data[1],torch.zeros(len(data[0]))]).permute(1,0).unsqueeze(1)
            
            road_data = self.space_to_state(space_data)
            order = torch.argsort(road_data[:,0])
            
            roadway_data[key] = road_data[order,:2]
            
        # 3. Get mean and stddev for line
        for key in roadway_data.keys():
            mean = torch.mean(roadway_data[key][:,1])
            std =  torch.std(roadway_data[key][:,1])
            print("Stats for {}: mean {}, stddev {}".format(key,mean,std))
        
        
        import matplotlib.pyplot as plt
        plt.figure()
        leg= []
        # 5. Plot
        
        for key in roadway_data.keys():
            plt.plot(roadway_data[key][:,0],roadway_data[key][:,1])
            leg.append(key)
        
        if self.all_splines:
            for spline in self.all_splines.values():
                xs,ys = spline[2],spline[3]
                
                space_data = torch.tensor([xs,ys,torch.zeros(len(xs))]).permute(1,0).unsqueeze(1)
                state_data = self.space_to_state(space_data)
                
                xs,ys = state_data[:,0],state_data[:,1]
                #xs,ys = interpolate.splev(np.linspace(0,25000,1000),tck)

                # sample at 1000 points
                plt.plot(xs,ys)
        
        
        # finally, plot the stored y splines
        
        for key in self.yellow_splines.keys():
            spl = self.yellow_splines[key]
            
            # sample spline
            r = np.linspace(min(xs),max(xs),2000)
            
            y = interpolate.splev(r,spl)
            
            plt.plot(r,y)
            
            leg.append("{} yellow line spline".format(key))
            
        plt.legend(leg)

        
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
         
    
    
    #%% Conversion Functions
    @safe_name
    def _im_sp(self,points,heights = None, name = None, direction = "EB",refine_heights = False):
        """
        Converts points by means of perspective transform from image to space
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object (guessed) heights or 0
        direction - "EB" or "WB" - specifies which correspondence to use
        refine_heights - bool - if True, points are reprojected back into image and used to rescale the heights
        RETURN:     [d,m,3] array of points in space 
        """
        # if name is None:
        #     name = list(self.correspondence.keys())[0]
        d = points.shape[0]
        if d == 0:
            return torch.empty(0,8,3)
        
        if type(name) == list and len(name[0].split("_")) == 1:
            temp_name = [sub_n+ "_WB" for sub_n in name]
            name = temp_name
            
        n_pts = points.shape[1]
        # convert points into size [dm,3]
        points = points.view(-1,2).double()
        points = torch.cat((points,torch.ones([points.shape[0],1],device=points.device).double()),1) # add 3rd row
        
        if heights is not None:
            
            if type(name) == list:
                H = torch.from_numpy(np.stack([self.correspondence[sub_n]["H"].transpose(1,0) for sub_n in name])) # note that must do transpose(1,0) because this is a numpy operation, not a torch operation ...
                H = H.unsqueeze(1).repeat(1,n_pts,1,1).view(-1,3,3).to(points.device)
                points = points.unsqueeze(1)
                new_pts = torch.bmm(points,H)
                new_pts = new_pts.squeeze(1)
            else:
                H = torch.from_numpy(self.correspondence[name]["H"]).transpose(0,1).to(points.device)
                new_pts = torch.matmul(points,H)
            
            # divide each point 0th and 1st column by the 2nd column
            new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
            new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
            
            # drop scale factor column
            new_pts = new_pts[:,:2] 
            
            # reshape to [d,m,2]
            new_pts = new_pts.view(d,-1,2)
            
            # add third column for height
            new_pts = torch.cat((new_pts,torch.zeros([d,new_pts.shape[1],1],device = points.device).double()),2)
            
            # overwrite last_4 points with first 4 (i.e. top points are gibberish and not meaningful)
            if n_pts == 8:
                new_pts[:,4:,:] = new_pts[:,:4,:]
            
                if type(heights) == torch.Tensor: 
                    new_pts[:,[4,5,6,7],2] = heights.unsqueeze(1).repeat(1,4).double().to(points.device)
            
        else:
            print("No heights were input")
            return
        
        if refine_heights:
            template_boxes = self.space_to_im(new_pts,name)
            heights_new = self.height_from_template(template_boxes, heights, points.view(d,8,3))
            new_pts[:,[4,5,6,7],2] = heights_new.unsqueeze(1).repeat(1,4).double().to(points.device)
            
        return new_pts
    
    @safe_name
    def _sp_im(self,points, name = None, direction = "EB"):
       """
       Projects 3D space points into image/correspondence using P:
           new_pts = P x points T  ---> [dm,3] T = [3,4] x [4,dm]
       performed by flattening batch dimension d and object point dimension m together
       
       name      - list of correspondence key names
       direction - "EB" or "WB" - speecifies which correspondence to use
       points    - [d,m,3] array of points in 3-space, m is probably 8
       RETURN:     [d,m,2] array of points in 2-space
       """
       
       d = points.shape[0]
       if d == 0:
           return torch.empty(0,8,2)
       
       if name is None:
           name = list(self.correspondence.keys())[0]
       elif type(name) == list and len(name[0].split("_")) == 1:
           name = self.get_direction(points,name)[0]
       n_pts = points.shape[1]
       # get directions and append to names
       
       
       
       
           
       # convert points into size [dm,4]
       points = points.view(-1,3)
       points = torch.cat((points.double(),torch.ones([points.shape[0],1],device = points.device).double()),1) # add 4th row
       
       
       # project into [dm,3]
       if type(name) == list:
               P = torch.from_numpy(np.stack([self.correspondence[sub_n]["P"] for sub_n in name]))
               P = P.unsqueeze(1).repeat(1,n_pts,1,1).reshape(-1,3,4).to(points.device)
               points = points.unsqueeze(1).transpose(1,2)
               new_pts = torch.bmm(P,points).squeeze(2)
       else:
           points = torch.transpose(points,0,1).double()
           P = torch.from_numpy(self.correspondence[name]["P"]).double().to(points.device)
           new_pts = torch.matmul(P,points).transpose(0,1)
       
       # divide each point 0th and 1st column by the 2nd column
       new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
       new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
       
       # drop scale factor column
       new_pts = new_pts[:,:2] 
       
       # reshape to [d,m,2]
       new_pts = new_pts.view(d,-1,2)
       return new_pts 
    
    def im_to_space(self,points, name = None,heights = None,classes = None,refine_heights = True):
        """
        Wrapper function on _im_sp necessary because it is not immediately evident 
        from points in image whether the EB or WB corespondence should be used
        
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object heights, 0, or None (use classes)
        classes   - None or [d] tensor of object classes
        RETURN:     [d,m,3] array of points in space 
        """
        
        points_ds = points * self.downsample
        
        if heights is None:
            if classes is None: 
                raise IOError("Either heights or classes must not be None")
            else:
                heights = self.guess_heights(classes)
        if type(name) != list or len(name[0].split("_")) == 1:        
            boxes  = self._im_sp(points_ds,name = name, heights = 0)
            
            # get directions and append to names
            name = self.get_direction(boxes,name)[0]
            
        # recompute with correct directions
        boxes = self._im_sp(points_ds,name = name, heights = heights, refine_heights=refine_heights)
        
        
        return boxes
    
    def space_to_im(self, points, name = None):
        """
        Wrapper function on _sp_im necessary because it is not immediately evident 
        from points in image whether the EB or WB corespondence should be used
        
        name    - list of correspondence key names
        points  - [d,m,3] array of points in 3-space
        RETURN:   [d,m,2] array of points in 2-space
        """
        
        boxes  = self._sp_im(points,name = name)     
        boxes /= self.downsample
        
        return boxes
        
    def im_to_state(self,points, name = None, heights = None,refine_heights = True,classes = None):
        """
        Converts image boxes to roadway coordinate boxes
        points    - [d,m,2] array of points in image
        name      - list of correspondence key names
        heights   - [d] tensor of object heights
        RETURN:     [d,s] array of boxes in state space where s is state size (probably 6)
        """
        space_pts = self.im_to_space(points,name = name, classes = classes, heights = heights,refine_heights = refine_heights)
        return self.space_to_state(space_pts)
    
    def state_to_im(self,points,name = None):
        """
        Converts roadway coordinate boxes to image space boxes
        points    - [d,s] array of boxes in state space where s is state size (probably 6)
        name      - list of correspondence key names
        RETURN:   - [d,m,2] array of points in image
        """
        space_pts = self.state_to_space(points)
        return self.space_to_im(space_pts,name = name)
    
    def space_to_state(self,points):
        """        
        Conversion from state plane coordinates to roadway coordinates via the following steps:
            1. If spline points aren't yet cached, cache them
            2. Convert space points to L,W,H,x_back,y_center
            3. Search coarse grid for best fit point for each point
            4. Search fine grid for best offset relative to each coarse point
            5. Final state space obtained
        
        Note that by convention 3D box point ordering  = fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl and roadway coordinates reference back center of vehicle

        
        points - [d,m,3] 
        RETURN:  [d,s] array of boxes in state space where s is state size (probably 6)
        """
        
        
        # 2. Convert space points to L,W,H,x_back,y_center
        d = points.shape[0]
        n_pts = points.shape[1]
        
        if d == 0:
            return torch.empty([0,6], device = points.device)
        
        new_pts = torch.zeros([d,6],device = points.device)
        
        # rear center bottom of vehicle is (x,y)
        
        if n_pts == 8:
            # x is computed as average of two bottom rear points
            new_pts[:,0] = (points[:,2,0] + points[:,3,0]) / 2.0
            
            # y is computed as average of two bottom rear points 
            new_pts[:,1] = (points[:,2,1] + points[:,3,1]) / 2.0

            
            # l is computed as avg length between bottom front and bottom rear
            #new_pts[:,2] = torch.abs ( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 )
            new_pts[:,2] = torch.pow((points[:,[0,1,4,5],:] - points[:,[2,3,6,7],:]).mean(dim = 1),2).sum(dim = 1).sqrt()
            
            # w is computed as avg length between botom left and bottom right
            new_pts[:,3] = torch.pow((points[:,[0,2,4,6],:] - points[:,[1,3,5,7],:]).mean(dim = 1),2).sum(dim = 1).sqrt()

            # h is computed as avg length between all top and all bottom points
            new_pts[:,4] = torch.mean(torch.abs( (points[:,0:4,2] - points[:,4:8,2])),dim = 1)
        
        else:
            new_pts[:,0] = points[:,0,0]
            new_pts[:,1] = points[:,0,1]
            
        
        # direction is +1 if vehicle is traveling along direction of increasing x, otherwise -1
        directions = self.get_direction(points)[1]
        new_pts[:,5] = directions #torch.sign( ((points[:,0,0] + points[:,1,0]) - (points[:,2,0] + points[:,3,0]))/2.0 ) 
        

        min_u = self.closest_spline_point(new_pts[:,:2])
        min_u = torch.from_numpy(min_u)
        spl_x,spl_y = interpolate.splev(min_u,self.median_tck)
        spl_x,spl_y = torch.from_numpy(spl_x).to(points.device),torch.from_numpy(spl_y).to(points.device)
        min_dist = torch.sqrt((spl_x - new_pts[:,0])**2 + (spl_y - new_pts[:,1])**2)
        
        new_pts[:,0] = min_u
        new_pts[:,1] = min_dist
        
        # if direction is -1 (WB), y coordinate is negative
        new_pts[:,1] *= new_pts[:,5]
        
        new_pts[:,5] *= self.polarity
        
        if self.yellow_offsets is not None:
            # shift so that yellow lines have constant y-position
            bins = (new_pts[:,0] / 10).int()
            bins = torch.clamp(bins,min = 0, max = len(self.yellow_offsets["WB"])-1)

            bins = bins.data.cpu().numpy()
            
            eb_offsets = self.yellow_offsets["EB"][bins] -12
            wb_offsets = self.yellow_offsets["WB"][bins] +12 
            eb_mask = torch.where(new_pts[:,5] == 1, 1, 0).to("cpu")
            wb_mask = 1- eb_mask
            
            yellow_offsets = wb_mask * wb_offsets + eb_mask * eb_offsets
            yellow_offsets = yellow_offsets.to(new_pts.device)
            new_pts[:,1] = new_pts[:,1] - yellow_offsets

        # 5. Final state space obtained
        return new_pts
        
    def state_to_space(self,points):
        """
        Conversion from state plane coordinates to roadway coordinates via the following steps:
            1. get x-y coordinate of closest point along spline (i.e. v = 0)
            2. get derivative of spline at that point
            3. get perpendicular direction at that point
            4. Shift x-y point in that direction
            5. Offset base points in each constituent direction
            6. Add top points
            
        Note that by convention 3D box point ordering  = fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl and roadway coordinates reference back center of vehicle
        
        points - [d,s] array of boxes in state space where s is state size (probably 6)
        
        """
        
        # 0. Un-offset points by yellow lines
        if self.yellow_offsets is not None:
            # shift so that yellow lines have constant y-position
            bins = (points[:,0] / 10).int()
            bins = torch.clamp(bins,min = 0, max = len(self.yellow_offsets["WB"])-1)

            bins = bins.data.cpu().numpy()

            eb_offsets = self.yellow_offsets["EB"][bins] -12
            wb_offsets = self.yellow_offsets["WB"][bins] +12 
            eb_mask = torch.where(points[:,5] == 1, 1, 0).to("cpu")
            wb_mask = 1- eb_mask
            
            yellow_offsets = wb_mask * wb_offsets + eb_mask * eb_offsets
            yellow_offsets = yellow_offsets.to(points.device)
            points[:,1] = points[:,1] + yellow_offsets
        
        # 1. get x-y coordinate of closest point along spline (i.e. v = 0)
        
        #### WARNING, this is untested!!!
        points = points.view(-1,points.shape[-1])
        
        d = points.shape[0]
        
        try:
            points[:,5] *= self.polarity
        except:
            print("Error in state_to_space, points is of dimension: {}".format(points.shape))
            return torch.empty([0,8,3], device = points.device)
        
        if d == 0:
            return torch.empty([0,8,3], device = points.device)
        
        closest_median_point_x, closest_median_point_y = interpolate.splev(points[:,0].cpu(),self.median_tck)
        
        # 2. get derivative of spline at that point
        l_direction_x,l_direction_y          = interpolate.splev(points[:,0].cpu(),self.median_tck, der = 1)

        # 3. get perpendicular direction at that point
        #w_direction_x,w_direction_y          = -1/l_direction_x  , -1/l_direction_y
        w_direction_x,w_direction_y = l_direction_y,-l_direction_x
        
        # numpy to torch - this is not how you should write this but I was curious
        [closest_median_point_x,
        closest_median_point_y,
        l_direction_x,
        l_direction_y,
        w_direction_x,
        w_direction_y] = [torch.from_numpy(arr).to(points.device) for arr in [closest_median_point_x,
                                                           closest_median_point_y,
                                                           l_direction_x,
                                                           l_direction_y,
                                                           w_direction_x,
                                                           w_direction_y]]
        direction = torch.sign(points[:,1])
                                                            
        # 4. Shift x-y point in that direction
        hyp_l = torch.sqrt(l_direction_x**2 + l_direction_y **2)
        hyp_w = torch.sqrt(w_direction_x**2 + w_direction_y **2)
        
        # shift associated with the length of the vehicle, in x-y space
        x_shift_l    = l_direction_x /hyp_l * points[:,2] * direction
        y_shift_l    = l_direction_y /hyp_l * points[:,2] * direction
        
        # shift associated with the width of the vehicle, in x-y space
        x_shift_w    = w_direction_x/hyp_w * (points[:,3]/2.0) * direction
        y_shift_w    = w_direction_y/hyp_w * (points[:,3]/2.0) * direction
        
        # shift associated with the perp distance of the object from the median, in x-y space
        x_shift_perp = w_direction_x/hyp_w * points[:,1] # * direction # points already incorporates sign you dingus!!
        y_shift_perp = w_direction_y/hyp_w * points[:,1] # * direction
        
        # 5. Offset base points in each constituent direction
        
        new_pts = torch.zeros([d,4,3],device = points.device)
        
        # shift everything to median point
        new_pts[:,:,0] = closest_median_point_x.unsqueeze(1).expand(d,4) + x_shift_perp.unsqueeze(1).expand(d,4)
        new_pts[:,:,1] = closest_median_point_y.unsqueeze(1).expand(d,4) + y_shift_perp.unsqueeze(1).expand(d,4)
        
        # shift front points
        new_pts[:,[0,1],0] += x_shift_l.unsqueeze(1).expand(d,2)
        new_pts[:,[0,1],1] += y_shift_l.unsqueeze(1).expand(d,2)
        
        new_pts[:,[0,2],0] += x_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[0,2],1] += y_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[1,3],0] -= x_shift_w.unsqueeze(1).expand(d,2)
        new_pts[:,[1,3],1] -= y_shift_w.unsqueeze(1).expand(d,2)
    
        #6. Add top points
        top_pts = new_pts.clone()
        top_pts[:,:,2] += points[:,4].unsqueeze(1).expand(d,4)
        new_pts = torch.cat((new_pts,top_pts),dim = 1)
        
        
        
        #test
        #test_direction = self.get_direction(new_pts)[1]
        
        return new_pts
    
    def get_direction(self,points,name = None,method = "angle"):
        """
        Find closest point on spline. use relative x_coordinate difference (in state plane coords) to determine whether point is below spline (EB) or above spline (WB)
        
        
        points    - [d,m,3] array of points in space
        name      - list of correspondence key names. THey are not needed but if supplied the correct directions will be appended to them
        RETURN:   - [d] list of names with "EB" or "WB" added, best guess of which side of road object is on and which correspondence should be used
                  - [d] tensor of int with -1 if "WB" and 1 if "EB" per object
        """
        mean_points = torch.mean(points, dim = 1).cpu()
        min_u  = self.closest_spline_point(mean_points)
        
        if method == "x_position":
            spl_x,_ = interpolate.splev(min_u, self.median_tck)
            
            direction = (torch.sign(torch.from_numpy(spl_x) - mean_points[:,0]).int().to(points.device))
        
        elif method == "angle":
            spl_x,spl_y         =  interpolate.splev(min_u    , self.median_tck)
            x_forward,y_forward =  interpolate.splev(min_u+250, self.median_tck)
            
            # now we have 3 points defining 2 vectors and one directioned angle
            # vector A -  closest point -> 250 feet ahead on spline 
            vec_A = torch.from_numpy(x_forward - spl_x) , torch.from_numpy( y_forward - spl_y)
            
            # vector B - closest point -> point
            vec_B = mean_points[:,0] - spl_x, mean_points[:,1] - spl_y
            
            # if angle AB is in [0,180], object is on WB sidez, otherwise on EB side
            # from https://stackoverflow.com/questions/2150050/finding-signed-angle-between-vectors
            angle = torch.atan2( vec_A[0]*vec_B[1] - vec_A[1]*vec_B[0], vec_A[0]*vec_B[0] + vec_A[1]*vec_B[1] )
            direction =  -1* torch.sign(angle).int().to(points.device)
        
        d_list = ["dummy which should never appear","EB","WB"]
        string_direction = [d_list[di.item()] for di in direction]
        
    
        new_name = None                        
        if name is not None:
            
            if type(name) == str:
                new_name = name + "_{}".format(string_direction[0])
            
            elif type(name) == list:
                new_name = []
                for n_idx in range(len(name)):
                    new_name.append(name[n_idx] + "_{}".format(string_direction[n_idx]))
            
        return new_name, direction
    
    
    def guess_heights(self,classes):
        """
        classes - [d] vector of string class names
        
        returns - [d] vector of float object height guesses
        """
        
        heights = torch.zeros(len(classes))
        
        for i in range(len(classes)):
            try:
                heights[i] = self.class_heights[classes[i]]
            except KeyError:
                heights[i] = self.class_heights["other"]
            
        return heights
    
    def height_from_template(self,template_boxes,template_space_heights,boxes):
        """
        Predicts space height of boxes in image space. Given a space height and 
        the corresponding image box (and thus image height), the relationship 
        between heights in different coordinate systems should be roughly estimable. 
        This strategy is used to guess the heights of the second set of boxes in
        image space according to : 
            template_im_heights:template_space_heights = new_im_heights:new_box heights
            
        template_boxes - [d,m,2,] array of points corresponding to d object boxes 
                         (typical usage would be to use boxes from previous frame
                         or apriori box predictions for current frame))
        template_space_heights - [d] array of corresponding object heights in space
        boxes - [d,m,2] array of points in image
        
        returns
        
        height - [d] array of object heights in space
        """
        
        # get rough heights of objects in image
        template_top = torch.mean(template_boxes[:,4:8,:],dim = 1)
        template_bottom = torch.mean(template_boxes[:,0:4,:],dim = 1)
        template_im_height = torch.sum(torch.sqrt(torch.pow((template_top - template_bottom),2)),dim = 1) *self.downsample
        template_ratio = template_im_height / template_space_heights.to(boxes.device)
        
        box_top    = torch.mean(boxes[:,4:8,:2],dim = 1)
        box_bottom = torch.mean(boxes[:,0:4,:2],dim = 1)
        box_height = torch.sum(torch.sqrt(torch.pow((box_top - box_bottom),2)),dim = 1)
        
        height = box_height / template_ratio
        return height

    
         
    
    #%% Plotting Functions
    
    def plot_boxes(self,im,boxes,color = (255,255,255),labels = None,thickness = 1, ORIGIN = True):
        """
        As one might expect, plots 3D boxes on input image
        
        im - cv2 matrix-style image
        boxes - [d,8,2] array of image points where d indexes objects
        color - 3-tuple specifying box color to plot
        """
                
        DRAW = [[0,1,2,0,1,0,0,0], #bfr
                [0,0,0,1,0,1,0,0], #bfl
                [0,0,0,3,0,0,4,1], #bbr
                [0,0,0,0,0,0,1,1], #bbl
                [0,0,0,0,0,1,1,0], #tfr
                [0,0,0,0,0,0,0,1], #tfl
                [0,0,0,0,0,0,0,1], #tbr
                [0,0,0,0,0,0,0,0]] #tbl
        
        DRAW_BASE = [[0,1,1,1], #bfr
                     [0,0,1,1], #bfl
                     [0,0,0,1], #bbr
                     [0,0,0,0]] #bbl
        
        for idx, bbox_3d in enumerate(boxes):
            
            # check whether box mostly falls within frame
            if torch.min(bbox_3d[:,0]) < -1000 or torch.max(bbox_3d[:,0]) > 3840+1000 or torch.min(bbox_3d[:,1]) < -1000 or torch.max(bbox_3d[:,1]) > 2160+1000:
                continue
            
            if type(color) == np.ndarray:
                c = (int(color[idx,0]),int(color[idx,1]),int(color[idx,2]))
            else:
                c = color
            

            
            for a in range(len(bbox_3d)):
                ab = bbox_3d[a]
                for b in range(a,len(bbox_3d)):
                    bb = bbox_3d[b]
                    if DRAW[a][b] == 1:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),c,thickness)
                    elif DRAW[a][b] == 2:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(0,255,0),thickness)
                    elif DRAW[a][b] == 3:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(255,0,0),thickness)
                    elif DRAW[a][b] == 4:
                            im = cv2.line(im,(int(ab[0]),int(ab[1])),(int(bb[0]),int(bb[1])),(0,0,255),thickness)
        
            if labels is not None:
                label = labels[idx]
                left  = bbox_3d[0,0]
                top   = bbox_3d[0,1]
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),3)
                im    = cv2.putText(im,"{}".format(label),(int(left),int(top - 10)),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),1)
            
            # draw origin
            if ORIGIN:
                bbl = bbox_3d[2,:]
                bbl = int(bbl[0]),int(bbl[1])
                im = cv2.circle(im,bbl,2,(255,255,255),-1)
            
        return im
        
    def plot_state_boxes(self,im,boxes,color = (255,255,255),labels = None, thickness = 1, name = None):
        """
        Wraps plot_boxes for state boxes by first converting from state (roadway coordinates) to image coordinates
        """
        im_boxes = self.state_to_im(boxes, name = name)
        return self.plot_boxes(im,im_boxes,color = color,labels = labels, thickness = thickness)
    
    def plot_points(self,im,points, color = (0,255,0)):
        """
        Lazily, duplicate each point 8 times as a box with 0 l,w,h then call plot_boxes
        points -  [d,2] array of x,y points in roadway coordinates / state 
        """
        rep_points = torch.cat(points,torch.zeros(points.shape[0],3),dim = 1)
        space_points = self.state_to_space(rep_points)
        self.plot_space_points(im,space_points,color = color)
    
    def plot_space_points(self,im,points,color = (255,0,0), name = None):
        """
        points -  [d,n,3] array of x,y points in roadway coordinates / state 
        """
        
        im_pts = self.space_to_im(points, name = name)
        
        for point in im_pts:
            cv2.circle(im,(int(point[0]),int(point[1])),2,color,-1)
        
        cv2.imshow("frame",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def fill_gaps(self):
        additions = {}
        for key in self.correspondence.keys():
            base, direction = key.split("_")
            if direction == "EB":
                new_key = base + "_WB"
                if new_key not in self.correspondence.keys():
                    additions[new_key] = self.correspondence[key].copy()
            elif direction == "WB":
                new_key = base + "_EB"
                if new_key not in self.correspondence.keys():
                    additions[new_key] = self.correspondence[key].copy()
        print("Adding {} missing correspondences: {}".format(len(additions),list(additions.keys())))
        for key in additions.keys():
            self.correspondence[key] = additions[key]
        
    def plot_homography(self,
                        MEDIAN  = False,
                        IM_PTS  = False,
                        FIT_PTS = False,
                        FOV     = False,
                        MASK    = False,
                        Z_AXIS  = False):
        pass
    
    
    #%% Testing Functions
    
    def test_transformation(self,im_dir):
        """
        A series of tests will be run
        
        1. Transform a set of image points into space then state, then back to space then image
           - get average and max error (in pixels)
           
        2. Transform a set of randomly created boxes into space then image, then back to space then state
           - get average state error (in feet)
           
        3. Transform the same set of boxes back into image, and get average pixel error
        """
        
        if True:
            print("___________________________________")
            print("Test 0: Get Direction Test")
            print("___________________________________")
            
            correct = 0
            EB_incorrect = 0 # points on EB side that are incorrectly called WB
            WB_incorrect = 0 # points on WB side that are incorrectly called EB
            
            for name in self.correspondence.keys():
                corr = self.correspondence[name]
                direction = name.split("_")[1]
                #print(direction)
                space_pts = torch.from_numpy(corr["space_pts"]).unsqueeze(1)
                space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1,1])), dim = -1)
                
                pred = self.get_direction(space_pts)[1]
                
                EB = torch.where(pred == 1,1,0).sum()
                WB = torch.where(pred == -1,1,0).sum()
                
                if direction == "EB":
                    correct += EB
                    if "c3" not in name and "c4" not in name:
                        EB_incorrect += WB

                elif direction == "WB":
                    correct += WB
                    if "c3" not in name and "c4" not in name:
                        WB_incorrect += EB
                    
            print("Correct side: {},   EB incorrect: {}, WB incorrect: {}".format(correct,EB_incorrect,WB_incorrect))
        
        ### Project each aerial imagery point into pixel space and get pixel error
        if True:
            print("___________________________________")
            print("Test 1: Pixel Reprojection Error")
            print("___________________________________")
            start = time.time()
            running_error = []
            for name in self.correspondence.keys():
                corr = self.correspondence[name]
                #name = name.split("_")[0]
                
                space_pts = torch.from_numpy(corr["space_pts"]).unsqueeze(1)
                space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1,1])), dim = -1)
                
                im_pts    = torch.from_numpy(corr["corr_pts"])
                
                #name = name.split("_")[0]
                namel = [name for _ in range(len(space_pts))]
                
                proj_space_pts = self.space_to_im(space_pts,name = namel).squeeze(1)
                error = torch.sqrt(((proj_space_pts - im_pts)**2).sum(dim = 1)).mean()
                #print("Mean error for {}: {}px".format(name,error))
                running_error.append(error)   
            end = time.time() - start
            print("Average Pixel Reprojection Error across all homographies: {}px in {} sec\n".format(sum(running_error)/len(running_error),end))
                
        
        ### Project each camera point into state plane coordinates and get ft error
        if True:
            print("___________________________________")
            print("Test 2: State Reprojection Error")
            print("___________________________________")
            running_error = []
            
            all_im_pts = []
            all_cam_names = []
            for name in self.correspondence.keys():
                corr = self.correspondence[name]
                #name = name.split("_")[0]
    
                
                space_pts = torch.from_numpy(corr["space_pts"])
                space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1])), dim = -1)
    
                im_pts    = torch.from_numpy(corr["corr_pts"]).unsqueeze(1).float()
                namel = [name for _ in range(len(space_pts))]
    
                all_im_pts.append(im_pts)
                all_cam_names += namel
    
                proj_im_pts = self.im_to_space(im_pts,name = namel, heights = 0, refine_heights = False).squeeze(1)
                
                error = torch.sqrt(((proj_im_pts - space_pts)**2).sum(dim = 1)).mean()
                #print("Mean error for {}: {}ft".format(name,error))
                running_error.append(error)
            print("Average Space Reprojection Error across all homographies: {}ft\n".format(sum(running_error)/len(running_error)))
            

        
        if True:
            ### Create a random set of boxes
            boxes = torch.rand(1000,6) 
            boxes[:,0] = boxes[:,0] * (self.median_u[-1] - self.median_u[0]) + self.median_u[0]
            boxes[:,1] = boxes[:,1] * 120 - 60
            boxes[:,2] *= 60
            boxes[:,3] *= 10
            boxes[:,4] *= 10
            boxes[:,5] = self.polarity * torch.sign(boxes[:,1])

            
            space_boxes = self.state_to_space(boxes)
            repro_state_boxes = self.space_to_state(space_boxes)
            
            error = torch.abs(repro_state_boxes - boxes)
            mean_error = error.mean(dim = 0)
            print("Mean State-Space-State Error: {}ft\n".format(mean_error))
        
        if True:
            print("___________________________________")
            print("Test 3: Random Box Reprojection Error (State-Im-State)")
            print("___________________________________")
            all_im_pts = torch.cat(all_im_pts,dim = 0)
            start = time.time()
            state_pts = self.im_to_state(all_im_pts, name = all_cam_names, heights = 0,refine_heights = False)
            print("im->state took {}s for {} boxes".format(time.time()-start, all_im_pts.shape[0]))
            ### Create a random set of boxes
            boxes = torch.rand(state_pts.shape[0],6) 
            boxes[:,:2] = state_pts[:,:2]
            #boxes[:,1] = boxes[:,1] * 240 - 120
            boxes[:,2] *= 30
            boxes[:,3] *= 10
            boxes[:,4] *= 10
            boxes[:,5] = self. polarity * torch.sign(boxes[:,1])
            
            #boxes[:,1] = 0.01
            # boxes[:,2] = 30
            # boxes[:,3:5] = .01
            
            # get appropriate camera for each object
            
            start = time.time()
            im_boxes = self.state_to_im(boxes,name = all_cam_names)
            print("state->im took {}s for {} boxes".format(time.time()-start, all_im_pts.shape[0]))

            # plot some of the boxes
            repro_state_boxes = self.im_to_state(im_boxes,name = all_cam_names,heights = boxes[:,4],refine_heights = True)
            
            
            error = torch.abs(repro_state_boxes - boxes)
            error = torch.mean(error,dim = 0)
            print("Average State-Im_State Reprojection Error: {} ft\n".format(error))
        
        
        if True:
            print("___________________________________")
            print("Test 4: Random Box Reprojection Error (Im-State-Im)")
            print("___________________________________")
            repro_im_boxes = self.state_to_im(repro_state_boxes, name = all_cam_names)
            
            error = torch.abs(repro_im_boxes-im_boxes)
            error = torch.mean(error)
            print("Average Im-State-Im Reprojection Error: {} px\n".format(error))
        
        if False: 
            #plot some boxes
            for pole in [1,2,3,4,5,6]:
                for camera in range(1,7):
                    try:
                        # pole = 40
                        # camera  = 2
                        plot_cam = "P{}C{}".format(str(pole).zfill(2),str(camera).zfill(2))
                        print(plot_cam)
                        im_path = os.path.join(im_dir,plot_cam) + ".png"
                        im = cv2.imread(im_path)
                        plot_boxes = []
                        plot_boxes_state = []
                        labels = []
                        for i in range(len(all_cam_names)):
                            if plot_cam in all_cam_names[i]:
                                plot_boxes.append(repro_state_boxes[i])
                                plot_boxes_state.append(boxes[i])     
                                labels.append(boxes[i,5])
                        name = [plot_cam for i in range(len(plot_boxes))]
                        
                        plot_boxes_state = torch.stack(plot_boxes_state)
                        im = self.plot_state_boxes(im,plot_boxes_state,color = (255,0,0), name = name,labels = labels)
                        
                        #plot_boxes = torch.stack(plot_boxes)
                        #im = self.plot_state_boxes(im, plot_boxes,color = (0,0,255),name = name)
                        
                        
                        
                        cv2.imshow("Frame", im)
                        cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                    except:
                        pass
               
            
        
#%% MAIN        
    
if __name__ == "__main__":
    
    for day in ["Wednesday"]:#,"Tuesday","Thursday","Friday"]:
        print("Generating Homography for {}".format(day))
        #im_dir = "/home/derek/Documents/i24/i24_homography/data_real"
        #space_dir = "/home/derek/Documents/i24/i24_homography/aerial/to_P24"
        #save_file =  "P01_P40b.cpkl"
    
        #im_dir = "/home/derek/Data/MOTION_HOMOGRAPHY_FINAL"
        space_dir = "/home/worklab/Documents/i24/i24_rcs/aerial/all_poles_aerial_labels"
        #space_dir = "/home/derek/Documents/i24/i24_homography/shifted_aerial_points"
    
        
        save_file = "CIRCLES_20_{}_1hour.cpkl".format(day)
        #im_dir = "/home/derek/Data/homo/working"
        im_dir = "/home/worklab/Data/homo/CIRCLES_20pre_{}_1hour".format(day)
    
        hg = Curvilinear_Homography(save_file = save_file,space_dir = space_dir, im_dir = im_dir,downsample = 1)
    
        hg._generate_lane_offset(space_dir,SHIFT = False,SPLINE_OFFSET = False)
        # hg._convert_landmarks(space_dir)
        hg.test_transformation(im_dir+"/4K")
    
        # #hg._generate_extents_file(im_dir)
        # #hg._generate_mask_images(im_dir,mask_save_dir = "/home/derek/Data/ICCV_2023/masks/scene3")
        # #hg._generate_extents_file(im_dir,mode = "", output_path = "cam_extents_polygon.json")
        # hg._fit_MM_offset(space_dir)
        # print("MM offset: {}".format(hg.MM_offset))
        # hg.save(save_file)
