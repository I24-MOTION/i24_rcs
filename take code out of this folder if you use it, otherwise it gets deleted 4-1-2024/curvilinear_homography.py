#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:58:31 2022

@author: derek
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

from scipy import interpolate


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
                 im_dir = None):
        """
        Initializes homography object.
        
        save_file - None or str - if str, specifies path to cached homography object
        space_dir - None or str - path to directory with csv files of attributes labeled in space coordinates
        im_dir    - None or str - path to directory with cpkl files of attributes labeled in image coordinates
        """
        
        # intialize correspondence
        
        self.correspondence = {}
        if save_file is not None and os.path.exists(save_file):
            with open(save_file,"rb") as f:
                # everything in correspondence is pickleable without object definitions to allow compatibility after class definitions change
                self.correspondence,self.median_tck,self.median_u,self.guess_tck = pickle.load(f)
            
            # reload  parameters of curvilinear axis spline
            # rather than the spline itself for better pickle reloading compatibility
                
        
        elif space_dir is None or im_dir is None:
            raise IOError("Either save_file or space_dir and im_dir must not be None")
        
        else:
            self.generate(space_dir,im_dir)
            self.median_tck = None
            self.median_u   = None
            self.guess_tck  = None
            self.save(save_file)
            
            #  fit the axis spline once and collect extents
            self._fit_spline(space_dir)
            self.save(save_file)
        
        


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
        with open(save_file,"wb") as f:
            pickle.dump([self.correspondence,self.median_tck,self.median_u,self.guess_tck],f)
        
        
    def generate(self,
                 space_dir,
                 im_dir,
                 downsample     = 1,
                 max_proj_error = 0.25,
                 scale_factor   = 3,
                 ADD_PROJ       = False):
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
            
            if ADD_PROJ:
                
                try:
                    with open("ae_cache_{}.cpkl".format(direction),"rb") as f:
                        additional_points = pickle.load(f)
                except:
                    # For each d2 lane marker, find the closest d3 lane marker
                    proj_lines = []
                    
                    for i in range(len(min_matches)):
                        j = min_matches[i]
                        pline = [d3_values[j],d2_values[i],d3_ids[j],d2_ids[i]]
                        proj_lines.append(pline)
                    
                    
                    
                    # compute the yellow line spline in state plane coordinates
                    
                    ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                    ae_data = ae_data[:,np.argsort(ae_data[1,:])]
                    
                    ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
                    
                    span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                    ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*scale_factor)), ae_tck)
                
                    additional_points = []
                    # for each d2 lane marker, find the intersection between the d2-d3 line and the yellow line spline
                    for p_idx, proj_line in enumerate(proj_lines):
                        print("On proj line {} of {}".format(p_idx,len(proj_lines)))
                        min_dist = np.inf
                        min_point = None
                        line = [proj_line[0][0],proj_line[0][1],proj_line[1][0],proj_line[1][1]]
                        for i in range(len(ae_x_prime)):
                            point = [ae_x_prime[i],ae_y_prime[i]]
                            
                            dist = line_to_point(line, point)
                            if dist < min_dist:
                                min_dist = dist
                                min_point = point
                        if min_dist > max_proj_error:
                            print("Issue")
                        else:
                            name = "{}_{}".format(proj_line[2],proj_line[3])
                            min_point.append(name)
                            additional_points.append(min_point)
                            
                    with open("ae_cache_{}.cpkl".format(direction),"wb") as f:
                        pickle.dump(additional_points,f)
                        
                
                for point in additional_points:
                    ae_x.append(point[0])
                    ae_y.append(point[1])
                    ae_id.append(point[2])
    
    
        # get all cameras
        cam_data_paths = glob.glob(os.path.join(im_dir,"*.cpkl"))
        
        for cam_data_path in cam_data_paths:
            
            
            # specify path to camera imagery file
            cam_im_path   = cam_data_path.split(".cpkl")[0] + ".png"
            camera = cam_data_path.split(".cpkl")[0].split("/")[-1]
            
            if "46" in camera or "47" in camera or "48" in camera:
                print("excluded validation system pole")
                continue
            
            # load all points
            with open(cam_data_path, "rb") as f:
                im_data = pickle.load(f)
                
            for direction in ["EB","WB"]:
                # get all non-curve matching points
                point_data = im_data[direction]["points"]
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
            if "{}_{}".format(camera,"EB") not in self.correspondence.keys():
                if "{}_{}".format(camera,"WB") in self.correspondence.keys():
                    self.correspondence["{}_{}".format(camera,"EB")] = self.correspondence["{}_{}".format(camera,"WB")]
            if "{}_{}".format(camera,"WB") not in self.correspondence.keys():
                if "{}_{}".format(camera,"EB") in self.correspondence.keys():
                    self.correspondence["{}_{}".format(camera,"WB")] = self.correspondence["{}_{}".format(camera,"EB")]
    
    def _fit_z_vp(self,cor,im_data,direction):
        
        print("fitting Z coordinate scale")
        
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
        
    def _fit_spline(self,space_dir,use_MM_offset = False):
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
                
                
                ae_spl_x = []
                ae_spl_y = []
                ae_spl_u = []  # u parameterizes distance along spline 
                
                
                for i in range(len(ae_x)):
                    
                    if "yel{}".format(line_side) in ae_id[i]:
                        ae_spl_x.append(ae_x[i])
                        ae_spl_y.append(ae_y[i])

                # 2. Fit a spline to each of EB, WB inside and outside
                 
                # compute the yellow line spline in state plane coordinates (sort points by y value since road is mostly north-south)
                ae_data = np.stack([np.array(ae_spl_x),np.array(ae_spl_y)])
                ae_data = ae_data[:,np.argsort(ae_data[1,:],)[::-1]]
            
                # 3. Sample the spline at fine intervals
                # get spline and sample points on spline
                ae_tck, ae_u = interpolate.splprep(ae_data, s=0, per=False)
                span_dist = np.sqrt((ae_spl_x[0] - ae_spl_x[-1])**2 + (ae_spl_y[0] - ae_spl_y[-1])**2)
                ae_x_prime, ae_y_prime = interpolate.splev(np.linspace(0, 1, int(span_dist*samples_per_foot)), ae_tck)
            

                # 4. Use finite difference method to determine the distance along the spline for each fit point
                fd_dist = np.concatenate(  (np.array([0]),  ((ae_x_prime[1:] - ae_x_prime[:-1])**2 + (ae_y_prime[1:] - ae_y_prime[:-1])**2)**0.5),axis = 0) # by convention fd_dist[0] will be 0, so fd_dist[i] = sum(int_dist[0:i])
                integral_dist = np.cumsum(fd_dist)
                
                # for each fit point, find closest point on spline, and assign it the corresponding integral distance
                for p_idx in range(len(ae_spl_x)):
                    px = ae_spl_x[p_idx]
                    py = ae_spl_y[p_idx]
                    
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
                
                tck, u = interpolate.splprep(ae_data.astype(float), s=0, u = ae_spl_u)
                splines["{}_{}".format(direction,line_side)] = [tck,u]
                
            
        # to prevent any bleedover
        del dist, min_dist, min_idx, ae_spl_y,ae_spl_x, ae_spl_u, ae_data
           
        # 6. Sample each of the 4 splines at fine intervals (every 2 feet)
        for key in splines:
            tck,u = splines[key]

            span_dist = np.abs(u[0] - u[-1])
            x_prime, y_prime = interpolate.splev(np.linspace(u[0], u[-1], int(span_dist/2)), tck)
            splines[key].append(x_prime)
            splines[key].append(y_prime)
            
        med_spl_x = []
        med_spl_y = []
        
        
        # 7. Move along one spline and at each point, find the closest point on each other spline
        # by default, we'll use EB_o as the base spline
        main_key = "EB_o"
        main_spl = splines[main_key]
        main_x = main_spl[2]
        main_y = main_spl[3]
        
        
        for p_idx in range(len(main_x)):
            px,py = main_x[p_idx],main_y[p_idx]
            points_to_average = [np.array([px,py])]
            
            for key in splines:
                if key != main_key:
                    arr_x,arr_y = splines[key][2], splines[key][3]
                    
                    dist = np.sqrt((arr_x - px)**2 + (arr_y - py)**2)
                    min_dist,min_idx= np.min(dist),np.argmin(dist)
                    
                    points_to_average.append( np.array([arr_x[min_idx],arr_y[min_idx]]))
            
            if len(points_to_average) != 4:
                print("Outlier removed")
                continue
            
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
        
        s = 10
        n_knots = len(med_data[0])
        while n_knots > 600:
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
        plt.figure(figsize = (20,20))
        plt.plot(med_data[0],med_data[1])
        plt.figure()
        plt.plot(med_spl_u)
        plt.plot(np.array(med_spl_u)[np.argsort(med_spl_u)])
        
        
        s = 10.0
        n_knots = len(med_data[0])
        while n_knots > 600:
            final_tck,final_u = interpolate.splprep(med_data.astype(float), s=s, u=med_spl_u)
            n_knots = len(med_tck[0])
            s = s**1.2
        
        #final_tck, final_u = interpolate.splprep(med_data, u = med_spl_u)
        self.median_tck = final_tck
        self.median_u = final_u
        
        if use_MM_offset:
            # 11. Optionally, compute a median spline distance offset from mile markers
            self.MM_offset = self._fit_MM_offset()
        
            # 12. Optionally, recompute the same spline, this time accounting for the MM offset
            med_spl_u += self.MM_offset
            final_tck, final_u = interpolate.splprep(med_data.astype(float), s=0, u = med_spl_u)
            self.median_tck = final_tck
            self.median_u = final_u
        
        # get the inverse spline g(x) = u for guessing initial spline point
        med_spl_u = np.array(med_spl_u)
        print(med_data.shape,med_spl_u.shape)


        # sort by strictly increasing x
        sorted_idxs = np.argsort(med_data[0])
        med_data = med_data[:,sorted_idxs]
        med_spl_u = med_spl_u[sorted_idxs]

        self.guess_tck = interpolate.splrep(med_data[0].astype(float),med_spl_u.astype(float))
        
        
    
    
    def closest_spline_point(self,points, epsilon = 0.01, max_iterations = 100):
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
        start = time.time()
        # intial guess at closest u values
        points = points.data.numpy()
        guess_u = interpolate.splev(points[:,0],self.guess_tck)
        
        it = 0
        max_change = np.inf
        while it < max_iterations and max_change > epsilon:
            spl_x,spl_y             = interpolate.splev(guess_u,self.median_tck)
            spl_xx,spl_yy = interpolate.splev(guess_u,self.median_tck, der = 1)
            spl_xxx,spl_yyy = interpolate.splev(guess_u,self.median_tck, der = 2)

            
            dist_proxy = (spl_x - points[:,0])**2 + (spl_y - points[:,1])**2
            dist_proxy_deriv = (spl_x-points[:,0])*spl_xx + (spl_y-points[:,1])*spl_yy
            dist_proxy_deriv2 = (2*spl_xx**2)+2*(spl_x-points[:,0])*spl_xxx + (2*spl_yy**2)+2*(spl_y-points[:,1])*spl_yyy
            
            
            new_u = guess_u - dist_proxy_deriv/dist_proxy_deriv2
            
            max_change = np.max(np.abs(new_u-guess_u))
            it += 1
            
            guess_u = new_u
            
            #print("Max step: {}".format(max_change))
         
        #print("Newton method took {}s for {} points".format(time.time() - start,points.shape[0]))
        return guess_u
            
    
    def _fit_MM_offset(self):
        return 0
    
    def _generate_extents_file(self,im_dir,output_path = "cam_extents.config"):
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
        for key in data.keys():
            key_data = data[key]
            minx = torch.min(key_data[:,0]).item()
            maxx = torch.max(key_data[:,0]).item()
            miny = torch.min(key_data[:,1]).item()
            maxy = torch.max(key_data[:,1]).item()
            
            extents[key] = [minx,maxx,miny,maxy]
        
        
           
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
        with open(output_path,"w",encoding='utf-8') as f:
            for key in keys:
                key_data = extents[key]
                line = "{}={},{},{},{}\n".format(key,int(key_data[0]),int(key_data[1]),int(key_data[2]),int(key_data[3]))
                f.write(line)

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
                    
                    mask_poly = np.array([pt for pt in mask]).reshape(
                        1, -1, 2).astype(np.int32)
                    mask_im= cv2.fillPoly(
                        mask_im, mask_poly,  255, lineType=cv2.LINE_AA)
                    
                    save_name = os.path.join(mask_save_dir,"{}_mask.png".format(camera))
                    cv2.imwrite(save_name,mask_im)
                    
                    mask_im = cv2.resize(mask_im,(1920,1090))
                    save_name2 = os.path.join(mask_save_dir,"{}_mask_1080.png".format(camera))
                    cv2.imwrite(save_name2,mask_im)
                    
                except:
                    pass
        
    

    
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
        if name is None:
            name = list(self.correspondence.keys())[0]
        
        if type(name) == list and len(name[0].split("_")) == 1:
            temp_name = [sub_n+ "_WB" for sub_n in name]
            name = temp_name
            
        d = points.shape[0]
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
                    new_pts[:,[4,5,6,7],2] = heights.unsqueeze(1).repeat(1,4).double()
            
        else:
            print("No heights were input")
            return
        
        if refine_heights:
            template_boxes = self.space_to_im(new_pts,name)
            heights_new = self.height_from_template(template_boxes, heights, points.view(d,8,3))
            new_pts[:,[4,5,6,7],2] = heights_new.unsqueeze(1).repeat(1,4).double()
            
        return new_pts
    
    @safe_name
    def _sp_im(self,points, name = None, direction = "EB"):
       """
       Projects 3D space points into image/correspondence using P:
           new_pts = P x points T  ---> [dm,3] T = [3,4] x [4,dm]
       performed by flattening batch dimension d and object point dimension m together
       
       name      - list of correspondence key names
       direction - "EB" or "WB" - speecifies which correspondence to use
       points    - [d,m,3] array of points in 3-space
       RETURN:     [d,m,2] array of points in 2-space
       """
       if name is None:
           name = list(self.correspondence.keys())[0]
       
       d = points.shape[0]
       n_pts = points.shape[1]
       # get directions and append to names

       if type(name) == list and len(name[0].split("_")) == 1:
           name = self.get_direction(points,name)[0]

           
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
        
        if heights is None:
            if classes is None: 
                raise IOError("Either heights or classes must not be None")
            else:
                heights = self.guess_heights(classes)
                
        boxes  = self._im_sp(points,name = name, heights = 0)
        
        # get directions and append to names
        name = self.get_direction(points,name)[0]
        
        # recompute with correct directions
        boxes = self._im_sp(points,name = name, heights = heights, refine_heights=refine_heights)
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
        spl_x,spl_y = torch.from_numpy(spl_x),torch.from_numpy(spl_y)
        min_dist = torch.sqrt((spl_x - new_pts[:,0])**2 + (spl_y - new_pts[:,1])**2)
        
        new_pts[:,0] = min_u
        new_pts[:,1] = min_dist
        
        # if direction is -1 (WB), y coordinate is negative
        new_pts[:,1] *= new_pts[:,5]

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
        
        # 1. get x-y coordinate of closest point along spline (i.e. v = 0)
        d = points.shape[0]
        closest_median_point_x, closest_median_point_y = interpolate.splev(points[:,0],self.median_tck)
        
        # 2. get derivative of spline at that point
        l_direction_x,l_direction_y          = interpolate.splev(points[:,0],self.median_tck, der = 1)

        # 3. get perpendicular direction at that point
        #w_direction_x,w_direction_y          = -1/l_direction_x  , -1/l_direction_y
        w_direction_x,w_direction_y = l_direction_y,-l_direction_x
        
        # numpy to torch - this is not how you should write this but I was curious
        [closest_median_point_x,
        closest_median_point_y,
        l_direction_x,
        l_direction_y,
        w_direction_x,
        w_direction_y] = [torch.from_numpy(arr) for arr in [closest_median_point_x,
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
        
        new_pts = torch.zeros([d,4,3])
        
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
    
    def get_direction(self,points,name = None):
        """
        Find closest point on spline. use relative x_coordinate difference (in state plane coords) to determine whether point is below spline (EB) or above spline (WB)
        
        
        points    - [d,m,3] array of points in space
        name      - list of correspondence key names. THey are not needed but if supplied the correct directions will be appended to them
        RETURN:   - [d] list of names with "EB" or "WB" added, best guess of which side of road object is on and which correspondence should be used
                  - [d] tensor of int with -1 if "WB" and 1 if "EB" per object
        """
        
        mean_points = torch.mean(points, dim = 1)
        min_u  = self.closest_spline_point(mean_points)
        
        spl_x,_ = interpolate.splev(min_u, self.median_tck)
        
        direction = torch.sign(torch.from_numpy(spl_x) - mean_points[:,0]).int()
        d_list = ["EB","WB"]
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
        template_im_height = torch.sum(torch.sqrt(torch.pow((template_top - template_bottom),2)),dim = 1)
        template_ratio = template_im_height / template_space_heights
        
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
            if torch.min(bbox_3d[:,0]) < -100 or torch.max(bbox_3d[:,0]) > 3940 or torch.min(bbox_3d[:,1]) < -100 or torch.max(bbox_3d[:,1]) > 2260:
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
        self.plot_boxes(im,im_boxes,color = color,labels = labels, thickness = thickness)
    
    def plot_points(self,im,points, color = (0,255,0)):
        """
        Lazily, duplicate each point 8 times as a box with 0 l,w,h then call plot_boxes
        points -  [d,2] array of x,y points in roadway coordinates / state 
        """
        rep_points = torch.cat((points,torch.zeros([points.shape[0],3]),torch.ones([points.shape[0],1])),dim = 1)
        space_points = self.state_to_space(rep_points)
        space_points = space_points[:,0,:]
        self.plot_space_points(im,space_points,color = color)
    
    def plot_space_points(self,im,points,color = (255,0,0), name = None):
        """
        points -  [d,n,3] array of x,y points in roadway coordinates / state 
        """
        
        im_pts = self.space_to_im(points, name = name).squeeze(1)
        
        for point in im_pts:
            cv2.circle(im,(int(point[0]),int(point[1])),2,color,-1)
        
        cv2.imshow("frame",im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        
        
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
        
        
        ### Project each aerial imagery point into pixel space and get pixel error
        if True:
            print("Test 1: Pixel Reprojection Error")
            start = time.time()
            running_error = []
            for name in self.correspondence.keys():
                corr = self.correspondence[name]
                name = name.split("_")[0]
                
                space_pts = torch.from_numpy(corr["space_pts"]).unsqueeze(1)
                space_pts = torch.cat((space_pts,torch.zeros([space_pts.shape[0],1,1])), dim = -1)
                
                im_pts    = torch.from_numpy(corr["corr_pts"])
                namel = [name for _ in range(len(space_pts))]
                
                proj_space_pts = self.space_to_im(space_pts,name = namel).squeeze(1)
                error = torch.sqrt(((proj_space_pts - im_pts)**2).sum(dim = 1)).mean()
                #print("Mean error for {}: {}px".format(name,error))
                running_error.append(error)   
            end = time.time() - start
            print("Average Pixel Reprojection Error across all homographies: {}px in {} sec\n".format(sum(running_error)/len(running_error),end))
            
            
        
        ### Project each camera point into state plane coordinates and get ft error
        if True:
            print("Test 2: State Reprojection Error")
            running_error = []
            
            all_im_pts = []
            all_cam_names = []
            for name in self.correspondence.keys():
                corr = self.correspondence[name]
                name = name.split("_")[0]
    
                
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
            boxes[:,5] = torch.sign(boxes[:,1])  # TODO fix

            

            space_boxes = self.state_to_space(boxes)
            repro_state_boxes = self.space_to_state(space_boxes)
            
            error = torch.abs(repro_state_boxes - boxes)
            mean_error = error.mean(dim = 0)
            print("Mean State-Space-State Error: {}ft\n".format(mean_error))
        
        if True:
            print("Test 3: Random Box Reprojection Error (State-Im-State)")
            
            all_im_pts = torch.cat(all_im_pts,dim = 0)
            state_pts = self.im_to_state(all_im_pts, name = all_cam_names, heights = 0,refine_heights = False)
            
            ### Create a random set of boxes
            boxes = torch.rand(state_pts.shape[0],6) 
            boxes[:,:2] = state_pts[:,:2]
            boxes[:,1] = 0 #boxes[:,1] * 240 - 120
            boxes[:,2] *= 0
            boxes[:,3] *= 0
            boxes[:,4] *= 0
            boxes[:,5] = torch.sign(boxes[:,1])  # TODO fix
            
            # get appropriate camera for each object
            
            
            im_boxes = self.state_to_im(boxes,name = all_cam_names)
            directions = boxes[:,5]
            
            # plot some of the boxes
            repro_state_boxes = self.im_to_state(im_boxes,name = all_cam_names,heights = boxes[:,4],refine_heights = True)
            
            
            error = torch.abs(repro_state_boxes - boxes)
            error = torch.mean(error,dim = 0)
            print("Average State-Im_State Reprojection Error: {} ft\n".format(error))
        
        
        if True:
            print("Test 3: Random Box Reprojection Error (Im-State-Im)")

            repro_im_boxes = self.state_to_im(repro_state_boxes, name = all_cam_names)
            
            error = torch.abs(repro_im_boxes-im_boxes)
            error = torch.mean(error)
            print("Average Im-State-Im Reprojection Error: {} px\n".format(error))
            
            
        if True:
            active_cam = "P08C02"
            im = cv2.imread(os.path.join(im_dir,"{}.png".format(active_cam)))
            
            self.plot_state_boxes(im, boxes,color = (255,0,0))
            self.plot_boxes(im,im_boxes,color = (0,255,0))
            
            cv2.imshow("frame",im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
#%% MAIN        
    
if __name__ == "__main__":
    im_dir = "/home/derek/Documents/i24/i24_homography/data_real"
    #space_dir = "/home/derek/Documents/i24/i24_homography/aerial/to_P24"
    im_dir = "/home/derek/Data/MOTION_HOMOGRAPHY_FINAL"
    space_dir = "/home/derek/Documents/i24/i24_homography/aerial/all_poles_aerial_labels"
    
    save_file =  "P8-P40_curvhg.cpkl"

    hg = Curvilinear_Homography(save_file = save_file,space_dir = space_dir, im_dir = im_dir)
    hg.test_transformation(im_dir)
    
    hg._generate_extents_file(im_dir)