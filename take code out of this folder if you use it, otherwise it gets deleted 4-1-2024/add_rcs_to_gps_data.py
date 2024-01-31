import pyproj
import pandas as pd
import numpy as np
import torch
#from i24_rcs import I24_RCS
from double_curvilinear_homography import Curvilinear_Homography as I24_RCS
import warnings
warnings.filterwarnings("ignore")

if True:
    hg = I24_RCS(save_file = "/home/derek/Documents/i24/i24_track/data/homography/CIRCLES_20_Wednesday.cpkl", downsample = 2)
    gps_data_file = "/home/derek/Data/CIRCLES_GPS/gps_message_raw2a.csv"
    
    
    def WGS84_to_TN(points):
        """
        Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
        Transform is expected to be accurate within ~2 feet
        
        points array or tensor of size [n_pts,2]
        returns out - array or tensor of size [n_pts,2]
        """
        
        wgs84=pyproj.CRS("EPSG:4326")
        tnstate=pyproj.CRS("epsg:2274")
        out = pyproj.transform(wgs84,tnstate, points[:,0],points[:,1])
        out = np.array(out).transpose(1,0)
        
        if type(points) == torch.Tensor:
            out = torch.from_numpy(out)
            
        return out
    
    # test_point = torch.tensor([36.0137613,-86.6198052]).unsqueeze(0).expand(100,2)
    # output = WGS84_to_TN(test_point)
    
    
    #dataframe = pd.read_csv(gps_data_file,delimiter = "\t")
    dataframe = pd.read_csv(gps_data_file,delimiter = ",")
    del dataframe["Gpstime"]
    del dataframe["HDOP"]
    del dataframe["PDOP"]
    del dataframe["VDOP"]
    del dataframe["file_tag_id"]
    
    # state->road fails otherwise
    longlim = [-86.70,-86.53]
    latlim = [35.93,36.09]
    
    dataframe = dataframe[dataframe["Lat"] > latlim[0]]
    dataframe = dataframe[dataframe["Lat"] < latlim[1]]
    dataframe = dataframe[dataframe["Long"] > longlim[0]]
    dataframe = dataframe[dataframe["Long"] < longlim[1]]
    
    lat  = dataframe["Lat"].tolist()
    long = dataframe["Long"].tolist()
    
    lat = np.array(lat)
    long = np.array(long)
    data = np.stack([lat,long]).transpose(1,0)
    
    state_plane = WGS84_to_TN(data)
    state_plane = torch.from_numpy(state_plane)
    state_plane = torch.cat((state_plane,torch.zeros([state_plane.shape[0],1])),dim = 1).unsqueeze(1)
    rcs = hg.space_to_state(state_plane)
    
    dataframe["state_x"] = state_plane[:,0,0]
    dataframe["state_y"] = state_plane[:,0,1]
    dataframe["rcs_x"]   = rcs[:,0].data.numpy()
    dataframe["rcs_y"]   = rcs[:,1].data.numpy()
    
    
    dataframe.to_csv(gps_data_file)
    

    
    import matplotlib.pyplot as plt
    plt.figure()

    plt.scatter(dataframe["state_x"][:1000000],dataframe["state_y"][:1000000])
    for corr in hg.correspondence.values():
        stp = torch.tensor(corr["space_pts"])
        plt.scatter(stp[:,0],stp[:,1],color = "r")
    plt.show()
        
    plt.figure()
    plt.plot(dataframe["Lat"],dataframe["Long"])
    #plt.plot(dataframe["rcs_x"],dataframe["rcs_y"])
    
    plt.figure()
    plt.scatter(dataframe["rcs_x"][:1000000],dataframe["rcs_y"][:1000000])
    plt.xlim([-5000,30000])
    plt.ylim([-2000,2000])

    #plt.scatter(rcs[:,0][:1000000],rcs[:,1][:1000000],color = "g")


    for corr in hg.correspondence.values():
        stp = torch.tensor(corr["space_pts"])
        stp = torch.cat((stp,torch.zeros([stp.shape[0],1])),dim = 1).unsqueeze(1)
        stp_rcs = hg.space_to_state(stp)
        plt.scatter(stp_rcs[:,0],stp_rcs[:,1],color = "r")
        
    plt.show()
    
    

if False:
    gps_data_file = "/home/derek/Documents/rcs_gps_data.csv"
    
    dataframe = pd.read_csv(gps_data_file,delimiter = ",")
    
    dataframe = dataframe[dataframe["Lat"] != "-"]
    dataframe = dataframe[dataframe["Long"] != "-"]
    import matplotlib.pyplot as plt
    #plt.plot(dataframe["state_x"],dataframe["state_y"])
    #plt.plot(dataframe["Lat"],dataframe["Long"])
    plt.scatter(dataframe["road_x"][:1000000],dataframe["road_y"][:1000000])
    plt.xlim([-5000,30000])
    plt.ylim([-2000,2000])



if False:
    hg = I24_RCS(save_file = "/home/derek/Documents/i24/i24_track/data/homography/CIRCLES_20_Wednesday.cpkl", downsample = 2)
    gps_data_file = "/home/derek/Data/CIRCLES_GPS/mvt_11_14_to_11_18_gps_vins.csv"
    
    
    def WGS84_to_TN(points):
        """
        Converts GPS coordiantes (WGS64 reference) to tennessee state plane coordinates (EPSG 2274).
        Transform is expected to be accurate within ~2 feet
        
        points array or tensor of size [n_pts,2]
        returns out - array or tensor of size [n_pts,2]
        """
        
        wgs84=pyproj.CRS("EPSG:4326")
        tnstate=pyproj.CRS("epsg:2274")
        out = pyproj.transform(wgs84,tnstate, points[:,0],points[:,1])
        out = np.array(out).transpose(1,0)
        
        if type(points) == torch.Tensor:
            out = torch.from_numpy(out)
            
        return out
    
    # test_point = torch.tensor([36.0137613,-86.6198052]).unsqueeze(0).expand(100,2)
    # output = WGS84_to_TN(test_point)
    
    
    #dataframe = pd.read_csv(gps_data_file,delimiter = "\t")
    dataframe = pd.read_csv(gps_data_file,delimiter = ",")

    
    lat  = dataframe["latitude"].tolist()
    long = dataframe["longitude"].tolist()
    
    lat = np.array(lat)
    long = np.array(long)
    data = np.stack([lat,long]).transpose(1,0)
    
    state_plane = WGS84_to_TN(data)
    state_plane = torch.from_numpy(state_plane)
    state_plane = torch.cat((state_plane,torch.zeros([state_plane.shape[0],1])),dim = 1).unsqueeze(1)
    rcs = hg.space_to_state(state_plane)
    
    dataframe["state_x"] = state_plane[:,0,0]
    dataframe["state_y"] = state_plane[:,0,1]
    dataframe["rcs_x"]   = rcs[:,0].data.numpy()
    dataframe["rcs_y"]   = rcs[:,1].data.numpy()
    #dataframe.to_csv("/home/derek/Data/CIRCLES_GPS/rcs_gps_message_raw2.csv")
    
    
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(dataframe["state_x"],dataframe["state_y"])
    plt.figure()
    plt.plot(dataframe["latitude"],dataframe["longitude"])
    plt.figure()
    plt.plot(dataframe["rcs_x"],dataframe["rcs_y"])
