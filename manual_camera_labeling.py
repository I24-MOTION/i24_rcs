import cv2
import time
import os
import numpy as np
import csv
import string
import re
import copy
import _pickle as pickle
import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torchvision.transforms.functional as F
from scipy.spatial import ConvexHull
import itertools
import sys


from transform_compute import compute_correspondences,get_all_dash_points



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

def find_vanishing_point(lines):
    """
    Finds best (L2 norm) vanishing point given a list of lines

    Parameters
    ----------
    lines : [(x0,y0,x1,y1), ...]

    Returns
    -------
    vp - (x,y)
    """
    
    # mx+b form
    #y0 = ax + c
    #y1 = bx + d
    
    line0 = lines[0]
    line1 = lines[1]
    a = (line0[3] - line0[1])/line0[2] - line0[0]
    b = (line1[3] - line1[1])/line1[2] - line1[0]
    c = line0[1] - a*line0[0]
    d = line1[1] - c*line1[0]
    
    # intersection
    px = (d-c)/(a-b)
    py = a*(d-c)/(a-b) + c
    best_dist = np.inf
    
    # using intersection as starting point, grid out a grid of 11 x 11 points with spacing g
    g = 1e+16
    n_pts = 31
    
    while g > 1:
        #print("Gridding at g = {}".format(g))

        # create grid centered around px,py with spacing g
        
        x_pts = np.arange(px-g*(n_pts//2),px+g*(n_pts//2),g)
        y_pts = np.arange(py-g*(n_pts//2),py+g*(n_pts//2),g)
        
        for x in x_pts:
            for y in y_pts:
                # for each point in grid, compute average distance to vanishing point
                dist = 0
                for line in lines:
                    dist += line_to_point(line,(x,y))**2
                   
                # keep best point in grid
                if dist < best_dist:
                    px = x 
                    py = y
                    best_dist = dist
                    #print("Best vp so far: ({},{}), with average distance {}".format(px,py,np.sqrt(dist/len(lines))))
    
                # regrid
        g = g / 10.0
            
    return [px,py]



def average_frame(sequence):
    
    frames = []
    resize = None  # (1920,1080)

    gpuID = 0

    nvDec = nvc.PyNvDecoder(sequence, gpuID)
    target_h, target_w = nvDec.Height(), nvDec.Width()

    to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(
    ), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(
    ), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)

    cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
    if nvc.ColorSpace.UNSPEC == cspace:
        cspace = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == crange:
        crange = nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)

    count = 0
    avg_frame = None

    # get frames from one file
    while True:

        if count % 1000 == 0:
            print("On frame {} for sequence {}".format(count, sequence))

        pkt = nvc.PacketData()

        # Obtain NV12 decoded surface from decoder;
        raw_surface = nvDec.DecodeSingleSurface(pkt)
        if raw_surface.Empty():
            break

        # Convert to RGB interleaved;
        rgb_byte = to_rgb.Execute(raw_surface, cc_ctx)

        # Convert to RGB planar because that's what to_tensor + normalize are doing;
        rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)

        # likewise, end of video file
        if rgb_planar.Empty():
            break

        # Create torch tensor from it and reshape because
        # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
        # and then copies data from plane pointer to allocated chunk;
        surfPlane = rgb_planar.PlanePtr()
        surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(
        ), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
        surface_tensor.resize_(3, target_h, target_w)

        if resize is not None:
            try:
                surface_tensor = torch.nn.functional.interpolate(
                    surface_tensor.unsqueeze(0), resize).squeeze(0)
            except:
                raise Exception(
                    "Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape, resize))

        # This is optional and depends on what you NN expects to take as input
        # Normalize to range desired by NN. Originally it's
        surface_tensor = surface_tensor.type(
            dtype=torch.cuda.FloatTensor)

        # apply normalization
        #surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if avg_frame is None:
            avg_frame = surface_tensor
        else:
            avg_frame += surface_tensor
        count += 1

        # if count > 100:
        #     break
    
        if count % 100 == 0:
            frames.append(surface_tensor.permute(1, 2, 0).data.cpu().numpy()[:,:,::-1])

    avg_frame /= count

    avg_frame = avg_frame.permute(1, 2, 0).data.cpu().numpy()
    return avg_frame, frames


def poly_area(polygon):
    """
    Returns the area of the polygon
    polygon - [n_vertices,2] tensor of clockwise points
    """
    x1 = polygon[:,0]
    y1 = polygon[:,1]
    
    x2 = x1.roll(1)
    y2 = y1.roll(1)
    
    # per this formula: http://www.mathwords.com/a/area_convex_polygon.htm
    area = -1/2.0 * (torch.sum(x1*y2) - torch.sum(x2*y1))
    
    return area

def get_hull(points, indices=False):
    hull = ConvexHull(points.clone().detach()).vertices.astype(int)

    if indices:
        return hull

    points = points[hull, :]
    return points


def clockify(polygon, clockwise=True, hull=False, center = None):
    """
    polygon - [n_vertices,2] tensor of x,y,coordinates for each convex polygon
    clockwise - if True, clockwise, otherwise counterclockwise
    returns - [n_vertices,2] tensor of sorted coordinates 
    """
    relist = False
    if type(polygon) == list:
        relist = True
        polygon = torch.stack([torch.tensor(pt) for pt in polygon]).float()
    
    # get center
    if center is None:
        center = torch.mean(polygon, dim=0)

    # get angle to each point from center
    diff = polygon - center.unsqueeze(0).expand([polygon.shape[0], 2])
    tan = torch.atan(diff[:, 1]/diff[:, 0])
    direction = (torch.sign(diff[:, 0]) - 1)/2.0 * -np.pi

    angle = tan + direction
    sorted_idxs = torch.argsort(angle)

    if not clockwise:
        sorted_idxs.reverse()

    polygon = polygon[sorted_idxs.detach(), :]

    if hull:
        polygon = get_hull(polygon)

    if relist:
        polygon = polygon.int()
        polygon = [(row[0].item(), row[1].item()) for row in polygon]

    return polygon

# define annotator


class CameraAnnotator:
    def __init__(self, frame, frame_list = None, cam_name="UNKNOWN", save_directory=None, load = False):

        self.frame_list = frame_list
        self.frame_list_idx = 0
        
        self.im = (frame).astype(np.uint8)
        self.cam_name = cam_name

        if save_directory is not None:
            self.save_file = os.path.join(save_directory, self.cam_name) + ".cpkl"
        else:
            self.save_file = None

        self.pos = (0, 0)
        self.new = False
        self.moved = False

        self.SHOW_LABELS = 1
        self.SHOW_VP = True
        self.DIRECTION = "EB"
        self.ACTIVE = "VP"
        self.clicked_point = None
        self.save_dir = save_dir
        self.temp_pts = None
        
        self.text_color = (0,0,0)
        
        self.active_text = ""
        
        self.active_feature = "lane1"
        self.active_curve_feature = "yellow"
        self.active_index   = 0
        self.active_curve_index = 0
        self.active_letter  = 0
        self.letters = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","aa","bb","cc","dd","ee","ff","gg","hh"]
        
        self.guess_text = "{}_{}_{}_{}".format(self.DIRECTION.lower(),self.active_feature,self.active_index,self.letters[self.active_letter])
        
        
        self.active_curve = "{}_{}_{}".format(self.DIRECTION.lower(),self.active_curve_feature,self.active_curve_index)
        

        
        # each list element will be (row,column,name)
        self.data = {
            "WB": {
                "curves": [],
                "points": [],
                "FOV": [],
                "mask": [],
                "vp":None,
                "z_vp":None
            },
            "EB": {
                "curves": [],
                "points": [],
                "FOV": [],
                "mask": [],
                "vp":None,
                "z_vp":None
            }
        }

        self.undo_cache = [copy.deepcopy(self.data.copy)]

        if load:
            self.load()
        self.plot()

        self.vp_cache = []
        self.z_cache = []
        
        self.corr = {}

    def save(self):
        try:
            with open(self.save_file, "wb") as f:
                pickle.dump(self.data, f)
            print("Saved annotations at {}".format(self.save_file))
        except:
            print("Invalid save file, unable to save")

    def load(self):
        try:
            with open(self.save_file, "rb") as f:
                self.data = pickle.load(f)
                self.undo_cache.append(copy.deepcopy(self.data))
        except:
            print("Invalid save file, unable to save")

    def get_correspondence(self):
        self.save()
        
        corr = compute_correspondences([self.cam_name],direction = self.DIRECTION,ADD_PROJ = False)
        self.corr[self.DIRECTION] = corr["{}_{}".format(self.cam_name,self.DIRECTION)]
        
        

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            self.new = True

        if event == cv2.EVENT_LBUTTONUP and self.ACTIVE == "Z AXIS":
            #height = float(self.keyboard_input(update_plot = True)) 
            height = 14
            ln = [self.clicked_point[0], self.clicked_point[1],x,y,height]
            self.z_cache.append(ln)
            self.undo_cache.append(copy.deepcopy(self.data))
            self.clicked_point = None
            self.new = True
            cv2.waitKey(1)
            

        if event == cv2.EVENT_RBUTTONDOWN:
            self.new = True

        elif event == cv2.EVENT_MOUSEMOVE:
            self.pos = (x, y)
            self.moved = True
            
            
            


    def plot_space_pts(self,im):
        """
        Projects 3D space points into image/correspondence using P:
            new_pts = P x points T  ---> [dm,3] T = [3,4] x [4,dm]
        performed by flattening batch dimension d and object point dimension m together
        
        points - [d,m,3] array of points in 3-space
        """
        try:
            corr = self.corr[self.DIRECTION]["P"]
        except:
            return []
        
        
        points = torch.stack([torch.tensor(self.corr[self.DIRECTION]["state_plane_pts"][0]),
                              torch.tensor(self.corr[self.DIRECTION]["state_plane_pts"][1]),
                              torch.zeros(len(self.corr[self.DIRECTION]["state_plane_pts"][1]))]).transpose(1,0)
        
        d = points.shape[0]
        
        # convert points into size [dm,4]
        points = points.view(-1,3)
        points = torch.cat((points.double(),torch.ones([points.shape[0],1],device = points.device).double()),1) # add 4th row
        

        points = torch.transpose(points,0,1).double()
        P = torch.from_numpy(corr).double().to(points.device)
        new_pts = torch.matmul(P,points).transpose(0,1)
        
        # divide each point 0th and 1st column by the 2nd column
        new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
        new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
        
        # drop scale factor column
        new_pts = new_pts[:,:2] 
        
        # reshape to [d,m,2]
        new_pts = new_pts.view(d,-1,2).squeeze()
        
        for pt in new_pts:
            pt = int(pt[0].item()), int(pt[1].item())
            im = cv2.circle(im,pt,3,(0,255,0),-1)
            
        return im
        

    def plot(self):
        
        self.cur_image = self.im.copy()
        
        if self.ACTIVE == "Z AXIS" and self.frame_list is not None:
            self.cur_image = (self.frame_list[self.frame_list_idx].copy()).astype(np.uint8)

        text_block = [
            "Camera: {}".format(self.cam_name),
            "Active Direction: {}".format(self.DIRECTION),
            "Active Command: {}".format(self.ACTIVE),
            "",
            "COMMANDS:",
            "q: save and quit",
            "1: toggle mode",
            "=: toggle direction",
            "t: toggle text labels",
            "u: undo",
            "!: load saved file",
            "f: FOV command",
            "p: POINT command",
            "m: MASK command",
            "c: CURVE command",
            "s: save",
            "a: AUTO POINTS command"
        ]

        try:
            self.corr[self.DIRECTION]
            self.cur_image = self.plot_space_pts(self.cur_image)
            
            
            
            if self.SHOW_VP:
                # project self.pos up by 10 feet and backproject, then plot line
                pt = torch.tensor([self.pos[0],self.pos[1],1]).unsqueeze(0).transpose(1,0)
                
                P =  torch.from_numpy(self.corr[self.DIRECTION]["P"])
                H = torch.from_numpy(self.corr[self.DIRECTION]["H"])
                new_pts = torch.matmul(H,pt.double()).transpose(0,1)
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                new_pts = new_pts[:,:2]
                new_pts = torch.cat((new_pts.transpose(0,1),14+torch.zeros([1,1]), torch.ones([1,1])))
                
                new_pts = torch.matmul(P,new_pts).transpose(0,1)
                new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
                new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
                
                top_pt = int(new_pts[0,0].item()), int(new_pts[0,1].item())
                
                cv2.line(self.cur_image,top_pt,self.pos,(0,0,255),2)
            
            
            
            
        except:
            
            # show vp
            vp = self.data[self.DIRECTION]["vp"]
            if vp is not None and self.SHOW_VP:
                cv2.line(self.cur_image,(int(vp[0]),int(vp[1])),self.pos,(0,0,0),1)
                
            # show z vp
            z_vp = self.data[self.DIRECTION]["z_vp"]
            if z_vp is not None and self.SHOW_VP:
                cv2.line(self.cur_image,(int(z_vp[0]),int(z_vp[1])),self.pos,(0,0,255),1)
                
                
                
        
        # if FOV is active, display FOV
        if self.ACTIVE == "FOV" or self.ACTIVE == "MASK":
            FOV_poly = self.data[self.DIRECTION]["FOV"]
            if len(FOV_poly) > 2:
                FOV_poly = np.stack([np.array(pt) for pt in FOV_poly]).reshape(
                    1, -1, 2).astype(np.int32)
                transparency = (np.ones(self.im.shape)*0.5)
                transparency = cv2.fillPoly(
                    transparency, FOV_poly,  (1, 0.8, 0.8), lineType=cv2.LINE_AA)
                self.cur_image = (transparency.astype(
                    float) * self.cur_image.astype(float)).astype(np.uint8)

        # # if mask is active, display mask
        # elif self.ACTIVE == "MASK":
            mask_poly = self.data[self.DIRECTION]["mask"]
            if len(mask_poly) > 2:
                mask_poly = np.stack([np.array(pt) for pt in mask_poly]).reshape(
                    1, -1, 2).astype(np.int32)
                transparency = (np.ones(self.im.shape))
                transparency[:, :, 0:2] = 0.8
                transparency = cv2.fillPoly(
                    transparency, mask_poly,  (1, 1, 1), lineType=cv2.LINE_AA)
                self.cur_image = (transparency.astype(
                    float) * self.cur_image.astype(float)).astype(np.uint8)

        if self.ACTIVE == "Z AXIS":
            for line in self.z_cache:
                self.cur_image = cv2.line(self.cur_image,(int(line[0]),int(line[1])),(int(line[2]),int(line[3])),(0,0,255),1)

        # display preliminary info in corner unless mouse_position is in that region
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        if self.pos[0] > 400 or self.pos[1] > (3+len(text_block))*30:
            for ridx, row in enumerate(text_block):
                self.cur_image = cv2.putText(
                    self.cur_image, row, (10, (ridx+2)*30), font, scale, (255, 255, 255), 1)

        # if show_labels, show labels

        # regardless, show current label if a label is currently being typed
        if self.clicked_point is not None and self.ACTIVE == "POINTS":
            
            if len(self.active_text) > 0:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_text, self.clicked_point, font, self.SHOW_LABELS, self.text_color, 1)
            else:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.guess_text, self.clicked_point, font, self.SHOW_LABELS, (100,100,100), 1)
                
            self.cur_image = cv2.circle(
                self.cur_image, self.clicked_point, 1, (100, 0, 100), -1)
            self.cur_image = cv2.circle(
                self.cur_image, self.clicked_point, 4, (100, 0, 100), 1)
            
        elif self.clicked_point is not None and self.ACTIVE == "DELETE":
            
            if len(self.active_text) > 0:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_text, self.clicked_point, font, self.SHOW_LABELS, self.text_color, 1)
            else:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.guess_text, self.clicked_point, font, self.SHOW_LABELS, (100,100,100), 1)
                
            
        elif self.temp_pts is not None and self.ACTIVE == "AUTO POINTS":
            
            if len(self.active_text) > 0:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_text, self.clicked_point, font, 1, self.text_color, 1)
            else:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.guess_text, self.clicked_point, font, 1, (100,100,100), 1)
                
            for pidx, point in enumerate(self.temp_pts):
                self.cur_image = cv2.circle(
                    self.cur_image, point, 1, (100, 0, 100), -1)
                self.cur_image = cv2.circle(
                    self.cur_image, point, 4, (100, 0, 100), 1)
                
                self.cur_image = cv2.putText(self.cur_image,self.letters[pidx],point,font,0.5,(100,0,100),1)

        # regardless, show current label if a label is currently being typed
        elif self.clicked_point is not None and self.ACTIVE == "CURVE":
            
            if len(self.active_text) > 0:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_text, self.clicked_point, font, 1, self.text_color, 1)
            elif self.active_curve is not None:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_curve, self.clicked_point, font, 1, (100,100,100), 1)
            
            else:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.guess_text, self.clicked_point, font, 1, (100,100,100), 1)
                
            self.cur_image = cv2.circle(
                self.cur_image, self.clicked_point, 1, (100, 0, 100), -1)
            self.cur_image = cv2.circle(
                self.cur_image, self.clicked_point, 4, (100, 0, 100), 1)
        
        elif self.clicked_point is not None and self.ACTIVE == "Z AXIS":
            
            if len(self.active_text) > 0:
                self.cur_image = cv2.putText(
                    self.cur_image, " " +  self.active_text, self.clicked_point, font, 1, self.text_color, 1)   

        # plot points
        for point in self.data[self.DIRECTION]["points"]:
            pt = (int(point[0]), int(point[1]))
            if self.SHOW_LABELS:
                self.cur_image = cv2.putText(
                    self.cur_image, " " + point[2], pt, font, self.SHOW_LABELS, self.text_color, 1)
            self.cur_image = cv2.circle(self.cur_image, pt, 1, (255, 0, 0), -1)
            self.cur_image = cv2.circle(self.cur_image, pt, 4, (255, 0, 0), -1)
            
        
            
        # plot curve points
        for point in self.data[self.DIRECTION]["curves"]:
            pt = (int(point[0]), int(point[1]))
            if self.SHOW_LABELS:
                self.cur_image = cv2.putText(
                    self.cur_image, " " + point[2], pt, font, self.SHOW_LABELS, self.text_color, 1)
            self.cur_image = cv2.circle(self.cur_image, pt, 1, (0, 255, 255), -1)
            self.cur_image = cv2.circle(self.cur_image, pt, 4, (0, 255, 255), 1)


        # END





    def impute_points(self):
        
        # ensure that homgraphy exists
        try:
            corr = self.corr[self.DIRECTION]["P"]
        except: return
        
        #get all aerial points
        ae_x,ae_y, ae_id = get_all_dash_points(self.DIRECTION)
        
        
        # get FOV pixel extents
        x_min = 3840
        x_max = 0
        y_min = 2160
        y_max = 0
        
        for point in self.data[self.DIRECTION]["FOV"]:
            if point[0] < x_min:
                x_min = point[0]
            if point[0] > x_max:
                x_max = point[0]
            if point[1] < y_min:
                y_min = point[1]
            if point[1] > y_max:
                y_max = point[1]
                
        
        
        
        # transform to image
        points = torch.stack([torch.tensor(ae_x),
                              torch.tensor(ae_y),
                              torch.zeros(len(ae_x))]).transpose(1,0)
        
        d = points.shape[0]
        
        # convert points into size [dm,4]
        points = points.view(-1,3)
        points = torch.cat((points.double(),torch.ones([points.shape[0],1],device = points.device).double()),1) # add 4th row
        

        points = torch.transpose(points,0,1).double()
        P = torch.from_numpy(corr).double().to(points.device)
        new_pts = torch.matmul(P,points).transpose(0,1)
        
        # divide each point 0th and 1st column by the 2nd column
        new_pts[:,0] = new_pts[:,0] / new_pts[:,2]
        new_pts[:,1] = new_pts[:,1] / new_pts[:,2]
        
        # drop scale factor column
        new_pts = new_pts[:,:2] 
        
        # reshape to [d,m,2]
        new_pts = new_pts.view(d,-1,2).squeeze()
        
        
        # remove all dashes with points outside of frame
        dash_dict = {}
        for i in range(len(ae_id)):
            if new_pts[i,0] > x_min and new_pts[i,1] > y_min and new_pts[i,0] < x_max and new_pts[i,1] < y_max:
                trunc_id = ae_id[i][:-2]
                try: dash_dict[trunc_id].append(new_pts[i])
                except: dash_dict[trunc_id] = [new_pts[i]]
        
        # for remaining points, get all sets of 4 dash points, 
        for key in dash_dict:
            if len(dash_dict[key]) == 4:
                # get average point for each set
                avg = sum(dash_dict[key]) / len(dash_dict[key])
        
                DUP = False
                for existing_point in self.data[self.DIRECTION]["points"]:
                    if key in existing_point[2]:
                        DUP = True
                        break
                if DUP: continue
        
                try:
                    # for each average point, run auto_points
                    self.clicked_point = avg.data.numpy().astype(int)
                    points = self.auto_points()
                    
                    if points is None:
                        p_idx = 0
                        while points is None and p_idx < 4:
                            
                            self.clicked_point = dash_dict[key][p_idx].data.numpy().astype(int)
                            points = self.auto_points()
                            p_idx += 1
                except:
                    continue
                    
                if points is not None:
                
                    self.active_letter = 0
            
                    # if successful, seed that dashed line with corresponding name
                    for pt in points:
                        new_name = "{}_{}_{}".format(self.DIRECTION.lower(),key,self.letters[self.active_letter])
                        self.data[self.DIRECTION]["points"].append([pt[0], pt[1], new_name])
                        self.active_letter += 1
    
    
                    # append to undo buffer
                    self.undo_cache.append(copy.deepcopy(self.data))
                    if len(self.undo_cache) > 100:
                        del self.undo_cache[0]
                        
                    # show
                    self.plot()
                    cv2.imshow("window", self.cur_image)
                    cv2.setWindowTitle("window", self.cam_name)
                    cv2.waitKey(1)
                        
        # increment default point
        self.active_letter  = 0
        self.clicked_point = None
        
        



    def keyboard_input(self, update_plot=False):
        keys = self.guess_text if self.ACTIVE != "CURVE" else self.active_curve
        if self.ACTIVE == "Z AXIS": keys = "" 
        letters = string.ascii_lowercase + string.digits + string.punctuation
        while not self.new:
            if update_plot:
                self.plot()
                cv2.imshow("window", self.cur_image)

            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\b"):
                keys = keys[:-1]
            if key == ord("\n") or key == ord("\r"):
                break
            self.active_text = keys
            
        #self.active_text = ""
        #if len(keys) == 0 and self.ACTIVE == "CURVE": keys = self.active_curve
        #if len(keys) == 0: keys = self.guess_text
        
        return keys

    def click_handler(self):
        
        if self.ACTIVE == "FOV":
            self.data[self.DIRECTION]["FOV"].append(self.clicked_point)
            if len(self.data[self.DIRECTION]["FOV"]) > 2:
                self.data[self.DIRECTION]["FOV"] = clockify(
                    self.data[self.DIRECTION]["FOV"], hull=True)

        elif self.ACTIVE == "MASK":
            self.data[self.DIRECTION]["mask"].append(self.clicked_point)
            if len(self.data[self.DIRECTION]["mask"]) > 2:
                self.data[self.DIRECTION]["mask"] = clockify(
                    self.data[self.DIRECTION]["mask"])

        elif self.ACTIVE == "POINTS":
            pt = self.clicked_point
            name = self.keyboard_input(update_plot=True)
            self.data[self.DIRECTION]["points"].append([pt[0], pt[1], name])
            
            # increment default point
            name_parts = name.split("_")
            self.active_feature = name_parts[1]
            self.active_index   = int(name_parts[2])
            self.active_letter  = self.letters.index(name_parts[3]) + 1
            self.guess_text = "{}_{}_{}_{}".format(self.DIRECTION.lower(),self.active_feature,self.active_index,self.letters[self.active_letter])

        elif self.ACTIVE == "AUTO POINTS":
            self.temp_pts = self.auto_points()
            if self.temp_pts is not None:
                name = self.keyboard_input(update_plot = True)
                
                # get name so we can assign unique letters
                name_parts = name.split("_")
                self.active_feature = name_parts[1]
                self.active_index   = int(name_parts[2])
                self.active_letter  = self.letters.index(name_parts[3]) 
                
                for pt in self.temp_pts:
                    new_name = "{}_{}_{}_{}".format(self.DIRECTION.lower(),self.active_feature,self.active_index,self.letters[self.active_letter])
                    self.data[self.DIRECTION]["points"].append([pt[0], pt[1], new_name])
                    self.active_letter += 1

                
                self.temp_pts = None
                
                # increment default point
                name_parts = name.split("_")
                self.active_feature = name_parts[1]
                self.active_index   = int(name_parts[2]) + 1
                self.active_letter  = 0
                self.guess_text = "{}_{}_{}_{}".format(self.DIRECTION.lower(),self.active_feature,self.active_index,self.letters[self.active_letter])
 
        elif self.ACTIVE == "CURVE":
            pt = self.clicked_point
            name = self.keyboard_input(update_plot=True)
            
            name_parts = name.split("_")
            self.active_curve_feature = name_parts[1]
            self.active_curve_index   = int(name_parts[2])
            name = "{}_{}_{}".format(self.DIRECTION.lower(),self.active_curve_feature,self.active_curve_index)
            self.data[self.DIRECTION]["curves"].append([pt[0], pt[1], name])
            
            # increment default curve
            self.active_curve_index += 1
            self.active_curve = "{}_{}_{}".format(self.DIRECTION.lower(),self.active_curve_feature,self.active_curve_index)
            
        
        elif self.ACTIVE == "VP":
            self.vp_cache.append(self.clicked_point)
            
            if len(self.vp_cache) == 4:
                self.get_vp()
                self.vp_cache = []
                self.ACTIVE = "POINTS"
                
        elif self.ACTIVE == "DELETE":
            name = self.keyboard_input(update_plot = True)
            print("Deleting points {}".format(name))
            dir_data = self.data[self.DIRECTION]
            for group in dir_data:
                group_data = dir_data[group]
                idxs = []
                for pidx,point in enumerate(group_data):
                    if name in point[2]:
                        idxs.append(pidx)
                
                idxs.sort()
                idxs.reverse()
                for idx in idxs:
                    del group_data[idx]
                            
        
        if self.ACTIVE != "Z AXIS":
            self.clicked_point = None
            
        self.undo_cache.append(copy.deepcopy(self.data))
        if len(self.undo_cache) > 100:
            del self.undo_cache[0]

    def undo(self):
        if len(self.undo_cache) > 1:
            self.data = self.undo_cache[-2]
            del self.undo_cache[-1]
            self.new = True
    
    def get_z_vp(self):
        vp = find_vanishing_point(self.z_cache)
        self.data["EB"]["z_vp"] = vp
        self.data["WB"]["z_vp"] = vp
        
        self.data["EB"]["z_vp_lines"] = self.z_cache.copy()
        self.data["WB"]["z_vp_lines"] = self.z_cache.copy()
        
        self.z_cache = None
        self.ACTIVE = "POINTS"
        self.save()
        self.SHOW_VP = True
        
    def get_vp(self):
        
        """
        Finds best (L2 norm) vanishing point given a list of lines

        Parameters
        ----------
        lines : [(x0,y0,x1,y1), ...]

        Returns
        -------
        vp - (x,y)
        """
        
        # mx+b form
        #y0 = ax + c
        #y1 = bx + d
        
        line0 = [self.vp_cache[0][0],self.vp_cache[0][1],self.vp_cache[1][0],self.vp_cache[1][1]]
        line1 = [self.vp_cache[2][0],self.vp_cache[2][1],self.vp_cache[3][0],self.vp_cache[3][1]]
        # a = (line0[3] - line0[1])/(line0[2] - line0[0]) if line0[0] < line0[2] else (line0[1] - line0[3])/(line0[0] - line0[2])
        # b = (line1[3] - line1[1])/(line1[2] - line1[0]) if line1[0] < line1[2] else (line1[1] - line1[3])/(line1[0] - line1[2])
        # c = line0[1] - a*line0[0]
        # d = line1[1] - b*line1[0]
        
        # # intersection
        # px = (d-b)/(a-c)
        # py = a*px + c
        [x1,y1,x2,y2] = line0
        [x3,y3,x4,y4] = line1
        
            
        D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
        px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-x4*y3))/D
        py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-x4*y3))/D
        
        key = input("Got VP: {}. Is this vanishing point into town or out of town? (I/o)".format([px,py]))
        if key.lower() == "o":
            OUTBOUND_VP = True
        else:
            OUTBOUND_VP = False
            
        self.data["EB"]["vp"] = (px,py,OUTBOUND_VP)
        self.data["WB"]["vp"] = (px,py,OUTBOUND_VP)
        self.save()


    def auto_points(self, show = False):
        gray = self.im
        final = self.im.copy()

        gray = np.float32(gray)
        gray = cv2.cvtColor(self.im,cv2.COLOR_BGR2GRAY)
        gray2 = gray.copy()
        
        ret,gray = cv2.threshold(gray2,160,255,cv2.THRESH_BINARY)
        gray = np.float32(gray)
        
        if show:
            cv2.imshow("frame",gray/255.0)
            cv2.waitKey(0)
        
        # get the cluster of all pixels that are connected to the clicked point
        x = self.clicked_point[0]
        y = self.clicked_point[1]
        clicked_val = gray[y,x]
        
        points_queue = [(x,y)]
        visited_list = []
        component = []
        
        while len(points_queue) > 0:
            pt = points_queue[0]
            x = pt[0]
            y = pt[1]
            points_queue = points_queue[1:]
        
            visited_list.append(pt)

            if gray[y,x] == clicked_val:
                component.append(pt)
            
                gray[y,x] = 100
                final[y,x] = [100,100,100]
                
                for i in range(x-1,x+2):
                    for j in range(y-1,y+2):
                        if (i,j) not in visited_list and (i,j) not in points_queue:
                            points_queue.append((i,j))
            
            if len(component) > 5000:
                print("Component exceed max size, try again")
                return None
                        
        if show:
            cv2.imshow("frame",gray/255.0)
            cv2.waitKey(0)
            
        # get the convex hull
        stack = np.stack([np.array([pt[0],pt[1]]) for pt in component]).astype(np.float64)
        hull_indices = ConvexHull(stack).vertices
        hull = stack[hull_indices,:]
        
        if show:
            for point in hull:
                cv2.circle(final,(int(point[0]),int(point[1])),3,(0,0,255),-1)
            cv2.imshow("frame",final)
            cv2.waitKey(0)
        
        # for all sets of 4unique points in the convex hull, 
        indices = np.arange(0,len(hull),1)
        
        combos = list(itertools.combinations(indices, 4))
        
        
        # calc squared distances
        hullt = torch.from_numpy(hull)
        
        
        hullt = hullt.unsqueeze(1).expand(hullt.shape[0],hullt.shape[0],2)
        dist = torch.pow(hullt - hullt.transpose(0,1),  2).sum(dim = 2)
        
        max_combo = None
        max_dist = 0
        for combo in combos:
            
            area = poly_area( clockify(torch.from_numpy(hull[combo,:]).double())   )
            sum_dist = torch.sqrt(dist[combo,:][:,combo].sum())
            
            
            if area*sum_dist > max_dist:
                max_dist = area*sum_dist
                max_combo = combo
            
            # 
            # if sum_dist > max_dist:
            #     max_dist = sum_dist
            #     max_combo = combo
        
        points = hull[max_combo,:]
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray2, points.astype(np.float32), (2,2), (-1, -1), criteria)
        points = corners
        
        points = points.astype(int)

            
        if show:
            final = self.im.copy()
            for point in points:
                cv2.circle(final,(int(point[0]),int(point[1])),3,(0,100,0),-1)
            cv2.imshow("frame",final)
            cv2.waitKey(0)
        
        # sort points clockwise relative to vanishing point
        center = points.mean(axis = 0)
        
        vp = torch.tensor(self.data[self.DIRECTION]["vp"][0:2]).unsqueeze(0)
        #vp = torch.tensor([int(vp[0]),int(vp[1])])
        points = torch.from_numpy(points)                                          
        points = torch.cat((points,vp))
        points = clockify(points,center = torch.from_numpy(center))
        vp_idx = (points == vp).float().mean(dim = 1).int().nonzero()
        
        order = [(vp_idx.item() + i)%5 for i in range(1,5)]
        points = points[order].int()
        
        if self.data[self.DIRECTION]["vp"][2] == True:
            points = points[[2,3,0,1]]
            
        points = points.tolist()
        
        return points
            

    def run(self):
        cv2.namedWindow("window")
        cv2.setMouseCallback("window", self.on_mouse, 0)

        while(True):  # one frame
            
            if self.new:
                
                self.new = False

                try:
                    if self.clicked_point is not None:
                        self.click_handler()
                except:
                    print("EXCEPTION")
                    self.clicked_point =  None
                    self.moved = False
                    fr = np.zeros(self.im.shape)
                    fr[:,:,2] = 255
                    
                    cv2.imshow("window",fr)
                    cv2.waitKey(10)

                self.plot()

            cv2.imshow("window", self.cur_image)
            #title = "{} toggle class (1), switch frame (8-9), clear all (c), undo(u),   quit (q), switch frame (8-9)".format(self.class_names[self.cur_class])
            cv2.setWindowTitle("window", self.cam_name)

            key = cv2.waitKey(1)

            if key == ord("q"):
                self.save()
                break
            elif key == ord("s"):
                self.save()
            elif key == ord("u"):
                self.undo()
            elif key == ord("!"):
                self.load()
            elif key == ord("%"):
                self.get_correspondence()
            elif key == ord("$"):
                self.get_correspondence()
                self.impute_points()
            elif key == ord("="):
                self.DIRECTION = "WB" if self.DIRECTION == "EB" else "EB"
            elif key == ord("t"):
                if self.SHOW_LABELS == 1:
                    self.SHOW_LABELS = 0.5
                elif self.SHOW_LABELS == 0.5:
                    self.SHOW_LABELS = 0.35
                elif self.SHOW_LABELS == 0.35:
                    self.SHOW_LABELS = 0
                elif self.SHOW_LABELS == 0:
                    self.SHOW_LABELS = 1
            elif key == ord("v"):
                 self.SHOW_VP = not self.SHOW_VP
            elif key == ord(" "):
                self.active_index += 1
                self.active_letter = 0
                self.guess_text = "{}_{}_{}_{}".format(self.DIRECTION.lower(),self.active_feature,self.active_index,self.letters[self.active_letter])


            elif key == ord("f"):
                self.ACTIVE = "FOV"
            elif key == ord("p"):
                self.ACTIVE = "POINTS"
            elif key == ord("m"):
                self.ACTIVE = "MASK"
            elif key == ord("c"):
                self.ACTIVE = "CURVE"
            elif key == ord("a"):
                self.ACTIVE = "AUTO POINTS"
            elif key == ord("z"):
                self.ACTIVE = "Z AXIS"
            elif key == ord("d"):
                self.ACTIVE = "DELETE"
                
            elif key == ord("^") and self.ACTIVE == "Z AXIS":
                self.get_z_vp()
                
            elif key == ord("9"):
                self.frame_list_idx = (self.frame_list_idx+1) % len(self.frame_list)
            elif key == ord("8"):
                self.frame_list_idx -= 1
                if self.frame_list_idx < 0:
                    self.frame_list_idx += len(self.frame_list)
                
            if self.data[self.DIRECTION]["vp"] is None:
                self.ACTIVE = "VP"

            if key != -1 or self.moved:
                self.new = True
                self.moved = False
                
            

        cv2.destroyAllWindows()


if __name__ == "__main__":
    for p in range(27,41):
     for c in ["01","02","03","04","05","06"]:
        base_dir = "/home/worklab/Data/dataset_beta/sequence_1"
        base_dir = "/home/worklab/Data/MOTION_homography_10_2022"
        base_dir = "/home/worklab/Data/homo/reference/4k"
        #save_dir = "/home/derek/Documents/i24/i24_homography/data_real"
        #save_dir = "/home/derek/Documents/i24/i24_homography/data"
        save_dir = "/home/worklab/Data/homo/working"
        try:
            camera_name = sys.argv[1]
            #print(camera_name)
            files = os.listdir(base_dir)
            for file in files:
                #print(file)
                if camera_name in file:
                    sequence = os.path.join(base_dir,file)
                    break
            sequence
        except:
            print("No camera name given. Using default Instead"        )
            camera_name = "P{}C{}".format(str(p).zfill(2),c)
            files = os.listdir(base_dir)
            for file in files:
                #print(file)
                if camera_name in file:
                    sequence = os.path.join(base_dir,file)
                    break
    
        frame_name = os.path.join(save_dir,"4k",camera_name) + ".png"
        #frame_name = os.path.join(save_dir,camera_name) + ".png"

        try:
            frame = cv2.imread(frame_name)
            frame_stack = None
            if frame is None: raise FileNotFoundError
        except:
            print("Generating average frame for camera {}".format(camera_name))
            frame, frame_stack = average_frame(sequence)
            frame = frame[:, :,::-1]
            cv2.imwrite(frame_name,frame)
            
        
    
        annotator = CameraAnnotator(frame, frame_list = frame_stack, cam_name=camera_name, save_directory=save_dir, load = True)
        annotator.run()
    #annotator.harris_wheel()