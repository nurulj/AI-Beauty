# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:36:41 2018

@author: holmes
"""
import numpy as np
from six.moves import cPickle
from PIL import Image,ImageOps
import datetime
import os
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# xy_file is containing 2d coord of tsne result (n_samples,n_features)
# xy_id_file containing file name corresponding to n_samples
xy_file = 'xycoor_avg30per.pkl'
xy_id_file = 'xy_id.pkl'
img_dir = '/data/AI-Beauty/PF-500K'

# img_res - desired generated 2D mosaic
# thumbnail_ratio - images ratio to img_res
# border_ratio - the black border size around thumbnails
# nearby_grid - the possible region to place thumbnails in case of repetative coords
# n - number of image plot from n_samples (sequentially)
img_res = (10000,10000)
thumbnail_ratio = (0.005,0.005)
border_ratio = 0.04
nearby_grid = (9,9) # odd
n = 520727


# helper function to return current time as string 
def current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M')

# read pickle files
with open(xy_file,'rb') as fid:
    xy = cPickle.load(fid)
    
with open(xy_id_file,'rb') as fid:
    xy_id = cPickle.load(fid)
    
time_start = time.time()    
# normalize the coord to 0-1 and rescaled to img_res
norm_x = (xy[:,0] - xy[:,0].min()) / (xy[:,0].max() - xy[:,0].min())
norm_y = (xy[:,1] - xy[:,1].min()) / (xy[:,1].max() - xy[:,1].min())

scaled_x = np.ceil(norm_x * img_res[0]).astype(np.int32)
scaled_y = np.ceil(norm_y * img_res[1]).astype(np.int32)

# figure out thumbnail size
thumnail_res = (int(round(img_res[0] * thumbnail_ratio[0])),
                int(round(img_res[1] * thumbnail_ratio[1])))

# figure out thumbnail border size
border_size = int(round(max(thumnail_res) * border_ratio))
if border_size <= 0:
    border_size = 1  

read_thumb_size = (thumnail_res[0] -  border_size*2,thumnail_res[1] -  border_size*2)       
                
# changeing the coords to thumbnail start coords                
thumb_mid = (thumnail_res[0]/2,thumnail_res[1]/2)
scaled_x -= thumb_mid[0]
scaled_y -= thumb_mid[1]

# prevent -ve coord
scaled_x[scaled_x < 0] = 0
scaled_y[scaled_y < 0] = 0

# prevent out of picture
scaled_x[(scaled_x+thumnail_res[0]) >= img_res[0]] = img_res[0] - thumnail_res[0]
scaled_y[(scaled_y+thumnail_res[1]) >= img_res[1]] = img_res[1] - thumnail_res[1]


# partition img_res to 2D grids according to thumbnail size
gridx_n = img_res[0] /thumnail_res[0]
gridy_n = img_res[1] /thumnail_res[1]
matgrid = np.zeros((gridx_n,gridy_n)).astype(np.bool)

# snap the floating number coords to integer
gridx = np.ceil(norm_x  * (gridx_n-1)).astype(np.int64)
gridy = np.ceil(norm_y  * (gridy_n-1)).astype(np.int64)


# gridcoord - holds the coords of thumbnails corresponds to matgrid
# valid - keep track if the candidate thumnails can find possible grid to fit
gridcoord = np.zeros((n,2)).astype(np.int64)
valid = np.zeros((n)).astype(np.bool)

# loop thru the n samples to find out where the image lies in matgrid
centergrid = np.asarray(nearby_grid)/2
for i in range(n):
    if matgrid[gridx[i],gridy[i]] == True:
        print('%i sth here (%i,%i)'%(i,gridx[i],gridy[i]))
        startgridx = gridx[i] - centergrid[0]
        startgridy = gridy[i] - centergrid[1]
        if startgridx < 0:
            startgridx = 0
        if startgridy < 0:
            startgridy = 0            
        endgridx = startgridx + nearby_grid[0] 
        endgridy = startgridy + nearby_grid[1] 
        if endgridx > gridx_n:
            endgridx = gridx_n
        if endgridy > gridy_n:
            endgridy = gridy_n         
        
        possiblex = np.arange(startgridx,endgridx)
        possibley = np.arange(startgridy,endgridy)
        mask =  np.zeros((gridx_n,gridy_n)).astype(np.bool)
        mask[startgridx:endgridx,startgridy:endgridy] = True
        mask[gridx[i],gridy[i]] = False
        mask[matgrid] = False
        xc,yc = np.where(mask)
        if xc.size != 0:
            a = (gridx[i] - xc)**2
            b = (gridy[i] - yc)**2
            c = np.sqrt(a + b)
            shortest = np.argmin(c)
            print('%i possible candidate grid (%i:%i,%i:%i)'%(i,startgridx,startgridy,endgridx,endgridy))
            print('%i closest candidate grid (%i,%i) - distance = %.2f'%(i,xc[shortest],yc[shortest],c[shortest]))
            matgrid[xc[shortest],yc[shortest]] = True
            gridcoord[i,0] = xc[shortest]
            gridcoord[i,1] = yc[shortest]
            valid[i] = True
        else:
            print('%i NO possible candidate grid (%i:%i,%i:%i)'%(i,startgridx,startgridy,endgridx,endgridy))
    else:
        matgrid[gridx[i],gridy[i]] = True   
        gridcoord[i,0] = gridx[i]
        gridcoord[i,1] = gridy[i]
        valid[i] = True
        #print('%i free space'%i)

# convert the grid coordinate to real img_res coordinates
realcoordx = gridcoord[:,0] * thumnail_res[0]
realcoordy = gridcoord[:,1] * thumnail_res[1]

# create image buffer
im = Image.new('RGB',img_res,'white')

# read the image provided in xy_id
xy_path = [os.path.join(img_dir,x) for x in xy_id]

# pasting image accordingly to buffer
for i in range(n):
    im2 = Image.open(xy_path[i])
    im2 = im2.resize(read_thumb_size)
    im2 = ImageOps.expand(im2,border=border_size,fill=0)
    im.paste(im2,(realcoordx[i],realcoordy[i]))


# save the results
im.save('%s_scatter.jpg'%current_time())
print 'done! Time elapsed: {} seconds'.format(time.time()-time_start)

