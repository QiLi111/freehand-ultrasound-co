""" invdisttree.py: inverse-distance-weighted interpolation using KDTree
    fast, solid, local
"""
from __future__ import division
import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree as KDTree
# from mayavi import mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch, time
from torch.autograd import Variable

    # http://docs.scipy.org/doc/scipy/reference/spatial.html

__date__ = "2010-11-09 Nov"  # weights, doc

#...............................................................................
class Invdisttree:
    """ inverse-distance-weighted interpolation using KDTree:
invdisttree = Invdisttree( X, z )  -- data points, values
interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
    interpolates z from the 3 points nearest each query point q;
    For example, interpol[ a query point q ]
    finds the 3 data points nearest q, at distances d1 d2 d3
    and returns the IDW average of the values z1 z2 z3
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
        = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

    q may be one point, or a batch of points.
    eps: approximate nearest, dist <= (1 + eps) * true nearest
    p: use 1 / distance**p
    weights: optional multipliers for 1 / distance**p, of the same shape as q
    stat: accumulate wsum, wn for average weights

How many nearest neighbors should one take ?
a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
    |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
    I find that runtimes don't increase much at all with nnear -- ymmv.

p=1, p=2 ?
    p=2 weights nearer points more, farther points less.
    In 2d, the circles around query points have areas ~ distance**2,
    so p=2 is inverse-area weighting. For example,
        (z1/area1 + z2/area2 + z3/area3)
        / (1/area1 + 1/area2 + 1/area3)
        = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
    Similarly, in 3d, p=3 is inverse-volume weighting.

Scaling:
    if different X coordinates measure different things, Euclidean distance
    can be way off.  For example, if X0 is in the range 0 to 1
    but X1 0 to 1000, the X1 distances will swamp X0;
    rescale the data, i.e. make X0.std() ~= X1.std() .

A nice property of IDW is that it's scale-free around query points:
if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
the IDW average
    (z1/d1 + z2/d2 + z3/d3)
    / (1/d1 + 1/d2 + 1/d3)
is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
is exceedingly sensitive to distance and to h.

    """
# anykernel( dj / av dj ) is also scale-free
# error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__( self, X, z, leafsize=10, stat=0 ):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = KDTree( X, leafsize=leafsize )  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None;

    def __call__( self, q, nnear=6, eps=0, p=1, weights=None ):
            # nnear nearest neighbours of each query point --
        q = np.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        self.distances, self.ix = self.tree.query( q, k=nnear, eps=eps )
        interpol = np.zeros( (len(self.distances),) + np.shape(self.z[0]) )
        jinterpol = 0
        for dist, ix in zip( self.distances, self.ix ):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= np.sum(w)
                wz = np.dot( w, self.z[ix] )
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1  else interpol[0]
    

def save2mha(data,sx,sy,sz,save_folder):
    # save 3D volume into volume, and then can use 3D slice to look

    img=sitk.GetImageFromArray(data.transpose([2,1,0])) # ZYX
    img.SetSpacing((sx,sy,sz))
    sitk.WriteImage(img,save_folder)

def save2img(x,y,z,data,save_folder):
    # volume = mlab.pipeline.volume(mlab.pipeline.scalar_field(data), 
    #                             #   vmin=0, 
    #                             #   vmax=0.8
    #                               )

    # mlab.draw()
    # mlab.savefig(save_folder)
    data_ravel = data.ravel()
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(
        x, y, z, 
               c=(data_ravel - np.min(data_ravel)) / (np.max(data_ravel) - np.min(data_ravel)), 
               cmap=plt.get_cmap('Greys')
               )

    # plt.show()
    plt.savefig(save_folder)
    plt.close()


def scatter2GridIndex(x_norm_i,y_norm_i,z_norm_i,x_low,x_up,y_low,y_up,z_low,z_up,frames_flatten_i,weight4pixel,intensity4pixel):
    # given the normalized coordinates of each axis. "Normalized" means that 
    # the unit is 1, such that index of floor(data) or ceil(data) is itself. 

    

    # compute the weight to neigbour points   
    weight4pixel[x_low][y_low][z_low] += (x_up-x_norm_i)*(y_up-y_norm_i)*(z_up-z_norm_i)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    weight4pixel[x_low][y_low][z_up] +=(x_up-x_norm_i)*(y_up-y_norm_i)*(z_norm_i-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    weight4pixel[x_low][y_up][z_low] +=(x_up-x_norm_i)*(y_norm_i-y_low)*(z_up-z_norm_i)
    weight4pixel[x_low][y_up][z_up]  +=(x_up-x_norm_i)*(y_norm_i-y_low)*(z_norm_i-z_low)

    weight4pixel[x_up][y_low][z_low]  +=(x_norm_i-x_low)*(y_up-y_norm_i)*(z_up-z_norm_i)
    weight4pixel[x_up][y_low][z_up]  +=(x_norm_i-x_low)*(y_up-y_norm_i)*(z_norm_i-z_low)
    weight4pixel[x_up][y_up][z_low]  +=(x_norm_i-x_low)*(y_norm_i-y_low)*(z_up-z_norm_i)
    weight4pixel[x_up][y_up][z_up]  +=(x_norm_i-x_low)*(y_norm_i-y_low)*(z_norm_i-z_low)


    # # save the intensity 

    # intensity4pixel_each_point = frames_flatten_i*weight4pixel_each_point 


    intensity4pixel[x_low][y_low][z_low] += frames_flatten_i*(x_up-x_norm_i)*(y_up-y_norm_i)*(z_up-z_norm_i)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    intensity4pixel[x_low][y_low][z_up] +=frames_flatten_i*(x_up-x_norm_i)*(y_up-y_norm_i)*(z_norm_i-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    intensity4pixel[x_low][y_up][z_low] +=frames_flatten_i*(x_up-x_norm_i)*(y_norm_i-y_low)*(z_up-z_norm_i)
    intensity4pixel[x_low][y_up][z_up]  +=frames_flatten_i*(x_up-x_norm_i)*(y_norm_i-y_low)*(z_norm_i-z_low)

    intensity4pixel[x_up][y_low][z_low]  +=frames_flatten_i*(x_norm_i-x_low)*(y_up-y_norm_i)*(z_up-z_norm_i)
    intensity4pixel[x_up][y_low][z_up]  +=frames_flatten_i*(x_norm_i-x_low)*(y_up-y_norm_i)*(z_norm_i-z_low)
    intensity4pixel[x_up][y_up][z_low]  +=frames_flatten_i*(x_norm_i-x_low)*(y_norm_i-y_low)*(z_up-z_norm_i)
    intensity4pixel[x_up][y_up][z_up]  +=frames_flatten_i*(x_norm_i-x_low)*(y_norm_i-y_low)*(z_norm_i-z_low)

    # intensity4pixel[x_low][y_low][z_low]  +=(frames_flatten_i*weight4pixel[x_low][y_low][z_low])   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    # intensity4pixel[x_low][y_low][z_up]  +=(frames_flatten_i*weight4pixel[x_low][y_low][z_up])  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    # intensity4pixel[x_low][y_up][z_low] +=(frames_flatten_i*weight4pixel[x_low][y_up][z_low])
    # intensity4pixel[x_low][y_up][z_up] +=(frames_flatten_i*weight4pixel[x_low][y_up][z_up])

    # intensity4pixel[x_up][y_low][z_low] +=(frames_flatten_i*weight4pixel[x_up][y_low][z_low])
    # intensity4pixel[x_up][y_low][z_up] +=(frames_flatten_i*weight4pixel[x_up][y_low][z_up])
    # intensity4pixel[x_up][y_up][z_low] +=(frames_flatten_i*weight4pixel[x_up][y_up][z_low])
    # intensity4pixel[x_up][y_up][z_up]  +=(frames_flatten_i*weight4pixel[x_up][y_up][z_up])



    # x_low,x_up = torch.floor(x_norm_i).to(torch.int), torch.ceil(x_norm_i).to(torch.int)
    # y_low,y_up = torch.floor(y_norm_i).to(torch.int), torch.ceil(y_norm_i).to(torch.int)
    # z_low,z_up = torch.floor(z_norm_i).to(torch.int), torch.ceil(z_norm_i).to(torch.int)


    # compute the weight to neigbour points  
    # weight4pixel[x_low][y_low][z_low] = (x_up-x_norm_i)*(y_up-y_norm_i)*(z_up-z_norm_i) if torch.isnan(weight4pixel[x_low][y_low][z_low])  else weight4pixel[x_low][y_low][z_low]+(x_up-x_norm_i)*(y_up-y_norm_i)*(z_up-z_norm_i)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    # weight4pixel[x_low][y_low][z_up] = (x_up-x_norm_i)*(y_up-y_norm_i)*(z_norm_i-z_low) if torch.isnan(weight4pixel[x_low][y_low][z_up]) else weight4pixel[x_low][y_low][z_up]+(x_up-x_norm_i)*(y_up-y_norm_i)*(z_norm_i-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    # weight4pixel[x_low][y_up][z_low] = (x_up-x_norm_i)*(y_norm_i-y_low)*(z_up-z_norm_i) if torch.isnan(weight4pixel[x_low][y_up][z_low])  else weight4pixel[x_low][y_up][z_low]+(x_up-x_norm_i)*(y_norm_i-y_low)*(z_up-z_norm_i)
    # weight4pixel[x_low][y_up][z_up] = (x_up-x_norm_i)*(y_norm_i-y_low)*(z_norm_i-z_low) if torch.isnan(weight4pixel[x_low][y_up][z_up]) else weight4pixel[x_low][y_up][z_up]+(x_up-x_norm_i)*(y_norm_i-y_low)*(z_norm_i-z_low)

    # weight4pixel[x_up][y_low][z_low] = (x_norm_i-x_low)*(y_up-y_norm_i)*(z_up-z_norm_i) if torch.isnan(weight4pixel[x_up][y_low][z_low]) else weight4pixel[x_up][y_low][z_low]+(x_norm_i-x_low)*(y_up-y_norm_i)*(z_up-z_norm_i)
    # weight4pixel[x_up][y_low][z_up] = (x_norm_i-x_low)*(y_up-y_norm_i)*(z_norm_i-z_low) if torch.isnan(weight4pixel[x_up][y_low][z_up]) else weight4pixel[x_up][y_low][z_up]+(x_norm_i-x_low)*(y_up-y_norm_i)*(z_norm_i-z_low)
    # weight4pixel[x_up][y_up][z_low] = (x_norm_i-x_low)*(y_norm_i-y_low)*(z_up-z_norm_i) if torch.isnan(weight4pixel[x_up][y_up][z_low]) else weight4pixel[x_up][y_up][z_low]+(x_norm_i-x_low)*(y_norm_i-y_low)*(z_up-z_norm_i)
    # weight4pixel[x_up][y_up][z_up] = (x_norm_i-x_low)*(y_norm_i-y_low)*(z_norm_i-z_low) if torch.isnan(weight4pixel[x_up][y_up][z_up]) else weight4pixel[x_up][y_up][z_up]+(x_norm_i-x_low)*(y_norm_i-y_low)*(z_norm_i-z_low)

    # # save the intensity  

    # intensity4pixel[x_low][y_low][z_low] = frames_flatten_i*weight4pixel[x_low][y_low][z_low]  if torch.isnan(intensity4pixel[x_low][y_low][z_low]) else intensity4pixel[x_low][y_low][z_low]+(frames_flatten_i*weight4pixel[x_low][y_low][z_low])   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    # intensity4pixel[x_low][y_low][z_up] = frames_flatten_i*weight4pixel[x_low][y_low][z_up] if torch.isnan(intensity4pixel[x_low][y_low][z_up]) else intensity4pixel[x_low][y_low][z_up]+(frames_flatten_i*weight4pixel[x_low][y_low][z_up])  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    # intensity4pixel[x_low][y_up][z_low] = frames_flatten_i*weight4pixel[x_low][y_up][z_low] if torch.isnan(intensity4pixel[x_low][y_up][z_low]) else intensity4pixel[x_low][y_up][z_low]+(frames_flatten_i*weight4pixel[x_low][y_up][z_low])
    # intensity4pixel[x_low][y_up][z_up] = frames_flatten_i*weight4pixel[x_low][y_up][z_up] if torch.isnan(intensity4pixel[x_low][y_up][z_up]) else intensity4pixel[x_low][y_up][z_up]+(frames_flatten_i*weight4pixel[x_low][y_up][z_up])

    # intensity4pixel[x_up][y_low][z_low] = frames_flatten_i*weight4pixel[x_up][y_low][z_low] if torch.isnan(intensity4pixel[x_up][y_low][z_low])  else intensity4pixel[x_up][y_low][z_low]+(frames_flatten_i*weight4pixel[x_up][y_low][z_low])
    # intensity4pixel[x_up][y_low][z_up] = frames_flatten_i*weight4pixel[x_up][y_low][z_up] if torch.isnan(intensity4pixel[x_up][y_low][z_up])  else intensity4pixel[x_up][y_low][z_up]+(frames_flatten_i*weight4pixel[x_up][y_low][z_up])
    # intensity4pixel[x_up][y_up][z_low] = frames_flatten_i*weight4pixel[x_up][y_up][z_low] if torch.isnan(intensity4pixel[x_up][y_up][z_low])  else intensity4pixel[x_up][y_up][z_low]+(frames_flatten_i*weight4pixel[x_up][y_up][z_low])
    # intensity4pixel[x_up][y_up][z_up] = frames_flatten_i*weight4pixel[x_up][y_up][z_up] if torch.isnan(intensity4pixel[x_up][y_up][z_up]) else intensity4pixel[x_up][y_up][z_up]+(frames_flatten_i*weight4pixel[x_up][y_up][z_up])

    return weight4pixel,intensity4pixel

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def eight_neighbour_points(x,y,z):
    # find the eight_neighbour_points
    # in this case, the sum of x_low and x_up must be equal to 1

    x_low = torch.floor(x).to(torch.int)
    y_low = torch.floor(y).to(torch.int)
    z_low = torch.floor(z).to(torch.int)

    return x_low,x_low+1,y_low,y_low+1,z_low,z_low+1

def eright_points_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,xl,yl,zl,initial):
    # get the coordinates of the erights neighbours of all scatter points
    # as the coordinates of scatter points are nomalised, without unit, such that the 
    # coordinates of eright grid neighbours points are also the index of these points

    x_up[x_up >= xl] = xl-1
    y_up[y_up >= yl] = yl-1
    z_up[z_up >= zl] = zl-1

    # get the coordinates of erights neighbours points for all scatter points

    p1 = torch.stack((x_low,y_low,z_low),dim=1)
    p2 = torch.stack((x_low,y_low,z_up),dim=1)
    p3 = torch.stack((x_low,y_up,z_low),dim=1)
    p4 = torch.stack((x_low,y_up,z_up),dim=1)

    p5 = torch.stack((x_up,y_low,z_low),dim=1)
    p6 = torch.stack((x_up,y_low,z_up),dim=1)
    p7 = torch.stack((x_up,y_up,z_low),dim=1)
    p8 = torch.stack((x_up,y_up,z_up),dim=1)

    # get the 1-d index of each points

    p1_1d = xyz2idx(p1,xl,yl,zl)
    p2_1d = xyz2idx(p2,xl,yl,zl)
    p3_1d = xyz2idx(p3,xl,yl,zl)
    p4_1d = xyz2idx(p4,xl,yl,zl)

    p5_1d = xyz2idx(p5,xl,yl,zl)
    p6_1d = xyz2idx(p6,xl,yl,zl)
    p7_1d = xyz2idx(p7,xl,yl,zl)
    p8_1d = xyz2idx(p8,xl,yl,zl)

    # update batched index
    p1_1d = p1_1d + initial*xl*yl*zl
    p2_1d = p2_1d + initial*xl*yl*zl
    p3_1d = p3_1d + initial*xl*yl*zl
    p4_1d = p4_1d + initial*xl*yl*zl

    p5_1d = p5_1d + initial*xl*yl*zl
    p6_1d = p6_1d + initial*xl*yl*zl
    p7_1d = p7_1d + initial*xl*yl*zl
    p8_1d = p8_1d + initial*xl*yl*zl




    

    return torch.cat((p1_1d,p2_1d,p3_1d,p4_1d,p5_1d,p6_1d,p7_1d,p8_1d))#.to(torch.long)


def eright_points_in_1d_1(x_low,x_up,y_low,y_up,z_low,z_up,xl,yl,zl):
    # get the coordinates of the erights neighbours of all scatter points
    # as the coordinates of scatter points are nomalised, without unit, such that the 
    # coordinates of eright grid neighbours points are also the index of these points

    x_up[x_up >= xl] = xl-1
    y_up[y_up >= yl] = yl-1
    z_up[z_up >= zl] = zl-1

    # get the coordinates of erights neighbours points for all scatter points

    p1 = torch.stack((x_low,y_low,z_low),dim=1)
    p2 = torch.stack((x_low,y_low,z_up),dim=1)
    p3 = torch.stack((x_low,y_up,z_low),dim=1)
    p4 = torch.stack((x_low,y_up,z_up),dim=1)

    p5 = torch.stack((x_up,y_low,z_low),dim=1)
    p6 = torch.stack((x_up,y_low,z_up),dim=1)
    p7 = torch.stack((x_up,y_up,z_low),dim=1)
    p8 = torch.stack((x_up,y_up,z_up),dim=1)

    # get the 1-d index of each points

    p1_1d = xyz2idx(p1,xl,yl,zl)
    p2_1d = xyz2idx(p2,xl,yl,zl)
    p3_1d = xyz2idx(p3,xl,yl,zl)
    p4_1d = xyz2idx(p4,xl,yl,zl)

    p5_1d = xyz2idx(p5,xl,yl,zl)
    p6_1d = xyz2idx(p6,xl,yl,zl)
    p7_1d = xyz2idx(p7,xl,yl,zl)
    p8_1d = xyz2idx(p8,xl,yl,zl)



    

    return torch.cat((p1_1d,p2_1d,p3_1d,p4_1d,p5_1d,p6_1d,p7_1d,p8_1d))#.to(torch.long)


def weight_intensity_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm,frames_flatten,options):
    # compute weight for each grid points
    # NOTE the order of the stacked eight neighbour points should be the same as in function `eright_points_in_1d`

    if options == 'bilinear':

        weight4pixel_x_low_y_low_z_low = (x_up-x_norm)*(y_up-y_norm)*(z_up-z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        weight4pixel_x_low_y_low_z_up =(x_up-x_norm)*(y_up-y_norm)*(z_norm-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        weight4pixel_x_low_y_up_z_low =(x_up-x_norm)*(y_norm-y_low)*(z_up-z_norm)
        weight4pixel_x_low_y_up_z_up  =(x_up-x_norm)*(y_norm-y_low)*(z_norm-z_low)

        weight4pixel_x_up_y_low_z_low  =(x_norm-x_low)*(y_up-y_norm)*(z_up-z_norm)
        weight4pixel_x_up_y_low_z_up  =(x_norm-x_low)*(y_up-y_norm)*(z_norm-z_low)
        weight4pixel_x_up_y_up_z_low  =(x_norm-x_low)*(y_norm-y_low)*(z_up-z_norm)
        weight4pixel_x_up_y_up_z_up  =(x_norm-x_low)*(y_norm-y_low)*(z_norm-z_low)

        intensity4pixel_x_low_y_low_z_low = frames_flatten*(x_up-x_norm)*(y_up-y_norm)*(z_up-z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        intensity4pixel_x_low_y_low_z_up =frames_flatten*(x_up-x_norm)*(y_up-y_norm)*(z_norm-z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        intensity4pixel_x_low_y_up_z_low =frames_flatten*(x_up-x_norm)*(y_norm-y_low)*(z_up-z_norm)
        intensity4pixel_x_low_y_up_z_up  =frames_flatten*(x_up-x_norm)*(y_norm-y_low)*(z_norm-z_low)

        intensity4pixel_x_up_y_low_z_low  =frames_flatten*(x_norm-x_low)*(y_up-y_norm)*(z_up-z_norm)
        intensity4pixel_x_up_y_low_z_up  =frames_flatten*(x_norm-x_low)*(y_up-y_norm)*(z_norm-z_low)
        intensity4pixel_x_up_y_up_z_low  =frames_flatten*(x_norm-x_low)*(y_norm-y_low)*(z_up-z_norm)
        intensity4pixel_x_up_y_up_z_up  =frames_flatten*(x_norm-x_low)*(y_norm-y_low)*(z_norm-z_low)
    
    elif options == 'IDW':

        weight4pixel_x_low_y_low_z_low = 1/torch.sqrt((x_low-x_norm)**2+ (y_low-y_norm)**2 + (z_low-z_norm)**2)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        weight4pixel_x_low_y_low_z_up =1/torch.sqrt((x_low-x_norm)**2+ (y_low-y_norm)**2 + (z_up-z_norm)**2)   #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        weight4pixel_x_low_y_up_z_low =1/torch.sqrt((x_low-x_norm)**2+ (y_up-y_norm)**2 + (z_low-z_norm)**2) 
        weight4pixel_x_low_y_up_z_up  =1/torch.sqrt((x_low-x_norm)**2+ (y_up-y_norm)**2 + (z_up-z_norm)**2) 

        weight4pixel_x_up_y_low_z_low  =1/torch.sqrt((x_up-x_norm)**2+ (y_low-y_norm)**2 + (z_low-z_norm)**2)
        weight4pixel_x_up_y_low_z_up  =1/torch.sqrt((x_up-x_norm)**2+ (y_low-y_norm)**2 + (z_up-z_norm)**2)
        weight4pixel_x_up_y_up_z_low  =1/torch.sqrt((x_up-x_norm)**2+ (y_up-y_norm)**2 + (z_low-z_norm)**2)
        weight4pixel_x_up_y_up_z_up  =1/torch.sqrt((x_up-x_norm)**2+ (y_up-y_norm)**2 + (z_up-z_norm)**2)

        intensity4pixel_x_low_y_low_z_low = frames_flatten*weight4pixel_x_low_y_low_z_low   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
        intensity4pixel_x_low_y_low_z_up =frames_flatten*weight4pixel_x_low_y_low_z_up #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
        intensity4pixel_x_low_y_up_z_low =frames_flatten*weight4pixel_x_low_y_up_z_low
        intensity4pixel_x_low_y_up_z_up  =frames_flatten*weight4pixel_x_low_y_up_z_up

        intensity4pixel_x_up_y_low_z_low  =frames_flatten*weight4pixel_x_up_y_low_z_low
        intensity4pixel_x_up_y_low_z_up  =frames_flatten*weight4pixel_x_up_y_low_z_up
        intensity4pixel_x_up_y_up_z_low  =frames_flatten*weight4pixel_x_up_y_up_z_low
        intensity4pixel_x_up_y_up_z_up  =frames_flatten*weight4pixel_x_up_y_up_z_up
    else:
        raise("Not supported")


    weight4pixel_8neighbour_pts = torch.cat((weight4pixel_x_low_y_low_z_low,weight4pixel_x_low_y_low_z_up,
                              weight4pixel_x_low_y_up_z_low,weight4pixel_x_low_y_up_z_up,
                            weight4pixel_x_up_y_low_z_low,weight4pixel_x_up_y_low_z_up,
                            weight4pixel_x_up_y_up_z_low,weight4pixel_x_up_y_up_z_up
                            ))
    
  
  
    intensity4pixel_8neighbour_pts = torch.cat((intensity4pixel_x_low_y_low_z_low,intensity4pixel_x_low_y_low_z_up,
                                intensity4pixel_x_low_y_up_z_low,intensity4pixel_x_low_y_up_z_up,
                                intensity4pixel_x_up_y_low_z_low,intensity4pixel_x_up_y_low_z_up,
                                intensity4pixel_x_up_y_up_z_low,intensity4pixel_x_up_y_up_z_up
                                ))

    return weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts


def combine_values(weight4pixel,intensity4pixel,weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts,idx_1d):
    # combine weighs and weighted inetnsity for the same grid points

    weight4pixel.scatter_add_(0, idx_1d.to(torch.long), weight4pixel_8neighbour_pts)
    intensity4pixel.scatter_add_(0, idx_1d.to(torch.long), intensity4pixel_8neighbour_pts)

    return weight4pixel,intensity4pixel

    # return weight4pixel_8neighbour_pts[0:int(weight4pixel_8neighbour_pts.size()[0]/8)],intensity4pixel_8neighbour_pts[0:int(weight4pixel_8neighbour_pts.size()[0]/8)]






def xyz2idx(xyz,xl,yl,zl):
    # Transform coordinates of a 3D volume of certain sizes into a list of indices of 1D array.
    
    index_1d = (xyz[:,2]+xyz[:,1]*zl+xyz[:,0]*(yl*zl))
    
    return index_1d





def bilinear_weights(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm):
    # compute the weight of each axis for all scattered points

    w1 = x_up-x_norm
    w2 = y_up-y_norm
    w3 = z_up-z_norm   
    w4 = z_norm-z_low  
    w5 = y_norm-y_low
    w6 = x_norm-x_low

    return w1,w2,w3,w4,w5,w6




def scatter2Grid_2(x_up_x_norm, y_up_y_norm, z_up_z_norm, z_norm_z_low, y_norm_y_low, x_norm_x_low,frames_flatten,X, Y, Z,x_low,x_up,y_low,y_up,z_low,z_up):

    # compute the contribution to neighbour points of all scatter points
    # weights
    w_neigr0 =(x_up_x_norm)*(y_up_y_norm)*(z_up_z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    w_neigr1 =(x_up_x_norm)*(y_up_y_norm)*(z_norm_z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    w_neigr2 =(x_up_x_norm)*(y_norm_y_low)*(z_up_z_norm)
    w_neigr3 =(x_up_x_norm)*(y_norm_y_low)*(z_norm_z_low)

    w_neigr4 =(x_norm_x_low)*(y_up_y_norm)*(z_up_z_norm)
    w_neigr5 =(x_norm_x_low)*(y_up_y_norm)*(z_norm_z_low)
    w_neigr6 =(x_norm_x_low)*(y_norm_y_low)*(z_up_z_norm)
    w_neigr7 =(x_norm_x_low)*(y_norm_y_low)*(z_norm_z_low)
    
    # intensity

    intensity_0 = w_neigr0*frames_flatten
    intensity_1 = w_neigr1*frames_flatten
    intensity_2 = w_neigr2*frames_flatten
    intensity_3 = w_neigr3*frames_flatten

    intensity_4 = w_neigr4*frames_flatten
    intensity_5 = w_neigr5*frames_flatten
    intensity_6 = w_neigr6*frames_flatten
    intensity_7 = w_neigr7*frames_flatten

    # sum the weight and intensity for the same grid points

    X_flatten,Y_flatten,Z_flatten = X.flatten(), Y.flatten(), Z.flatten()
    grid_data = torch.stack([X_flatten,Y_flatten,Z_flatten], dim=1)

    grid_0,grid_1,grid_2,grid_3,grid_4,grid_5,grid_6,grid_7 = nn_grid(x_low,x_up,y_low,y_up,z_low,z_up)

    indices_0 = index_in_matrix(grid_0,grid_data)
    indices_1 = index_in_matrix(grid_1,grid_data)
    indices_2 = index_in_matrix(grid_2,grid_data)
    indices_3 = index_in_matrix(grid_3,grid_data)
    indices_4 = index_in_matrix(grid_4,grid_data)
    indices_5 = index_in_matrix(grid_5,grid_data)
    indices_6 = index_in_matrix(grid_6,grid_data)
    indices_7 = index_in_matrix(grid_7,grid_data)

    w_neigr0_final = torch.mm(w_neigr0,indices_0)
    w_neigr1_final = torch.mm(w_neigr1,indices_1)
    w_neigr2_final = torch.mm(w_neigr2,indices_2)
    w_neigr3_final = torch.mm(w_neigr3,indices_3)
    w_neigr4_final = torch.mm(w_neigr4,indices_4)
    w_neigr5_final = torch.mm(w_neigr5,indices_5)
    w_neigr6_final = torch.mm(w_neigr6,indices_6)
    w_neigr7_final = torch.mm(w_neigr7,indices_7)

    intensity_0_final = torch.mm(intensity_0,indices_0)
    intensity_1_final = torch.mm(intensity_1,indices_1)
    intensity_2_final = torch.mm(intensity_2,indices_2)
    intensity_3_final = torch.mm(intensity_3,indices_3)
    intensity_4_final = torch.mm(intensity_4,indices_4)
    intensity_5_final = torch.mm(intensity_5,indices_5)
    intensity_6_final = torch.mm(intensity_6,indices_6)
    intensity_7_final = torch.mm(intensity_7,indices_7)

    w_final = w_neigr0_final+w_neigr1_final+w_neigr2_final+w_neigr3_final+w_neigr4_final+w_neigr5_final+w_neigr6_final+w_neigr7_final
    i_final = intensity_0_final+intensity_1_final+intensity_2_final+intensity_3_final+intensity_4_final+intensity_5_final+intensity_6_final+intensity_7_final
    return i_final/w_final







    





def nn_grid(x_low,x_up,y_low,y_up,z_low,z_up):
    #  compute all the gird coordinates  


    grid_1 = torch.stack([x_low,y_low,z_low], dim=1)
    grid_2 = torch.stack([x_low,y_low,z_up], dim=1)
    grid_3 = torch.stack([x_low,y_up,z_low], dim=1)
    grid_4 = torch.stack([x_low,y_up,z_up], dim=1)

    grid_5 = torch.stack([x_up,y_low,z_low], dim=1)
    grid_6 = torch.stack([x_up,y_low,z_up], dim=1)
    grid_7 = torch.stack([x_up,y_up,z_low], dim=1)
    grid_8 = torch.stack([x_up,y_up,z_up], dim=1)

    return grid_1,grid_2,grid_3,grid_4,grid_5,grid_6,grid_7,grid_8



def index_in_matrix(matrix_of_vectors,another_matrix):
    equality_matrix = (matrix_of_vectors.unsqueeze(1) == another_matrix.unsqueeze(0)).all(dim=-1)
    equality_matrix =equality_matrix.long()
    return equality_matrix

    

def scatter2Grid_2(x_up_x_norm, y_up_y_norm, z_up_z_norm, z_norm_z_low, y_norm_y_low, x_norm_x_low,frames_flatten,X, Y, Z,x_low,x_up,y_low,y_up,z_low,z_up):

    # compute the contribution to neighbour points of all scatter points
    # weights
    w_neigr0 =(x_up_x_norm)*(y_up_y_norm)*(z_up_z_norm)   #x_low_y_low_z_low = (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    w_neigr1 =(x_up_x_norm)*(y_up_y_norm)*(z_norm_z_low)  #(1-(x-x_low))(1-(y-y_low))(1-(z_up-z))
    w_neigr2 =(x_up_x_norm)*(y_norm_y_low)*(z_up_z_norm)
    w_neigr3 =(x_up_x_norm)*(y_norm_y_low)*(z_norm_z_low)

    w_neigr4 =(x_norm_x_low)*(y_up_y_norm)*(z_up_z_norm)
    w_neigr5 =(x_norm_x_low)*(y_up_y_norm)*(z_norm_z_low)
    w_neigr6 =(x_norm_x_low)*(y_norm_y_low)*(z_up_z_norm)
    w_neigr7 =(x_norm_x_low)*(y_norm_y_low)*(z_norm_z_low)
    
    # intensity

    intensity_0 = w_neigr0*frames_flatten
    intensity_1 = w_neigr1*frames_flatten
    intensity_2 = w_neigr2*frames_flatten
    intensity_3 = w_neigr3*frames_flatten

    intensity_4 = w_neigr4*frames_flatten
    intensity_5 = w_neigr5*frames_flatten
    intensity_6 = w_neigr6*frames_flatten
    intensity_7 = w_neigr7*frames_flatten

    # sum the weight and intensity for the same grid points

    # all the left-up corner points, left-low corner points, right-up corner points,...
    grid_0,grid_1,grid_2,grid_3,grid_4,grid_5,grid_6,grid_7 = nn_grid(x_low,x_up,y_low,y_up,z_low,z_up)

    

def interpolation_3D_pytorch_batched(scatter_pts,frames,time_log,saved_folder_test,scan_name,device,option,volume_size,volume_position = None):
    # interpolate 
    # given scatter points in 3D, return the grid points

    # inteplote from scatter data to grid data
    # different from typical inteplotion methods, we index all the scattered points,
    # compute the contribution to the nergbours
    '''

    The idea is to loop all scatter points, compute the contribution to the 2**3 neighbour grid points, 
    and then sum those contributions which are into the same grid points from difference scatter points
    
    intensity of grid points = (weight1*intensity1 + weight2*intensity2 + ... + weightn*intensityn)/(weight1+weight2+..._weightn)
    where n denotes the number of scatter points which has contributions into this grid point

    The core iead is to find the index of contribution scatter points for each grid point
    and sum up all the contribution for each grid point



    Two things are import, the first one is to normalise the coordinates of the scatter points to a unitless one,
    the second is to convert the interpolation problem into a math problem - sum up all the contributions into the same grid points, which can be regarded as a math problem - sum up with known index

    Steps:
    1) Normalise the coordinates of the scatter points to a unitless one, such that the coordinates of the grid points (which is the 8 neigibour grid points computed using torch.floor or torch.ceil) is also the index of this point
       Normalise method: (pts - min(pts))/ step
       where
       xsize,ysize,zsize = int(((max_x)-(min_x))),int(((max_y)-(min_y))),int(((max_z)-(min_z))) #modify xsize as you like
       xstep,ystep,zstep = (max_x-min_x)/(xsize-1),(max_y-min_y)/(ysize-1),(max_z-min_z)/(zsize-1)
        
       For exmaple, in 1D, scatter point A = 3.5, B = 5.6, C = 8.2, D = 10.7
       either size or step is specified
       for example, size = 4
       step = (10.7-3.5)/(4-1) = 2.4
       grid = {}
       after normalisition, A = (3.5-3.5)/2.4 = 0, B = (5.6-3.5)/2.4 = 0.875, C = (8.2-3.5)/2.4 = 1.958, D = (10.7-3.5)/2.4 = 3.0
       after torch.floor, A_low = 1, B_low = 2, C_low = 3

    2) Find 8 neighbour grid points of all scatter points. NOTE: don't use torch.floor and torch.ceil simultaneously, as the distance between x_low and X_up should be eauql to 1.
       In terms of integers, torch.floor and torch.ceil will both be the integers
       Use torch.floor and torch.floor + 1 instead.

    3) For each scatter point, compute the contribution for the 8 neighbour grid points
       Can use a simple bilinear method: the weight for the grid data can be the (1-distance) to the grid data, the final weight is multiplied by each sxis
       for example, for grid points x_low, y_low, z_low =  (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    
    4) convert the 3d array index into 1d index to use TORCH.TENSOR.SCATTER_ADD_ function

       given the volume = torch.zeros((X_shape,Y_shape,Z_shape)),
       the 3D index (x,y,z) can be converted to 1D index: z + y* Z_shape + x * (Z_shape * Y_shape)

    5) sum up contributions into the same grid data uising TORCH.TENSOR.SCATTER_ADD_
    
    '''

    batchsize = scatter_pts.shape[0]
 
    scatter_pts = torch.permute(scatter_pts[:,:,0:3,:], (0,1,3, 2))
    scatter_pts = torch.reshape(scatter_pts,(-1,scatter_pts.shape[-1]))
    frames_flatten = frames.flatten()

    
    
    

    # only intepolete the grid points within the ground truth inteploted volume
    if volume_position != None:
        
        pts_per_seq = int(scatter_pts.shape[0]/batchsize)
       
        min_x,max_x = torch.min(volume_position[0]),torch.max(volume_position[0])
        min_y,max_y = torch.min(volume_position[1]),torch.max(volume_position[1])
        min_z,max_z = torch.min(volume_position[2]),torch.max(volume_position[2])

        # obtain the index that within the groundtruth intepoleted volume
        inside_min_x = torch.where(scatter_pts[:,0] >= min_x, 1, 0)
        inside_max_x = torch.where(scatter_pts[:,0] <= max_x, 1, 0)
        inside_min_y = torch.where(scatter_pts[:,1] >= min_y, 1, 0)
        inside_max_y = torch.where(scatter_pts[:,1] <= max_y, 1, 0)
        inside_min_z = torch.where(scatter_pts[:,2] >= min_z, 1, 0)
        inside_max_z = torch.where(scatter_pts[:,2] <= max_z, 1, 0)

        index_inside = inside_min_x * inside_max_x * inside_min_y * inside_max_y * inside_min_z * inside_max_z
        scatter_pts = scatter_pts[index_inside==1,:]
        frames_flatten = frames_flatten[index_inside==1]

        
        pts_per_batch=[]
        for i_batch in range(0,batchsize):
            pts_per_batch.append(torch.sum(index_inside[pts_per_seq*i_batch:pts_per_seq*(i_batch+1)]==1))
        
        initial = torch.zeros(pts_per_batch[0])
        for i_batch in range(1,batchsize):
            initial = torch.cat((initial,i_batch*torch.ones(pts_per_batch[i_batch])),dim=0) 
        initial = initial.to(device)

    else:  
    
        min_x,max_x = torch.min(scatter_pts[:,0]),torch.max(scatter_pts[:,0])
        min_y,max_y = torch.min(scatter_pts[:,1]),torch.max(scatter_pts[:,1])
        min_z,max_z = torch.min(scatter_pts[:,2]),torch.max(scatter_pts[:,2])

        pts_per_seq = int(scatter_pts.shape[0]/batchsize)
        initial = torch.zeros(pts_per_seq)
        for i_batch in range(1,batchsize):
            initial = torch.cat((initial,i_batch*torch.ones(pts_per_seq)),dim=0) 
        initial = initial.to(device)

        
    
    
    
    
    if volume_size == 'fixed_interval':

        # x = torch.linspace((min_x.item()), (max_x.item()), int(((max_x.item())-(min_x.item()))))
        # y = torch.linspace((min_y.item()), (max_y.item()), int(((max_y.item())-(min_y.item()))))
        # z = torch.linspace((min_z.item()), (max_z.item()), int(((max_z.item())-(min_z.item()))))
        
        x = torch.linspace((min_x), (max_x), int(((max_x)-(min_x))))
        y = torch.linspace((min_y), (max_y), int(((max_y)-(min_y))))
        z = torch.linspace((min_z), (max_z), int(((max_z)-(min_z))))
        
        X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
        X, Y, Z =X.to(device), Y.to(device), Z.to(device) 

        # number of pixels
        xsize,ysize,zsize = int(((max_x.item())-(min_x.item())))+2,int(((max_y.item())-(min_y.item())))+2,int(((max_z.item())-(min_z.item())))+2 #modify xsize as you like
        # set the spacing as 1mm, can also set a fixed dimention of the volume, then 
        # each reconstructed US volume will have the same size. Such that can batched into the NN
        # in this case, xsize = fixed_x_size, ysize = fixed_y_size, zsize = fixed_z_size,
        # where fixed_x_size,fixed_y_size,fixed_z_size can be the maxmium value that GPU can handle
        # If the input US sequence has the same length, the number of frames can be set as fixed_x_size
        
        # can be set st 2,2,2 to have a smaller size
        xstep,ystep,zstep = 1,1,1#(max_x.item()-min_x.item())/(xsize-1),(max_y.item()-min_y.item())/(ysize-1),(max_z.item()-min_z.item())/(zsize-1)
       
        # the shape of the volume should be divisiable by 16, such that this can be taken by monai.voxelmorph
        X_shape = max(X.shape[0]+(16 - X.shape[0]%16),X.shape[0]+1)
        Y_shape = max(X.shape[1]+(16 - X.shape[1]%16),X.shape[1]+1)
        Z_shape = max(X.shape[2]+(16 - X.shape[2]%16),X.shape[2]+1)
        
    elif volume_size == 'fixed_volume_size':
    # set a fixed volume size as 128*128*128
    
        volume_size = 127

        x = torch.linspace(0, volume_size, volume_size)
        y = torch.linspace(0, volume_size, volume_size)
        z = torch.linspace(0, volume_size, volume_size)
        X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
        X, Y, Z =X.to(device), Y.to(device), Z.to(device) 
        xsize,ysize,zsize = volume_size,volume_size,volume_size #modify xsize as you like
        xstep,ystep,zstep = (max_x.item()-min_x.item())/(xsize-1),(max_y.item()-min_y.item())/(ysize-1),(max_z.item()-min_z.item())/(zsize-1)
        X_shape = X.shape[0]+1
        Y_shape = X.shape[1]+1
        Z_shape = X.shape[2]+1


    # the initialised value should be pay attention to and this should be same as the background value in the US image,
    # pay attention to whether the image is normalised using (*-min)/(max-min) or (*-mean)/std
    # if the first one, as the orignal pixel value of background is 0, so no difference between backdround in the two kinds of normalisition
    # if the second one, the backgorund value would not be 0. In this case, the initialised value 0 is not acceptable
    
    

   

    # if scan_name.startswith('pred_step'):
    #     weight4pixel = Variable(torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device), requires_grad=True).clone()#torch.full(X.shape, torch.nan).to(device) #torch.from_numpy(np.asarray([[[None]*(X.shape[2])]*(X.shape[1])]*(X.shape[0]), dtype=np.float32)).to(device) #([([[None] * X.shape[2]])* X.shape[1]])* X.shape[0]
    #     intensity4pixel = Variable(torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device), requires_grad=True).clone()#torch.full(X.shape, torch.nan).to(device)
    # elif scan_name.startswith('gt_step'):
    weight4pixel = torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device)#torch.full(X.shape, torch.nan).to(device) #torch.from_numpy(np.asarray([[[None]*(X.shape[2])]*(X.shape[1])]*(X.shape[0]), dtype=np.float32)).to(device) #([([[None] * X.shape[2]])* X.shape[1]])* X.shape[0]
    intensity4pixel = torch.zeros((batchsize,X_shape,Y_shape,Z_shape)).to(device)#torch.full(X.shape, torch.nan).to(device)
    

    # this is very important. In this way, the corridinates of each point is normalised.
    # That is, the normalised coordinates is the same as the index of the points 
    x_norm = ((scatter_pts[:,0]-min_x)/xstep).to(device)
    y_norm = ((scatter_pts[:,1]-min_y)/ystep).to(device)
    z_norm = ((scatter_pts[:,2]-min_z)/zstep ).to(device) 
    

    x_low,x_up,y_low,y_up,z_low,z_up = eight_neighbour_points(x_norm,y_norm,z_norm) 
        

    # # in-efficient way to compute bilinear intepoletion - using for loop

    # time_s = time.time()
    # for i in range(gt_pts.shape[0]):
    #     # times_s_2 = time.perf_counter()
    #     weight4pixel,intensity4pixel = scatter2GridIndex(x_norm[i],y_norm[i],z_norm[i],x_low[i],x_up[i],y_low[i],y_up[i],z_low[i],z_up[i],frames_flatten[i],weight4pixel,intensity4pixel)

    # time_e = time.time()-time_s


    # efficient way to compute bilinear intepoletion using vectorlised method, together with scatter_add_ in pytorch, which is diffrenciable and can backforward
                
    time_s = time.time()

    # update the index of batched index
    
    

    

    neighbour_pts_idx = eright_points_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,X_shape,Y_shape,Z_shape,initial)
    
    

    weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts = weight_intensity_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm,frames_flatten,options = option)
    weight4pixel,intensity4pixel = combine_values(weight4pixel.flatten(),intensity4pixel.flatten(),weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts,neighbour_pts_idx)

    weight4pixel = torch.reshape(weight4pixel,(batchsize,X_shape,Y_shape,Z_shape))
    intensity4pixel = torch.reshape(intensity4pixel,(batchsize,X_shape,Y_shape,Z_shape))
    
    time_e = time.time()-time_s

    # some pixel has 0 weight or 0 intensity, check
    # weight = 0 means no points contribute to this pixel, and the intensity of this pixel should be 0
    # intensity = 0 means either no points contribute to this pixel or the contributed pixels are all have pixel 0
    nan_index = (weight4pixel == 0).nonzero() # 
    
    # convert 0 to 1, to aviod 0/0
    weight4pixel[nan_index[:,0],nan_index[:,1],nan_index[:,2],nan_index[:,3]] = 1

    volume_intensity = intensity4pixel/weight4pixel
    # volume_intensity[nan_index[:,0],nan_index[:,1],nan_index[:,2],nan_index[:,3]] = 0
    if torch.sum(torch.isnan(volume_intensity))!=0:
        raise("Intensity of pixels has %d nan values" %(volume_intensity == 0).nonzero().shape[0])


    # print(f'volume_intensity_min_max:{torch.min(volume_intensity)},{torch.max(volume_intensity)}')


    # # save time
    # with open(time_log, 'a') as time_file:
    #     print('Pytorch_tensor GPU: %.3f' % (time_e),file=time_file)
    #     print('\n')


    # # # # compute the normalised axis info for ploting - save mha for ploting
    # don't need to normalise 
    # min_x_norm,max_x_norm = torch.min(x_norm),torch.max(x_norm)
    # min_y_norm,max_y_norm = torch.min(y_norm),torch.max(y_norm)
    # min_z_norm,max_z_norm = torch.min(z_norm),torch.max(z_norm)


    # xx_norm = torch.linspace((min_x_norm), (max_x_norm), int(((max_x_norm)-(min_x_norm)).cpu().numpy()))
    # yy_norm = torch.linspace((min_y_norm), (max_y_norm), int(((max_y_norm)-(min_y_norm)).cpu().numpy()))
    # zz_norm = torch.linspace((min_z_norm), (max_z_norm), int(((max_z_norm)-(min_z_norm)).cpu().numpy()))

    # save2mha(volume_intensity.cpu().numpy(),
    #         sx=np.double(x.cpu().numpy()[1]-x.cpu().numpy()[0]),
    #         sy = np.double(y.cpu().numpy()[1]-y.cpu().numpy()[0]),
    #         sz = np.double(z.cpu().numpy()[1]-z.cpu().numpy()[0]),
    #         save_folder=saved_folder_test+'/'+scan_name+'_pytorch_GPU.mha'
    #         )

    return volume_intensity, [X,Y,Z]


def interpolation_3D_pytorch(scatter_pts,frames,time_log,saved_folder_test,scan_name,device,option):
    # interpolate 
    # given scatter points in 3D, return the grid points

    # inteplote from scatter data to grid data
    # different from typical inteplotion methods, we index all the scattered points,
    # compute the contribution to the nergbours
    '''

    The idea is to loop all scatter points, compute the contribution to the 2**3 neighbour grid points, 
    and then sum those contributions which are into the same grid points from difference scatter points
    
    intensity of grid points = (weight1*intensity1 + weight2*intensity2 + ... + weightn*intensityn)/(weight1+weight2+..._weightn)
    where n denotes the number of scatter points which has contributions into this grid point

    The core iead is to find the index of contribution scatter points for each grid point
    and sum up all the contribution for each grid point



    Two things are import, the first one is to normalise the coordinates of the scatter points to a unitless one,
    the second is to convert the interpolation problem into a math problem - sum up all the contributions into the same grid points, which can be regarded as a math problem - sum up with known index

    Steps:
    1) Normalise the coordinates of the scatter points to a unitless one, such that the coordinates of the grid points (which is the 8 neigibour grid points computed using torch.floor or torch.ceil) is also the index of this point
       Normalise method: (pts - min(pts))/ step
       where
       xsize,ysize,zsize = int(((max_x)-(min_x))),int(((max_y)-(min_y))),int(((max_z)-(min_z))) #modify xsize as you like
       xstep,ystep,zstep = (max_x-min_x)/(xsize-1),(max_y-min_y)/(ysize-1),(max_z-min_z)/(zsize-1)
        
       For exmaple, in 1D, define 1D grid {1,3,5,7,9} and scatter point A = 3.5, B = 5.6, C = 8.2
       step = (9-1)/4 = 2
       after normalisition, A = (3.5-1)/2 = 1.25, B = (5.6-1)/2 = 2.3, C = (8.2-1)/2 = 3.6
       after torch.floor, A_low = 1, B_low = 2, C_low = 3

    2）Find 8 neighbour grid points of all scatter points. NOTE: don't use torch.floor and torch.ceil simultaneously, as the distance between x_low and X_up should be eauql to 1.
       In terms of integers, torch.floor and torch.ceil will both be the integers
       Use torch.floor and torch.floor + 1 instead.

    3) For each scatter point, compute the contribution for the 8 neighbour grid points
       Can use a simple bilinear method: the weight for the grid data can be the (1-distance) to the grid data, the final weight is multiplied by each sxis
       for example, for grid points x_low, y_low, z_low =  (1-(x-x_low))(1-(y-y_low))(1-(z-z_low))
    
    4) convert the 3d array index into 1d index to use TORCH.TENSOR.SCATTER_ADD_ function

       given the volume = torch.zeros((X_shape,Y_shape,Z_shape)),
       the 3D index (x,y,z) can be converted to 1D index: z + y* Z_shape + x * (Z_shape * Y_shape)

    5) sum up contributions into the same grid data uising TORCH.TENSOR.SCATTER_ADD_
    
    '''


    min_x,max_x = torch.min(scatter_pts[:,0,:]),torch.max(scatter_pts[:,0,:])
    min_y,max_y = torch.min(scatter_pts[:,1,:]),torch.max(scatter_pts[:,1,:])
    min_z,max_z = torch.min(scatter_pts[:,2,:]),torch.max(scatter_pts[:,2,:])


    x = torch.linspace(int(min_x), int(max_x), int(((max_x)-(min_x))))
    y = torch.linspace(int(min_y), int(max_y), int(((max_y)-(min_y))))
    z = torch.linspace(int(min_z), int(max_z), int(((max_z)-(min_z))))
    X, Y, Z = torch.meshgrid(x, y, z,indexing='ij')
    X, Y, Z =X.to(device), Y.to(device), Z.to(device) 

    xsize,ysize,zsize = int(((max_x)-(min_x))),int(((max_y)-(min_y))),int(((max_z)-(min_z))) #modify xsize as you like
    xstep,ystep,zstep = (max_x-min_x)/(xsize-1),(max_y-min_y)/(ysize-1),(max_z-min_z)/(zsize-1)
        
    scatter_pts = torch.permute(scatter_pts[:,0:3,:], (0, 2, 1))
    scatter_pts = torch.reshape(scatter_pts,(-1,scatter_pts.shape[-1]))
    frames_flatten = frames.flatten()


    # the initialised value should be pay attention to and this should be same as the background value in the US image,
    # pay attention to whether the image is normalised using (*-min)/(max-min) or (*-mean)/std
    # if the first one, as the orignal pixel value of background is 0, so no difference between backdround in the two kinds of normalisition
    # if the second one, the backgorund value would not be 0. In this case, the initialised value 0 is not acceptable
    weight4pixel = torch.zeros((X.shape[0]+1,X.shape[1]+1,X.shape[2]+1)).to(device)#torch.full(X.shape, torch.nan).to(device) #torch.from_numpy(np.asarray([[[None]*(X.shape[2])]*(X.shape[1])]*(X.shape[0]), dtype=np.float32)).to(device) #([([[None] * X.shape[2]])* X.shape[1]])* X.shape[0]
    intensity4pixel = torch.zeros((X.shape[0]+1,X.shape[1]+1,X.shape[2]+1)).to(device)#torch.full(X.shape, torch.nan).to(device)

    # this is very important. In this way, the corridinates of each point is normalised.
    # That is, the normalised coordinates is the same as the index of the points 
    x_norm = ((scatter_pts[:,0]-min_x)/xstep).to(device)
    y_norm = ((scatter_pts[:,1]-min_y)/ystep).to(device)
    z_norm = ((scatter_pts[:,2]-min_z)/zstep ).to(device) 
    

    x_low,x_up,y_low,y_up,z_low,z_up = eight_neighbour_points(x_norm,y_norm,z_norm) 
        

    # # in-efficient way to compute bilinear intepoletion - using for loop

    # time_s = time.time()
    # for i in range(gt_pts.shape[0]):
    #     # times_s_2 = time.perf_counter()
    #     weight4pixel,intensity4pixel = scatter2GridIndex(x_norm[i],y_norm[i],z_norm[i],x_low[i],x_up[i],y_low[i],y_up[i],z_low[i],z_up[i],frames_flatten[i],weight4pixel,intensity4pixel)

    # time_e = time.time()-time_s


    # efficient way to compute bilinear intepoletion using vectorlised method, together with scatter_add_ in pytorch, which is diffrenciable and can backforward
                
    time_s = time.time()

    neighbour_pts_idx = eright_points_in_1d_1(x_low,x_up,y_low,y_up,z_low,z_up,X.shape[0]+1,X.shape[1]+1,X.shape[2]+1)
    weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts = weight_intensity_in_1d(x_low,x_up,y_low,y_up,z_low,z_up,x_norm,y_norm,z_norm,frames_flatten,options = option)
    weight4pixel,intensity4pixel = combine_values(weight4pixel.flatten(),intensity4pixel.flatten(),weight4pixel_8neighbour_pts,intensity4pixel_8neighbour_pts,neighbour_pts_idx)

    weight4pixel = torch.reshape(weight4pixel,(X.shape[0]+1,X.shape[1]+1,X.shape[2]+1))
    intensity4pixel = torch.reshape(intensity4pixel,(X.shape[0]+1,X.shape[1]+1,X.shape[2]+1))
    
    time_e = time.time()-time_s

    # some pixel has 0 weight or 0 intensity, check
    # weight = 0 means no points contribute to this pixel, and the intensity of this pixel should be 0
    # intensity = 0 means either no points contribute to this pixel or the contributed pixels are all have pixel 0
    nan_index = (weight4pixel == 0).nonzero() # 

    volume_intensity = intensity4pixel/weight4pixel
    volume_intensity[nan_index[:,0],nan_index[:,1],nan_index[:,2]] = 0
    if torch.sum(torch.isnan(volume_intensity))!=0:
        raise("Intensity of pixels has %d nan values" %(volume_intensity == 0).nonzero().shape[0])


    # print(f'volume_intensity_min_max:{torch.min(volume_intensity)},{torch.max(volume_intensity)}')


    # # save time
    # with open(time_log, 'a') as time_file:
    #     print('Pytorch_tensor GPU: %.3f' % (time_e),file=time_file)
    #     print('\n')


    # # # compute the normalised axis info
    # min_x_norm,max_x_norm = torch.min(x_norm),torch.max(x_norm)
    # min_y_norm,max_y_norm = torch.min(y_norm),torch.max(y_norm)
    # min_z_norm,max_z_norm = torch.min(z_norm),torch.max(z_norm)


    # xx_norm = torch.linspace((min_x_norm), (max_x_norm), int(((max_x_norm)-(min_x_norm)).cpu().numpy()))
    # yy_norm = torch.linspace((min_y_norm), (max_y_norm), int(((max_y_norm)-(min_y_norm)).cpu().numpy()))
    # zz_norm = torch.linspace((min_z_norm), (max_z_norm), int(((max_z_norm)-(min_z_norm)).cpu().numpy()))

    # save2mha(volume_intensity.cpu().numpy(),
    #         sx=np.double(x.cpu().numpy()[1]-x.cpu().numpy()[0]),
    #         sy = np.double(y.cpu().numpy()[1]-y.cpu().numpy()[0]),
    #         sz = np.double(z.cpu().numpy()[1]-z.cpu().numpy()[0]),
    #         save_folder=saved_folder_test+'/'+scan_name+'_'+option+'_pytorch_GPU.mha'
    #         )
    
    #  save2mha(volume_intensity.cpu().numpy(),
    #         sx=np.double(xx_norm.cpu().numpy()[1]-xx_norm.cpu().numpy()[0]),
    #         sy = np.double(yy_norm.cpu().numpy()[1]-yy_norm.cpu().numpy()[0]),
    #         sz = np.double(zz_norm.cpu().numpy()[1]-zz_norm.cpu().numpy()[0]),
    #         save_folder=saved_folder_test+'/'+scan_name+'_'+option+'_pytorch_GPU.mha'
    #         )

    return volume_intensity





    