# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import pyplot, colors, cm
from pickle import load
from copy import deepcopy
from datetime import datetime, timedelta
from sys import path
path.append('C:/Wetlands/Repository')
from egm_lib64 import *
from egm_plot import *
pyplot.rcParams.update({'font.size':12})
run_id = 'runs_c111/flat2'
year = '2000'

# Reading water-depth data file
with open('./'+run_id+'/depth_'+year+'.pkl', 'rb') as f:
    ts, h = load(f)

# Reading ssc data file
with open('./'+run_id+'/ssc_'+year+'.pkl', 'rb') as f:
    ts, ssc = load(f)

# Reading EGM data
pkl_file = './'+run_id+'/egm_data.pkl'
with open(pkl_file, 'rb') as f:
    data = load(f)
    ed = data[int(year)]

# Initialising grid
xa, ya = [], []
for c in range(ed.NoC):
    if ed.x[c] not in xa:
        xa.append(ed.x[c])
    if ed.y[c] not in ya:
        ya.append(ed.y[c])
xa = np.array(sorted(xa), dtype=float)
ya = np.array(sorted(ya, reverse=True), dtype=float)

X, Y = np.meshgrid(xa, ya)
Z = np.zeros(X.shape) * np.nan    #water level
W = np.zeros(X.shape) * np.nan    #sediment concentration
    
# Get pcolormesh grid just to use the location->reference to map cells in the 2D-domain
g = getGrid(egm = ed)

# Fourth dimension colormap
norm = colors.Normalize(0, 111)
m = pyplot.cm.ScalarMappable(norm=norm, cmap='inferno_r')

# Picking the last tidal cycle (12h-long, 600s time-steps)
i0 = ssc.shape[0] - int(12*3600/600)
iN = ssc.shape[0]
for i in range(i0+10, iN-9):    
    
    # Water Level
    Z[g.ref[:,0], g.ref[:,1]] = h[i,g.loc] + ed.z[g.loc]
    
    # Sediment concentration
    W[g.ref[:,0], g.ref[:,1]] = ssc[i,g.loc]
    norm = colors.Normalize(vmin=0, vmax=111)
    m = cm.ScalarMappable(cmap=cm.inferno_r, norm=norm)
    m.set_array([])
#    fcolors = m.to_rgba(W)
    
    # Figure title and file name
    timestamp = str('Time = %.1f h' % ((i - i0) * 600 / 3600))
    figname = './figures/sediment_map/' + run_id.replace('/','_') + str('-%6.6i.png' % i)
    
    fig = pyplot.figure()
    ax = fig.gca(projection='3d')
#    axMap = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=fcolors, vmin=0, vmax=111, shade=False)
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, facecolors=pyplot.cm.inferno_r(norm(W)), vmin=0, vmax=111, shade=False)
    
    cbar = pyplot.colorbar(m, ax=ax)
    cbar.ax.set_ylabel('Sediment Concentration (g/m$^3$)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_zlim(0.1,1.3)
    ax.view_init(elev=25., azim=245)
    
    ax.set_title(timestamp)
    fig.tight_layout()
    pyplot.savefig(figname)
    pyplot.close(fig)
    print( str('%-40s' % figname), '... ok!')

# To create an animation, run the following in a cmd prompt
#>magick .\figures\sediment_map\*.png -delay 20 -loop 0 sedlevel.gif