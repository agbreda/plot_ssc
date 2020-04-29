import numpy as np
from matplotlib import pyplot
from matplotlib import colors as matcolors
from pickle import load


# Index value and name of vegetation types
vegetation_names = ['No Vegetation', 'Mangrove', 'Saltmarsh', 'Grassland']
#colourmap = ['#00b3b3', '#bf4040', '#ffdf80', '#40bf80']
colourmap = ['#346fa3', '#6c8061', '#daa60b', '#aade87']
ticksPosition = [0, 1, 2, 3, 4]
ticksLabels =  [''] + vegetation_names
myCmap, myNorm = matcolors.from_levels_and_colors(ticksPosition, colourmap, extend='neither')




class grid(object):
    ''' Class to store grid properties of the simulation area. Please see getGrid() function description. '''
    
    def __init__(self, X, Y, Z, loc, ref):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.loc = loc
        self.ref = ref




def createTicks(gcp):
    ''' Returns an array with values between the gravity center points (gcp). gcp must be already sorted (in any order) '''
    # First tick
    hi = 0.5 * abs(gcp[0] - gcp[1]) #hi = half interval
    
    if gcp[0] > gcp[-1]:
        tics = [gcp[0] + hi]
    else:
        tics = [gcp[0] - hi]
    
    # Inner ticks
    for i in range(gcp.shape[0] - 1):
        tics.append(0.5 * (gcp[i] + gcp[i+1]))
    
    # Last tick
    hi = 0.5 * abs(gcp[-2] - gcp[-1])
    
    if gcp[0] > gcp[-1]:
        tics.append(gcp[-1] - hi)
    else:
        tics.append(gcp[-1] + hi)
    
    return tics


def getGrid(pkl = None, k = None, txt = None, egm = None, inp = None, reversedY = True):
    ''' Open input file, pickle or txt, retrieve grid information and create a proper numpy matrix for pcolormesh
    plots and a reference arrays to correlate cells and their position in the mesh.
    
    Inputs:
        (Option 1)
        pkl |str| = name of the pickle file where egmdata is stored
        k |*| = key of the dictionary stored in the pickle which will used to access grid info
        (Option 2)
        txt |str| = hydrodynamic input file with the cells' info.
        (Option 3)
        egm |egmdata class| = variable of egmdata class itself
        (Other inputs)
        inp |list of ints| = list of cells which will be excluded from the plot (cell 1 = 0)
        
    Outputs (returned in a grid-class variable):
        X, Y |np.array((rows+1, columns+1), float)| = 2D-arrays with the X and Y coordinates, respectively, at the
            interface of grid cells
        Z |np.array((rows, columns), float)| = a nan-values 2D array to be used as base matrix to stored any spatial
            variable. X, Y and Z must be used in matplotlib.pyplot.pcolormesh only.
        loc |np.array((*,), int)| = array with the index of the cells which will be plotted. It is the total number
            of grid cell less the number of cells assigned as True in 'inp'.
        ref |np.array((loc.shape[0],2), int)| = position of each cell in the Z matrix. Example:
            be 'i' a cell and ref[i] = np.array([45,67]), than Z[ref[i][0], ref[i][1]] is used to access the value
            (or assign one) of the plotted variable for cell i, which is in located in the 45th row (y-coord) and the
            67th column (x-coord).
    Hint:
        Be P the 1D-array, with N elements (the total number of cells in the grid), storing the value of a desired
        variable. To plot it you must first pass the values in P to Z simple doing this:
            Z[ref[:,0], ref[:,1]] = P[loc]
    '''
    # Retrieving data due to input type
    if (pkl is not None) and (k is not None):
        
        with open(pkl, 'rb') as a:
            b = load(a)
            data = b[k]
        
    elif (txt is not None):
        
        a, b = getNumbers(txt)
        data = egmdata(a, b)
        data.readInfo(txt)
        
    elif (egm is not None):
        data = egm
        
    else:
        raise IOError('Invalid input.')
    
    # Creating array of desired cells
    if inp is None:
        loc = np.arange(data.NoC)
    else:
        loc = np.array([i for i in range(data.NoC) if i not in inp], dtype = np.int32)
    
    # Creating arrays with x and y values in the center of cells and the Z matrix
    xa, ya = [], []
    
    for e in loc:
        
        if data.x[e] not in xa:
            xa.append(data.x[e])
            
        if data.y[e] not in ya:
            ya.append(data.y[e])
            
    xa = np.array(sorted(xa), dtype=np.float32)
    
    if reversedY:
        ya = np.array(sorted(ya, reverse=True), dtype=np.float32)
        
    else:
        ya = np.array(sorted(ya), dtype=np.float32)
        
    Z = np.ones((ya.shape[0], xa.shape[0]), dtype=np.float32) * np.nan
    
    # Creating the relationship matrix between cells and Z grid
    ref = np.zeros((loc.shape[0], 2), dtype=np.int32)
    
    for j in range(loc.shape[0]):
        
        i = loc[j]
        xi = np.where(xa == data.x[i])[0][0]
        yi = np.where(ya == data.y[i])[0][0]
        ref[j,0], ref[j,1] = yi, xi
    
    # Creating X and Y meshgrid matrixes
    Xtics = createTicks(xa)
    Ytics = createTicks(ya)
    X, Y = np.meshgrid(Xtics, Ytics)
    
    return grid(X, Y, Z, loc, ref)


def join_figures(img_files, output_name='joined.png'):
    ''' Join images from the files in the input list into a single output image. The disposition
    of the images follows the same order found in input list.
    
    Inputs:
        img_files |2D-list| = List containg on every 'row' the image file names (str) that will
            displayed side by side. Every following row add another sequence of images below the
            previous ones. If instead the file name there is a None variable, the output image will
            left its place blank.
        output_name |str| = file name of the final mosaic.
    
    This function does not return anything
    '''
    
    # Opening and storing images
    open_img = []
    nrows = len(img_files)
    
    for row in range(nrows):
        
        if row == 0:
            ncols = len(img_files[row])
        else:
            ncols = max(ncols, len(img_files[row]))
        
        open_img.append([])
        
        for col in range(len(img_files[row])):
            
            if img_files[row][col] == None:
                
                open_img[row].append(None)
                
            else:
                
                imx = pyplot.imread(img_files[row][col])
                open_img[row].append(imx)
    
    # Defining sizes
    h = np.zeros(nrows, dtype = int)    #rows' heights
    w = np.zeros(ncols, dtype = int)    #columns' widths
    
    for row in range(nrows):
        for col in range(ncols):
            
            try:
                imx = open_img[row][col]
                
            except IndexError:
                imx = None
                open_img[row].append(None)
            
            if imx is None:
                continue
            
            h[row] = max(h[row], imx.shape[0])
            w[col] = max(w[col], imx.shape[1])
    
    ht = np.sum(h)
    wt = np.sum(w)
    
    # Joining matrices to form the mosaic
    mosaic = np.zeros((ht,wt,4))
    top = 0
    
    for row in range(nrows):
        
        left = 0
        
        for col in range(ncols):
            
            imx = open_img[row][col]
            
            if imx is None:
                
                left += w[col]
                continue
            
            # Defining rows' range within the mosaic to display the current image
            img_heigh = imx.shape[0]
            l0 = top + (h[row] - img_heigh) // 2
            lN = l0 + img_heigh
            
            # Defining columns' range within the mosaic to display the current image
            img_width = imx.shape[1]
            c0 = left + (w[col] - img_width) // 2
            cN = c0 + img_width
            
            # Adding images
            if len(imx.shape) == 2:
                
                # Grayscale image
                mosaic[l0:lN, c0:cN, 0] = imx[:,:]
                mosaic[l0:lN, c0:cN, 1] = imx[:,:]
                mosaic[l0:lN, c0:cN, 2] = imx[:,:]
                mosaic[l0:lN, c0:cN, 3] = 1.0
            
            else:
                
                if imx.shape[2] == 3:
                    
                    # RGB image
                    mosaic[l0:lN, c0:cN, 0:3] = imx[:,:,:]
                    mosaic[l0:lN, c0:cN, 3] = 1.0
                    
                else:
                    
                    # RGBA image
                    mosaic[l0:lN, c0:cN, 0:4] = imx[:,:,:]
            
            # Preparing to move to next column
            left += w[col]
        
        top += h[row]
    
    # Saving final mosaic
    pyplot.imsave(output_name, mosaic)
            
    #END


def plotProfile(axis, ed, atr, vmin=None, vmax=None, xadjust=0):
    ''' Create a line plot of a given variable (object attribute) for a default/given set of cells. It uses the
    matplotlib.pyplot.plot and the attribute data from the egmdata-class variable.
    
    Inputs:
        axis |matplotlib.pyplot.axis object| = axis where the pcolormesh will be draw
        ed |egmdata object| = variable with the plotting attribute 'self.<atr>'. See below
        atr |str| = attribute of 'ed' to be plotted. The function getattr(ed,atr) is used to retrieve the data.
        vmin, vmax |float| = minimum and maximum values for y-axis. If is None it lets matplotlib autocalculate.
        xadjust |float| = add a constant value to the x-coordinate
    
    Outputs:
        same 'axis' variable after the plots are assigned to it
    '''    
    # Default sequences of cells, s, and the labels for the legend, l
    s = [np.arange(926,990), np.arange(662,726), np.arange(332,396), np.arange(2,66)]
    l = [' 10 m', ' 50 m', '100 m', '150 m']
    c = ['#ff1e00', '#ffb49d', '#22a8be', '#36fa0e', '#df0095']
    
    # Distance from the tidal-input-channel
    x = [ed.x[c] + xadjust for c in s]
    
    # Profile's variable
    y = getattr(ed, atr)
    
    # Plotting accretion series
    for i in range(len(s)):
        axis.plot(x[i], y[s[i]], color=c[i], label=l[i])
    
    # Setting y-axis limits, and turning on legend and grid
    axis.set_ylim(vmin, vmax)
    axis.legend()
    axis.grid()
    
    return axis


def plotVegMap(axis, ed, gd, colorbar=True):
    ''' Create a map of vegetation categories. It uses the matplotlib.pyplot.pcolormseh, the vegetation data (see
    the vegetation function above) in the egmdata-class variable, and grid properties in the grid-class variable.
    
    Inputs:
        axis |matplotlib.pyplot.axis object| = axis where the pcolormesh will be draw
        ed |egmdata object| = variable with properties self.V for vegetation (check vegetation function above)
        gd |grid object|    = variable with grid properties for pcolormesh plot
        colorbar |bool|     = add a vegetation categories colorbar if True
    
    Ouputs:
        axis = after the plot (and colorbars) assigned to it
        cb |matplotlib.pyplot.colorbar| = colorbar of vegetation categories (returned only if colorbar is True)        
    '''
    # Treating vegetation data for the plot
    for i in range(ed.NoC):
        if ed.V[i] == 3:
            ed.V[i] = 1
        elif ed.V[i] == 1:
            ed.V[i] = 3
    
    gd.Z[gd.ref[:,0], gd.ref[:,1]] = ed.V[gd.loc] + 0.5
    gd.Z[:,0:2] = np.nan
    gd.Z[15,2:] = np.nan
    
    # Plot
    Vmap = axis.pcolormesh(gd.X, gd.Y, gd.Z, cmap=myCmap, norm=myNorm)
    
    # Colorbar, or not
    if colorbar:
        
        Vcbar = pyplot.colorbar(Vmap, ax=axis, orientation='horizontal')
        Vcbar.set_ticks(ticksPosition)
        Vcbar.ax.set_xticklabels(ticksLabels, horizontalalignment='right')
        Vcbar.set_label('Vegetation Type', weight='bold')        
        return Vmap, Vcbar
    
    else:
        
        return Vmap


def plotVegMap2(axis, data, gd, channel=True, adjust_x = 0, adjust_y = 0, xlabel = "default", ylabel = "default"):
    ''' Create a map of vegetation categories. It uses the matplotlib.pyplot.pcolormseh, the vegetation data (see
    the vegetation function above) in the egmdata-class variable, and grid properties in the grid-class variable.
    
    Inputs:
        axis |matplotlib.pyplot.axis object| = axis where the pcolormesh will be draw
        data |np.array((NoC,), int)| = values of vegetation type
        gd |grid object| = variable with grid properties for pcolormesh plot
        channel |bool| = True if the central row is a channel
        adjust_x |float| = value added to gd.X if needed
        adjust_y |float| = value added to gd.Y usually to centralize the grid in the central row
    
    Return:
        Vmap |matplotlib.pyplot.pcolormesh object| = object returned from the pcolormesh function   
    '''
    # Swap numbers between mangrove (3 -> 1) and grassland (1 -> 3)
    a = np.where(data == 3)
    b = np.where(data == 1)
    data[a] = 1
    data[b] = 3
    
    # Populate grid.Z and add 0.5 to put each vegetation in the middle of classification bar
    Z = np.copy(gd.Z)
    Z[gd.ref[:,0], gd.ref[:,1]] = data[gd.loc] + 0.5
    
    # Leep the central row blank if it represents a channel
    if channel: Z[15,:] = np.nan
    
    # Plot
    Vmap = axis.pcolormesh(gd.X + adjust_x, gd.Y + adjust_y, Z, cmap=myCmap, norm=myNorm)
    
    # Axis options
    axis.set_yticks([-150,-100,-50,0,50,100,150])

    if xlabel == "default":
        axis.set_xlabel('Distance from tide input edge (m)')
    elif xlabel == None:
        pass
    else:
        axis.set_xlabel(xlabel)

    if ylabel == "default":
        axis.set_ylabel('Cross section (m)')
    elif ylabel == None:
        pass
    else:
        axis.set_ylabel(ylabel)
    
    axis.grid()
    
    return Vmap
    

def plotBPmap(axis, ed, gd, colorbar='both', vmin=(None,None), vmax=(None,None)):
    ''' Create a map of biomass production (BP) in mangrove and saltmarsh areas. It uses the matplotlib.pyploy.pcolormesh,
    BP data in egmdata-class variable and grid properties from a grid-class variable.

    Inputs:
        axis |matplotlib.pyplot.axis object| = axis where the pcolormesh will be draw
        ed |egmdata object| = variable with properties self.V for vegetation and self.BP for biomass production
        gd |grid object|    = variable with grid properties for pcolormesh plot
        colorbar <optional> |str or None| = 'both', 'mangrove', 'saltmarsh' or None. Choose which colorbars will be created
        vmin |tuple with 2 float or None elements| = minimum value for mangrove, vmin[0], and saltmarsh, vmin[1], in the pcolormesh
            plots and colorbars. None = matplotlib decide
        vmax |tuple with 2 float or None elements| = similar to vmin, but regarding the maximum value.
        
    Outputs:    
        axis = after the plot and colorbars assigned to it
        cbm |matplotlib.pyplot.colorbar object| = colorbar for biomass production in mangrove area. colorbar need to be
            'both' or 'mangrove'
        cbs |matplotlib.pyplot.colorbar object| = colorbar for biomass production in saltmarsh area. colorbar need to be
            'both' or 'saltmarsh'
        
        OBS: if colorbar is 'both', then the colorbar objects are returned in a single tuple (cbm, cbs)
    '''
    axis.set_facecolor('#cccccc')
    
    # Plot mangrove biomass production
    bp = np.copy(gd.Z)
    caps = {'vmin':vmin[0], 'vmax':vmax[0]}
    
    for i, c in enumerate(gd.loc):
        if ed.V[c] == 3:
            bp[gd.ref[i,0], gd.ref[i,1]] = ed.BP[c]
    
    pm = axis.pcolormesh(gd.X, gd.Y, bp, cmap="Greens", **caps)
    
    if colorbar in ['both', 'mangrove']:
        cbm = pyplot.colorbar(pm, ax=axis)
        cbm.set_label('Mangrove biomass production (g/m$^2$)')
    
    # Plot saltmarsh biomass production
    bp = np.copy(gd.Z)
    caps = {'vmin':vmin[1], 'vmax':vmax[1]}
    
    for i, c in enumerate(gd.loc):
        if ed.V[c] == 2:
            bp[gd.ref[i,0], gd.ref[i,1]] = ed.BP[c]
    
    ps = axis.pcolormesh(gd.X, gd.Y, bp, cmap="Oranges", **caps)
    
    if colorbar in ['both', 'saltmarsh']:
        cbs = pyplot.colorbar(ps, ax=axis)
        cbs.set_label('Saltmarsh biomass production (g/m$^2$)')
    
    # Returning
    if colorbar == 'both':
        return axis, (cbm, cbs)
    elif colorbar == 'mangrove':
        return axis, cbm
    elif colorbar == 'saltmarsh':
        return axis, cbs
    elif colorbar == None:
        return axis
    else:
        raise ValueError('colorbar have a inapropriate value')


def predominantVegetation(V, nrows, ncols, skip_rows=[], skip_columns=[]):
    ''' Compute the predominant vegetation type on each column of a 2D tidal-flat. Assumes that vegetation type values are:
        0 = no vegetation; 1 = grassland; 2 = saltmarsh; 3 = mangrove
    
    Inputs:
        V |np.array((NoC,), int)| = Array of vegetation-type values in NoC number of cells
        nrows |int| = number of rows/lines in the 2D domain
        ncols |int| = number of columns in the 2D domain (note that nrows * ncols == NoC)
        skip_rows |list or np.array((:,), int)| = list or array with the indexes of rows to not be considered in the computation
        skip_columns |list or np.array((:,), int)| = list or array with the indexes of columns that will be excluded in the
            computation
    
    Return:
        pdmV |np.array((ncols - len(skip_columns),), float)| = array with the predominant vegetation type in each column
            considered in the computation
    '''
    pdmV = []
    
    # Computing predominant vegetation        
    for col in range(ncols):
        
        if col in skip_columns:
            continue
        
        pdmV.append(0)
            
        # Counting the number of occurrences for each vegetation type
        veg_count = [0, 0, 0, 0]
            
        for row in range(nrows):
            
            if row in skip_rows:
                continue
            
            c = row*ncols + col
            veg_count[V[c]] += 1
            
        # Checking the vegetation type with the highest count
        highest = veg_count[0]
        
        for veg_type in range(1,4):
            
            if veg_count[veg_type] >= highest:
                
                pdmV[-1] = veg_type
                highest = veg_count[veg_type]
    
    return np.array(pdmV, dtype=np.float16)


def smoothVegetation(v, window=(-1,0,1), nrev=2, overlap=True):
    ''' This function aims to smooth the vegetation distribution over time by appling a filter of
    predominant vegetation followed by overlapping rules.
    
    Input:
        v |np.array((ny,nx))| = predominant vegetation matrix for ny-years and nx-columns
        window |tuple| = window of time-steps where will be computed the predominant vegetation
        nrev |int| = number of re-applications of the procedure
        
    Return:
        vr |v| = overlaped vegetation matrix
    '''
    for rev in range(nrev):
        
        vr = np.zeros(v.shape)
        
        for i in range(v.shape[0]):
            
            for j in range(v.shape[1]):
                
                count = np.zeros(4, dtype=int)
                vr[i,j] = v[i,j]    # initialise with current vegetation
                
                for k in window:
                    
                    l = i + k
                    
                    if l < 0 or l >= v.shape[0]:
                        continue
                    
                    count[int(v[l,j])] += 1
                
                m = np.max(count)
                w = np.where(count == m)[0]
                
                if m.size == 1:    #there is just one predominant vegetation
                    vr[i,j] = w[0]
                    
                #Observing overlapping rules
                if overlap:
                    
                    #mangrove overlap no-vegetation
                    if (vr[i,j] == 3 and v[i,j] == 0) or (vr[i,j] == 0 and v[i,j] == 3):
                        vr[i,j] = 3
                                        
                    #saltmarsh overlap mangrove
                    if (vr[i,j] == 2 and v[i,j] == 3) or (vr[i,j] == 3 and v[i,j] == 2):
                        vr[i,j] = 2
                    
                    #freswater vegetation overlap saltmarsh
                    if (vr[i,j] == 1 and v[i,j] == 2) or (vr[i,j] == 2 and v[i,j] == 1):
                        vr[i,j] = 1
        
        v[:,:] = vr[:,:]
    
    return vr


def addVegetationBar(Vmap, axis, cb_aspect=20, tl_fontsize=10, putTitle=True):
    ''' Add the vegetation class colorbar related to the pcolormesh plot.
    
    Inputs:
        Vmap |matplotlib.pyplot.pcolormesh object| = object returned from the pcolormesh function
        axis |matplotlib.pyplot.axis object or np.array of axes objects| = axis/axes from where the colorbar will take space
            to be draw
        cb_aspect |int| = aspect of the colorbar, 20 is matplotlib default, higher make it thinner
        tl_fontsize |int| = font size of the vegetation names in the colorbar
        putTitle |bool| = True if will set the title "Vegetation Type"
        
    Return:
        Vcbar |matplotlib.pyplot.colorbar| = colorbar of vegetation categories
    '''
    Vcbar = pyplot.colorbar(Vmap, ax=axis, orientation='horizontal', aspect=cb_aspect)
    Vcbar.set_ticks(ticksPosition)
    Vcbar.ax.set_xticklabels(ticksLabels, horizontalalignment='right', fontsize=tl_fontsize)
    if putTitle:
        Vcbar.set_label('Vegetation Type', weight='bold')
    
    return Vcbar


def plotVegTime(axis, ed, predVegArgs, adjust_x=0, years=None, colorbar=True, textArgs=None):
    ''' Plot the predominant vegetation along the longitudinal axis (main flow direction) for a range of years.
    It uses the matplotlib.pyploy.pcolormesh, V data in egmdata-class variable, a range of years (or from the egmdata variable
    or from the input argument), and the predominantVegetation function with its arguments in kargs.
    
    Inputs:
        axis |matplotlib.pyplot.axis object| = axis where the pcolormesh will be draw
        ed |egmdata object| = variable with properties self.V for vegetation and self.BP for biomass production
        predVegArgs |dict| = dictionary with arguments to call predominantVegetation(**predVegArgs)
        years |list| = list of years to be plotted in the Y-axis. If not provided, use ed.years instead
        colorbar |boolean| = True to insert the colorbar or False to not do it
        textArgs |dict| = dictionary with arguments to write text in the chart (always in)
        
    Return:
        vtMap |matplotlib.pyplot.pcolormesh object| = object returned from the pcolormesh function
        vtBar |matplotlib.pyplot.colorbar| = colorbar of vegetation categories (returned only if colorbar is True)
    '''
    
    # Create y ticks (ticks between each year)
    if years is None:
        
        years = ed['years'][:]
    
    nYears, y = len(years), []
    
    for i in range(nYears):
        
        if i == 0:            
            d = (years[i+1] - years[i]) / 2
            y.append(years[i] - d)
                        
        elif i == (nYears - 1):            
            d = (years[i] - years[i-1]) / 2
            
        else:            
            d = (years[i+1] - years[i]) / 2
        
        y.append(years[i] + d)
                
    # Create x ticks (10 m cell size)
    x = []
    
    for col in range(predVegArgs['ncols']):
        
        if col in predVegArgs['skip_columns']:
            continue
        
        if len(x) == 0:            
            x.append(col*10)
            
        x.append((col+1)*10)
        
    # Create grid
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    x += adjust_x
    xg, yg = np.meshgrid(x,y)
        
    # Create empty data-matrix
    data = np.zeros((y.size - 1, x.size - 1), dtype=np.float32)
    
    # Fill data-matrix
    for i in range(nYears):
        
        y = years[i]
        data[i,:] = predominantVegetation(ed[y].V, **predVegArgs)
    
    # Swap numbers between mangrove (3 -> 1) and grassland (1 -> 3)
    a = np.where(data == 3)
    b = np.where(data == 1)
    data[a] = 1
    data[b] = 3
    data += 0.5    #add 0.5 to put each vegetation in the middle of classification bar
    
    # Plot the pcolormesh
    vtMap = axis.pcolormesh(xg, yg, data, cmap=myCmap, norm=myNorm)
    
    # Axis labels and grid
    axis.set_xlabel("Distance from tide input channel (m)")
    axis.set_ylabel("Year")
    axis.grid()
    
    # Text in the chart
    if textArgs != None:
        
        if 'kwargs' in textArgs.keys():    # extra arguments to format the text
            axis.text(textArgs['x'], textArgs['y'], textArgs['label'], transform=axis.transAxes, **textArgs['kwargs'])
            
        else:            
            axis.text(textArgs['x'], textArgs['y'], textArgs['label'], transform=axis.transAxes)
        
    # Colorbar, or not
    if colorbar:        
        vtBar = addVegetationBar(vtMap, axis)
        return vtMap, vtBar
    
    else:        
        return vtMap


def accumulateAccretion(edict, yref, y0=0):
    ''' Sum the accretion over time until a given year.
    
    Inputs:
        edict |dictionary| = dictionary created in a EGM simulation with some keys storing the simulation setup and,
            while keys of each simulated year stores a egmdata object, included the accretion in the step interval
        yref |int| = last year of the accumulation period
        y0 |int| = first year of the accumulation period
        
    Return:
        aA |np.array((NoC,) float)| = array with accumulated accretion, in mm, for each cell between y0 and yref
    '''
    aA = np.zeros(edict[yref].NoC, dtype=np.float32)
    
    for y in edict['years']:
        
        if y0 <= y < yref:    #it need to be "< yref" because with eco_step is > 1, it will accumulate elevation
                              #from a longer period
            aA[:] += edict[y].A[:]
    
    return aA * 1000


def extendArray(a, n, t):
    ''' Extend array 'a' of type 't' to 'n' size filling the extended dimension with zeros
    
    Inputs:
        a |np.array| = input array that will be extended
        n |int| = new array size (n > a.size)
        t |str| = type of new array
        
    Return:
        b |np.array((n,), dtype=t)| = new array with the same content of 'a' until a.size and zero afterwards
    '''
    if t == 'int16':
        b = np.zeros(n, dtype=np.int16)
    elif t == 'int32':
        b = np.zeros(n, dtype=np.int32)
    elif t == 'float32':
        b = np.zeros(n, dtype=np.float32)
    elif t == 'float64':
        b = np.zeros(n, dtype=np.float64)
    else:
        print('\n ERROR\n Could not recognize array type:', t, '\n')
        exit()
    
    b[0:a.size] = a[:]
    return b


def reviewEGM(ed):    
    ''' Apply the mirrorEGM function to all egm-data found in ed. '''
    
    if "years" in ed:
        
        for yr in ed["years"]:
            
            ed[yr] = mirrorEGM(ed[yr])
        
        tide = ed['tiles']['TIDE']
        extra = [row*68 + col + 1 for col in range(4) for row in range(15,31)]
        ed['tiles']['TIDE'] = (tide[0], tuple(list(tide[1]) + extra))
            
    else:
        
        ed = mirrorEGM(ed)
        
    return ed


def mirrorEGM(ed):
    
    ed.NoC += 15*68
    
    ed.number = extendArray(ed.number, ed.NoC, 'int32')
    ed.border = extendArray(ed.border, ed.NoC, 'int32')
    ed.x      = extendArray(ed.x, ed.NoC, 'float64')
    ed.y      = extendArray(ed.y, ed.NoC, 'float64')
    ed.z      = extendArray(ed.z, ed.NoC, 'float64')
    ed.dx     = extendArray(ed.dx, ed.NoC, 'float64')
    ed.dy     = extendArray(ed.dy, ed.NoC, 'float64')
    ed.botwid = extendArray(ed.botwid, ed.NoC, 'float64')
    ed.latslp = extendArray(ed.latslp, ed.NoC, 'float64')
    ed.chndep = extendArray(ed.chndep, ed.NoC, 'float64')
    
    ed.V = extendArray(ed.V, ed.NoC, 'int32')
    ed.H = extendArray(ed.H, ed.NoC, 'float64')
    ed.D = extendArray(ed.D, ed.NoC, 'float64')
    ed.C = extendArray(ed.C, ed.NoC, 'float64')
    ed.A = extendArray(ed.A, ed.NoC, 'float64')
    
    for row in range(15):
        for col in range(68):
            
            #north cell (existing one that will be copied)
            nc = (14 - row) * 68 + col
            
            #south cell (new one that will be created)
            sc = (16 + row) * 68 + col
            
            #center cell (for reference)
            cc = 15*68 + col
            
            #mirroring domain properties (cells only)
            ed.number[sc] = sc + 1
            ed.border[sc] = ed.border[nc]
            ed.x[sc] = ed.x[nc]
            ed.y[sc] = ed.y[cc] + (ed.y[cc] - ed.y[nc])
            ed.z[sc] = ed.z[nc]
            ed.dx[sc] = ed.dx[nc]
            ed.dy[sc] = ed.dy[nc]
            ed.botwid[sc] = ed.botwid[nc]
            ed.latslp[sc] = ed.latslp[nc]
            ed.chndep[sc] = ed.chndep[nc]
            
            #mirroring EGM data
            ed.V[sc] = ed.V[nc]
            ed.H[sc] = ed.H[nc]
            ed.D[sc] = ed.D[nc]
            ed.C[sc] = ed.C[nc]
            ed.A[sc] = ed.A[nc]
    
    return ed


def prepareProfileData1(ed, data):
    ''' Generate the arguments of plotProfile for a typical plot in the hypothetical 31x66 cells tidal flat
    used in the eco-geomorphologic simulations from Angelo's thesis.
    
    Inputs:
        ed |egmdata object| = variable from egmdata class created in a EGM simulation
        data |np.array((N,), float)| = array of plottable data with size N
    
    Return:
        dictionary containing the following variables (keys follows plotProfile argument names)
        x |list of np.array((:,), float)| = list of arrays with the distances along the profile
        y |list of np.array((:,), float)| = list of arrays with values along the profile
        labels |list of strings| = list of profile labels
    '''
    
    # Default plotting profiles
    dpp = [
        {'seq': np.arange(926,990), 'name': ' 10 m'},
        {'seq': np.arange(662,726), 'name': ' 50 m'},
        {'seq': np.arange(332,396), 'name': '100 m'},
        {'seq': np.arange(2,66), 'name': '150 m'}
    ]
    
    # Distances along each profile
    x = [ed.x[a['seq']] - 20.0 for a in dpp]
    
    # Values along each profile
    y = [data[a['seq']] for a in dpp]
    
    # Profile labels
    l = [a['name'] for a in dpp]
    
    return {'x': x, 'y': y, 'labels': l}

            
def plotProfiles1(ax, x, y, labels, yLabel, minVal, maxVal):
    
    minX, maxX = 9.e9, -9.e9
    c = ['#ff1e00', '#ffb49d', '#0033cc', '#36fa0e', '#df0095']
    
    # Plotting given data
    for i in range(len(x)):

        ax.plot(x[i], y[i], label = labels[i], color=c[i], clip_on = False)
        minX = min(minX, np.nanmin(x[i]))
        maxX = max(maxX, np.nanmax(x[i]))
    
    # X-axis limits
    ax.set_xlim(minX, maxX)
    
    # Y-axis limits and label
    ax.set_ylim(minVal, maxVal)
    ax.set_ylabel(yLabel)
    
    # Show legend and grid
    #ax.legend()
    ax.grid()


def prepareProfileData2(ed, data, dpp=None):
        
    # Default plotting profiles
    if dpp == None:
        dpp = [
            {'seq': np.arange(2,66),      'name': ' 150 m'},
            {'seq': np.arange(332,396),   'name': ' 100 m'},
            {'seq': np.arange(662,726),   'name': '  50 m'},
            {'seq': np.arange(992,1056),  'name': '   0 m'},
            {'seq': np.arange(1322,1386), 'name': ' -50 m'},
            {'seq': np.arange(1652,1716), 'name': '-100 m'},
            {'seq': np.arange(1982,2046), 'name': '-150 m'}
        ]
    
    # Distances along each profile
    x = [ed.x[a['seq']] - 20.0 for a in dpp]
    
    # Values along each profile
    y = [data[a['seq']] for a in dpp]
    
    # Profile labels
    l = [a['name'] for a in dpp]
    
    return {'x': x, 'y': y, 'labels': l}


def prepareProfileData3(ed, data):
    # Similar to ...Data2 but it shows the difference from the central row
        
    # Default plotting profiles
    dpp = [
        {'seq': np.arange(2,66),      'name': ' 150 m'},
        {'seq': np.arange(332,396),   'name': ' 100 m'},
        {'seq': np.arange(662,726),   'name': '  50 m'},
        {'seq': np.arange(992,1056),  'name': '   0 m'},
        {'seq': np.arange(1322,1386), 'name': ' -50 m'},
        {'seq': np.arange(1652,1716), 'name': '-100 m'},
        {'seq': np.arange(1982,2046), 'name': '-150 m'}
    ]
    
    # Distances along each profile
    x = [ed.x[a['seq']] - 20.0 for a in dpp]
    
    # Values along each profile
    y = [data[a['seq']] for a in dpp]

    for i in range(len(dpp)):
        if i == 3: continue
        for j in range(len(y[i])):
            y[i][j] -= y[3][j]
    y[3] = [0.0 for a in range(len(y[3]))]
    
    # Profile labels
    l = [a['name'] for a in dpp]
    
    return {'x': x, 'y': y, 'labels': l}


def plotProfiles2(ax, x, y, labels, yLabel, minVal, maxVal):
    
    minX, maxX = 9.e9, -9.e9
    c = ['#ccccff', '#4d4dff', '#0000cc', '#a6a6a6', '#cc0000', '#ff4d4d', '#ffcccc']
    
    # Plotting given data
    for i in range(len(x)):

        ax.plot(x[i], y[i], label = labels[i], color=c[i], clip_on = False)
        minX = min(minX, np.nanmin(x[i]))
        maxX = max(maxX, np.nanmax(x[i]))
    
    # X-axis limits
    ax.set_xlim(minX, maxX)
    
    # Y-axis limits and label
    ax.set_ylim(minVal, maxVal)
    ax.set_ylabel(yLabel)
    
    # Show legend and grid
    #ax.legend()
    ax.grid()
