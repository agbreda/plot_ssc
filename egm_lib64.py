import numpy as np
from os import system
from pickle import dump, load
from copy import deepcopy
from datetime import datetime
from math import isclose



# Classes
#-----------------------------------------------------------------------------------------------------------------------
class egmdata(object):
    ''' Class to centralize arrays of data of the current simulation. '''
    
    def __init__(self, noc, nol):
        self.NoC = noc
        self.NoL = nol
        self.cell1  = np.zeros(nol, dtype = np.int32)
        self.cell2  = np.zeros(nol, dtype = np.int32)
        self.lktype = np.zeros(nol, dtype = np.int32)
        self.number = np.zeros(noc, dtype = np.int32)
        self.border = np.zeros(noc, dtype = np.int32)
        self.x      = np.zeros(noc, dtype=np.float64)
        self.y      = np.zeros(noc, dtype=np.float64)
        self.z      = np.zeros(noc, dtype=np.float64)
        self.dx     = np.zeros(noc, dtype=np.float64)
        self.dy     = np.zeros(noc, dtype=np.float64)
        self.botwid = np.zeros(noc, dtype=np.float64)
        self.latslp = np.zeros(noc, dtype=np.float64)
        self.chndep = np.zeros(noc, dtype=np.float64)
        self.params = np.zeros((nol,12), dtype=np.float64)
    
    def readInfo(self, vincul_file):
        
        with open(vincul_file, 'r') as archive:
            fct = archive.readlines()
        
        #Links-type info
        for i in range(self.NoL):
            self.cell1[i], self.cell2[i], self.lktype[i] = map(int, fct[i+2].split())
        
        #Cells data
        for i in range(self.NoC):
            aux = fct[i+self.NoL+2].split()
            self.number[i], self.border[i] = int(aux[0]), int(aux[1])
            self.z[i], self.x[i], self.y[i], self.dx[i], self.dy[i] = map(float, aux[2:7])
            self.botwid[i], self.latslp[i], self.chndep[i] = map(float, aux[7:10])
    
    def readParam(self, param_file):
        
        with open(param_file, 'r') as archive:
            fct = archive.readlines()
        
        #Parameters data    
        for i in range(self.NoL):        
            aux = fct[i].split()
            
            if int(aux[0]) != self.cell1[i] or int(aux[1]) != self.cell2[i]:            
                print('PARAM.DAT, LINE', i+1, ':', aux)
                print('VINCUL.DAT, LINE', i+3,':', [self.cell1[i], self.cell2[i], self.lktype[i]])
                raise ValueError('Cells number are different for the same link.')
            
            for j in range(len(aux)-2):
                self.params[i,j] = float(aux[j+2])


class STM(object):

    def __init__(self, Ucd, ws, ssc_in):
        ''' Fixed parameters        
        Ucd |float| is the critical velocity (m/s)
        ws  |float| is the fall velocity (m/s)
        ssc_in |np.array((:,), float)| is the sediment concentration at input cells (g/m3)
        
        Note that ssc_in is an 1d-array without an expected dimension size. If it holds just one value, this one
        will be repeated throught all the time steps. If two or more values, than the initial time step assume the
        value at ssc_in[0], the second time step the value at ssc_in[1], and so on. If there are more steps than
        ssc_in values, the array is repeated until the reach the end of the simulation.
        '''
        
        self.Ucd = Ucd
        self.ws  = ws
        self.ssc_in = ssc_in
        self.tolerance = 0.1    #Error Tolerance between two iteration of the implicit solution

    def set_domain_properties(self, egm):
        # egm is a egmdata class object
        self.nbcells, self.links, self.signal = relationships(egm)
        
        # Identifing horizontal and vertical links
        same_x = [[] for i in range(egm.NoC)]
        same_y = [[] for i in range(egm.NoC)]
        
        for i in range(egm.NoC):
            
            xi, yi = egm.x[i], egm.y[i]
            
            for j in range(self.nbcells[i].size):
                
                nc = self.nbcells[i][j]
                xj, yj = egm.x[nc], egm.y[nc]
                
                if xi == xj:    #vertical links
                    same_x[i].append(self.links[i][j])
                    
                elif yi == yj:    #horizontal links
                    same_y[i].append(self.links[i][j])
                    
                else:
                    print('\n ERROR!\n Cells', i+1, 'and', nc+1, 'are not alligned!\n')
                    exit()
                    
            same_x[i] = np.array(same_x[i], dtype=np.int32)
            same_y[i] = np.array(same_y[i], dtype=np.int32)
            
        self.same_x = same_x[:]
        self.same_y = same_y[:]

    def set_input_cells(self, lic):
        # lic is a numpy boolean array
        self.input_cells = lic

    def model(self, delta_t, H, V, Q, egm, prevSSC=None, returnSettled=False, informRun=True):
        ''' Sediment Transport Model
        Compute the suspended solid concentration in each cell of the domain along the simulation period.

        Inputs:
        delta_t |float| = time step length (seconds)
        H |np.array((NT,NoC), float)| = matrix of water depths from the hydrodynamic model (metres). NT is the number
            of time steps of this model and NoC is the number of cells.
        V, Q |np.array((NT,NoL), float)| = matrix of water velocity (m/s) and discharge (m3/s) from the hydrodynamic
            model. NoL is the number of links.
        egm |egmdata()| = variable from egmdata class with all properties already given
        prevSSC |np.array((NoC), float)| = array with initial values of SSC. If it is None, set the concentration
            to 0 in the entire domain.
        returnSettled |boolean| = define if amount of settled sediment will be computed and returned
        informRun |boolean| = define if run statistics will be printed

        Outputs:
        C |np.array((NT,NoC), float)| = matrix of sediment (suspended solid) concentration (g/m3).
        settled |np.array((NT,NoC), float)| = settled volume of sediments matrix (g/m2). Returned only if
            returnSettled is True.
        '''
        info = [[], []]

        # Constants
        gama = delta_t / (egm.dx * egm.dy)    # = time step [s] / area [m^2]
        
        # Initializing sediment concentration matrix, C
        C, tc = np.zeros(H.shape, dtype=np.float64), 0
        if prevSSC is not None:
            C[0,:] = prevSSC[:]
        C[0,self.input_cells] = self.ssc_in[tc]
        if returnSettled:
            settled = np.zeros(H.shape, dtype=np.float64)    #settled sediment [g/m2]

        # Auxiliar matrices of sediment concentration of the implicit method
        Cst, Cnd = np.zeros(egm.NoC, dtype=np.float64), np.zeros(egm.NoC, dtype=np.float64)

        # Other matrices of the model
        u = np.zeros(egm.NoC, dtype=np.float64)       #average water velocity [m/s]
        Pd = np.zeros(egm.NoC, dtype=np.float64)      #probability of deposition [adim.]
        den = np.zeros(egm.NoC, dtype=np.float64) * np.nan    # denominator of the formula to find C_(i)^(t+Dt)
#        gama = np.zeros(egm.NoC, dtype=np.float64)    #time step [s] / area [m^2]


        # COMPUTING SEDIMENT CONCENTRATION OVER TIME
        #---------------------------------------------------------------------------------------------------------------
        for t in range(1, H.shape[0]):

            #Set initial guess of future concentration as the values in the past time step
            Cst[:] = C[t-1,:]
            Cnd[:] = C[t-1,:]
            
            #Concentration in the input cells are known
            tc += 1
            if tc == self.ssc_in.shape[0]: tc = 0            
            Cst[self.input_cells] = self.ssc_in[tc]
            Cnd[self.input_cells] = self.ssc_in[tc]

            #Other initialisations
            Pd[:] = 0.0    # deposition probability
            flows = []     # discharges in/out the cell

            #Computing variables that does not change in the iteration process
            for c in range(egm.NoC):
                
#                # Average velocity
#                u[c] = np.average( np.absolute( V[t,self.links[c]] ) )
                
                # Magnitude of cell's velocity vector
                vel_x = np.average(V[t,self.same_y[c]])
                vel_y = np.average(V[t,self.same_x[c]])
                u[c] = (vel_x**2 + vel_y**2)**0.5

                # Probability (actually, a percentage) of deposition
                if u[c] < self.Ucd:
                    Pd[c] = 1 - (u[c]/self.Ucd)**2
                
#                # Surface area
#                gama[c] = delta_t / (egm.dx[c] * egm.dy[c])    # gama = time step [s] / area [m^2]
#                
#                if egm.botwid[c] > 0.0:
#                    
#                    if H[t,c] > egm.chndep[c]:                        
#                        pass
#                    
#                    else:    #when the water content is within the channel, area = bottom width * channel length
#                        gama[c] = delta_t / (egm.botwid[c] * egm.dx[c])

                # Discharges in/out the cell
                qc = Q[t,self.links[c]] * self.signal[c]
                flows.append(qc)

                # Sum of negative discharges
                negSum = np.sum(qc[qc<0])

                # Compounding the denominator of the formula to find C_(c)^(t)
                den[c] = H[t,c] - delta_t * Pd[c] * self.ws - gama[c] * negSum

                # Condition to assure stability (if you pay attention to the signals, den > H[t,c] always)
                if abs(den[c]) < 0.000001:
                    Cnd[c] = 0.0

            # Inicial values for maximum error and iteration number
            maxError, it = 9.e9, 0

            # Iterating until reach the error tolerance
            while(maxError > self.tolerance):

                for c in range(egm.NoC):

                    # Do not compute the sediment concentration at input cells
                    if self.input_cells[c]:
                        continue

                    # Do not compute the sediment concentration where it can cause instability
                    if abs(den[c]) < 0.000001:
                        continue

                    # Sum of (discharge x concentration) of incoming cells
                    posSum = 0.0

                    for j in range(flows[c].size):

                        if flows[c][j] > 0:

                            posSum += flows[c][j] * Cst[self.nbcells[c][j]]

                    # Computing new guess for sediment concentration
                    Cnd[c] = (H[t-1,c] * C[t-1,c] + gama[c] * posSum) / den[c]

                maxError = np.max(np.absolute(Cnd - Cst))
                it += 1
                Cst[:] = Cnd[:]

            # Passing values of concentration
            C[t,:] = Cnd[:]
            info[0].append(maxError)
            info[1].append(float(it))

            # Settled sediment [g/m2]
            if returnSettled:
                settled[t,:] = delta_t * Pd[:] * (-1 * self.ws) * C[t,:]
        #---------------------------------------------------------------------------------------------------------------

        # Model running efficiency
        if informRun:
            avgE = sum(info[0]) / len(info[0])
            avgN = sum(info[1]) / len(info[1])
            msg = ' > Sediment Transport Model: [avg Error, max Error, avg Nit, max Nit] ='
            print(msg, str('[%f, %f, %f, %f]' % ( avgE, max(info[0]), avgN, max(info[1]) )) )

        # Returning calculated data
        if returnSettled:
            return C, settled

        else:
            return C
#-----------------------------------------------------------------------------------------------------------------------



# Functions for simulation
#-----------------------------------------------------------------------------------------------------------------------
def default_files(simul_name='playa'):
    ''' Return the files names generated by the Simulaciones interface or that it will be generated by the
    hydrodynamic model. '''
    names = {
        'anpanta': 'anpanta.dat',
        'contab': 'contab.dat',
        'contar': 'contar.dat',
        'depth': 'depths_'+simul_name+'.txt',
        'velocity': 'veloc_'+simul_name+'.txt',
        'discharge': 'flows_'+simul_name+'.txt',
        'gase': 'gase.dat',
        'gener': 'gener.dat',
        'h1h2': 'h1h2.dat',
        'hrugosi': 'hrugosi.dat',
        'inicial': 'inicial.dat',
        'last_state': 'last_state.txt',
        'lluvia': 'lluvia.dat',
        'param': 'param.dat',
        'vincul': 'vincul.dat',
        'special_tiles': 'special_tiles.dat',
        'exec': 'acc_hydro_short.exe'
    }
    return names

    
def readHydroData(data_file_name, rdec=None):
    ''' This function reads the results from the hydrodynamic model (water depth, velocity or discharge).
    Inputs:
        data_file_name |str| = the path/name of the hydrodynamic results
            The first line of the file must have to contain the values of 'n', 't' (i.e. number of cells or links and
            the time-step between records).
            From the second line on, there are the records for each cell/link in a given time step. The data from the
            next time step is concatenated to the previous one, thus one must know before-hand the number of cells/links
            to properly split the data between the time-steps.
        rdec |int| = round the number in the output matrix to 'rdec' decimals
    Outputs:
        ts |float| = the time step in seconds between each data series
        data |numpy.array((t,n), float)| = matrix of cell/link-property in 't' time steps and 'n' cells/links.
    '''
    f = open(data_file_name, 'r')
    fct = f.readlines()
    f.close()    
    aux = fct[0].split()
    num, ts = int(aux[0]), float(aux[1])    
    d = []
    
    for l in fct[1:]:
        for x in l.split():
            
            try:
                d.append(float(x))
            except ValueError:
                pass
    
    nt = len(d) // num
    d = np.array(d, dtype=np.float64).reshape((nt,num))
#    print('\nRead', data_file_name, ' with shape =', d.shape)
    
    if rdec == None:
        return ts, d
    else:
        return ts, np.around(d, decimals=rdec)


def getNumbers(vincul_file):
    ''' Return the number of cells and the number of links which are wrote in this order in the
    first line of the vincul.dat file. '''
    
    #Reading vincul.dat file content
    archive = open(vincul_file, 'r')
    line1 = archive.readline()
    archive.close()
    
    #Number of cells, NoC, and number of links, NoL
    return list(map(int, line1.split()))


def averageSSC(ssc, h, cd=None, i0=None, iN=None):
    ''' Compute the depth-weigthed-average of suspended solid concentration (SSC).
    Inputs:
        ssc |np.array((NT,NoC), float)| = matrix of sediment (suspended solid) concentration (g/m3).
            NT is the number of time steps; NoC is the number of cells in the domain.
        h |np.array((NT,NoC), float)| = matrix of water depth data
        cd |np.array((NoC,), float)| = array of cell's channel depths
        i0 |int| = position between 0 and NT-1 (or iN) to start the calculation of the average value
        iN |int| = position between 0 (or i0) and NT-1 to end the calculation of the average value
    Outputs:
        avg |np.array((NoC,), float)| = array with average values of SSC in each of the NoC cells
    '''

    if i0 == None:
        i0 = 0
    if iN == None:
        iN = ssc.shape[0]
    
    noc = ssc.shape[1]
    
    if cd is None:
        cd = np.zeros(noc, dtype=np.float64)

    avg = np.zeros(noc, dtype=np.float64)

    # Considering water levels to compute the average
    for c in range(noc):

        num, den = 0.0, 0.0
        
        for t in range(i0, iN):
            
            olh = h[t,c] - cd[c]    #over-land water depth

            if olh > 0:
                num += ssc[t,c] * olh
                den += olh

        if den > 0:
            avg[c] = num/den
    
#    Cmax = np.max(avg)
#    
#    if Cmax > 37.001:
#        
#        cell = np.where(avg == Cmax)[0]
#        print(' WARNING: Maximum concentration of', Cmax, 'g/m3 at cells:', cell)

    return avg


def sscAtPeak(ssc, h):
    ''' Get the SSC at the highest depth.
    Inputs:
        ssc |np.array((NT,NoC), float)| = matrix of sediment (suspended solid) concentration (g/m3).
            NT is the number of time steps; NoC is the number of cells in the domain.
        h |np.array((NT,NoC), float)| = matrix of water depth data (m)
    Return:
        C |np.array((NoC), float)| = array with sediment concentration at the highest depths.
    '''
    C = np.zeros((ssc.shape[1]), dtype=np.float64)
    
    for i in range(ssc.shape[1]):
        
        hMax = -9.e9
        
        for j in range(ssc.shape[0]):
            
            if h[j,i] > hMax:
                
                hMax = h[j,i]
                C[i] = ssc[j,i]
    
    Cmax = np.max(C)
    
    if Cmax > 37:
        
        cell = np.where(C == Cmax)[0]
        print(' WARNING: Maximum concentration of', Cmax, 'g/m3 at cells:', cell)
    
    return C


def diffusion(z, D, nrows, ncols, start_at_column, ignore_rows=[]):
    ''' Apply a simple "geomorphic diffusion" model for landform evolution, in which the downhill flow of soil
    is assumed to be proportional to the (downhill) gradient of the land surface multiplied by a transport coefficient.
    
    Inputs:
        z |np.array((NoC), float)| = array with terrain elevation in the simulated domain (NoC = Number of Cells)
        D |float| = transport rate coefficient (hillslope diffusivity) m2/year
        nrows, ncols |int, int| = rectangular domain dimensions
        start_at_column |int| = first column of the tidal-flat part of the domain (to avoid including the tide-input
            creek in the model)
        ignore_rows |list of ints| = rows where the model won't be applied.
    Return:
        zd |np.array((NoC), float)| = terrain array after diffusion
    
    IMPORTANT: The model is applied for a single EGM-timestep, thus the unity of D is actually m2/(year*EGM-timestep)'''
    dx, zd = 10, z.copy()
    for row in range(nrows):
        if row in ignore_rows:
            continue
        c0 = row*ncols + start_at_column
        cN = (row+1)*ncols
        prep = z[c0] - (z[c0+1] - z[c0])
        posp = z[cN-1] + (z[cN-1] - z[cN-2])
        qs = -D * np.diff(z[c0:cN], prepend=prep, append=posp)/dx
        dzdt = -np.diff(qs)/dx
        zd[c0:cN] += dzdt
    return zd


def createAverageMatrix(datain, inp=None, max_close_cells=2):
    ''' Identify the index of close cells to use in the areal average procedures.
    Inputs:
        datain |class egmdata| = Data object with all the information of a egmdata class
        inp |np.array((NoC), boolean)| = array to indicate cells which must not be listed as neighbor cell
        max_close_cells |int| = Window size, in number of adjacent cells, to look for linked cells
    Outputs:
        matrix |list of np.array((NoC,), int)| = list of arrays with the index of adjacent cells
    '''
    if inp is None:
        inp = np.array([False for i in range(datain.NoC)])
    
    matrix = []

    for i in range(datain.NoC):        
        neighs = [i]
        
        if inp[i]:
            matrix.append(np.array(neighs, dtype=np.int32))
            continue
        
        visited = []
        
        for j in range(max_close_cells):        
            new_neighs = []
            
            for icel in neighs:
                
                if icel in visited:    continue
                
                for l in range(datain.NoL):
                    
                    #Avoid cells linked by weir structure
                    if datain.lktype[l] == 1 or datain.lktype[l] == 11:
                        continue
                    
                    if datain.number[icel] == datain.cell1[l]:
                        new_neighs.append(datain.cell2[l] - 1)
                        
                    elif datain.number[icel] == datain.cell2[l]:
                        new_neighs.append(datain.cell1[l] - 1)
                        
                    else:
                        pass
                
                visited.append(icel)
            
            if len(new_neighs) == 0:
                break
                
            else:
                for adjc in new_neighs:
                    if adjc not in neighs:
                        neighs.append(adjc)
        
        # Removing cells whic are in the input list or are channel type
        aux = np.array([icel for icel in neighs if not inp[icel]], dtype=np.int32)
        matrix.append(aux)
    
    return matrix


def readInputSeries(contar_file):
    
    archive = open(contar_file, 'r')
    fct = archive.readlines()
    archive.close()
    
    line = fct[0].split()
    ws = {'Ndata': int(line[0]), 'dt': float(line[1]), 'Nseries': int(line[2])}
    ws['info'] = np.zeros((ws['Nseries'], 3), dtype = np.int32)
    ws['data'] = np.zeros((ws['Nseries'], ws['Ndata']), dtype=np.float64)
    
    l = 1
    for s in range(ws['Nseries']):
        
        ws['info'][s] = list(map(int, fct[l].split()))
        l += 1
        
        for n in range(ws['Ndata']):
            
            ws['data'][s,n] = float(fct[l])
            l += 1
    
    return ws


def readSpecialTiles(spec_tiles_file):
    
    archive = open(spec_tiles_file, 'r')
    fct = archive.readlines()
    archive.close()
    
    specs, i = {}, 0
    
    while i < len(fct):
        
        line = fct[i].split()
        tile, roughness, ncells = line[0], float(line[1]), int(line[2])
        selection = []
        
        for j in range(ncells):
            
            i += 1
            selection.append(int(fct[i]))
            
        specs[tile] = (roughness, tuple(selection))
        i += 1
    
    return specs


def boundary_depths(bl, z):
    ''' Create a series of water depths on every boundary-conditioned cell by subtracting the bottom elevation from the
    water level.

    Inputs:
        bl |np.array((ndata,), float)| = series of 'ndata' water LEVELS records to be applied on all active HxT boundary cells
        z |np.array((nbound,), float)| = elevation of every 'nbound' HxT boundary cells

    Returns:
        iwh |np.array((nbound,ndata), float)| = series of water DEPTHS on every HxT boundary cell
    '''
    iwh = np.zeros((z.size, bl.size), dtype=np.float64)    #input water depths

    for j in range(z.size):
        for i in range(bl.size):
            iwh[j,i] = max(0.0, bl[i] - z[j])

    return iwh
    

def updateInputSeries(datain, inc):
    ''' Update the input water level series by adding a constant value
    datain = {
            'Ndata': int, 'dt': float, 'Nseries': int,
            'info': np.array((Nseries,3), int),
            'data': np.array((Nseries,Ndata), float)
        }
    '''
    dataout = deepcopy(datain)
    dataout['data'] += inc
    return dataout


def updateParameters(ed, cmann, specs=None, w_new=1.0):
    ''' Update the land-phase roughness coefficient for every link in the domain, the channel depth
    in river-river links and the land-phase elevation in land-river links.
        
    Inputs:
        ed |egm_lib.egmdata()| = variable from class egmdata() with all information of the EGM run.
            It needs to have the vegetation array "ed.V" within! It will also use the channel depth
            "ed.chndep" and cell elevation "ed.z".
        cmann |dict| = dictionary with the Manning's n for every vegetation class in ed.V. The
            vegetation class are the keys and the roughness coefficient the values.
        specs |dict| = dictionary to identify cells with especiall treatment. Each key is a string
            naming the selection. The value for each key is a tuple of two elements. The first is
            the Manning's n that will be assigned to the cells in the list stored in the second
            position.
        w_new |float| = value between 0 and 1 to weight the average with the current roughness
            values. If one, it will change the Manning's n to the new value, discarding the current
            values.
        
    Return:
        pars |np.array((NoL,12), float)| = parameters' matrix similar to ed.params but with the
            updated values.        
    '''
    if specs == None:
        special_tiles = []
    else:
        special_tiles = specs.keys()
    
    pars = deepcopy(ed.params)
    
    for i in range(ed.NoL):
        
        # Getting the roughness in cell1 and cell2 of the current link
        c1, c2 = ed.cell1[i] - 1, ed.cell2[i] - 1
        n1, n2 = cmann[ed.V[c1]], cmann[ed.V[c2]]
        
        # Checking if the cells are in special areas
        for s in special_tiles:
            
            if ed.cell1[i] in specs[s]["cells"]:
                n1 = specs[s]["roughness"]
            
            if ed.cell2[i] in specs[s]["cells"]:
                n2 = specs[s]["roughness"]
        
        # Updating parameters
        if ed.lktype[i] == 61:
            
            # Land-Land link: update roughness between cells
            pars[i,2] = w_new * 0.5 * (n1 + n2) + (1 - w_new) * pars[i,2]
            
        elif ed.lktype[i] == 6:
            
            # River-Land link: update roughness between cells and elevation of land-phase
            pars[i,2] = w_new * 0.5 * (n1 + n2) + (1 - w_new) * pars[i,2]
            
            if ed.chndep[c1] > 0.0:    # first cell is the river
                pars[i,3] = ed.chndep[c1] + ed.z[c1]
                pars[i,4] = ed.z[c2]
                
            else:    # first cell is land type (second is river type)                
                pars[i,3] = ed.z[c1]
                pars[i,4] = ed.chndep[c2] + ed.z[c2]
            
        elif ed.lktype[i] == 0:
            
            # River-River link: update land-phase roughness and average channel depth
            pars[i,3] = w_new * 0.5 * (n1 + n2) + (1 - w_new) * pars[i,3]
            pars[i,6] = 0.5 * (ed.chndep[c1] + ed.chndep[c2])
            
        else:
            
            #Other link types: No changes at all
            pass
        
    return pars


def updateFiles(data, file_names):
    
    #Rewriting file with cell's proprierties
    archive = open(file_names['vincul'], 'r')
    fct = archive.readlines()
    archive.close()
    
    for i in range(data.NoC):
        
        line = str('%i %i' % (data.number[i], data.border[i]))
        line += str(' %.4f %.3f %.3f %.3f %.3f' % (data.z[i], data.x[i], data.y[i], data.dx[i], data.dy[i]))
        line += str(' %.4f %.4f %.4f\n' % (data.botwid[i], data.latslp[i], data.chndep[i]))
        fct[i + data.NoL + 2] = line
    
    archive = open(archive.name, 'w')
    archive.writelines(fct)
    archive.close()
    
    #Rewriting file with link's parameters values
    archive = open(file_names['param'], 'r')
    fct = archive.readlines()
    archive.close()
    
    for i in range(data.NoL):
        
        line = str('%5i %5i' % (data.cell1[i], data.cell2[i]))
        
        if data.lktype[i] == 61:                             #land-land link type
            line += str(' %9.3f %9.3f %9.5f\n' % tuple(data.params[i,0:3]))
            
        elif data.lktype[i] == 6:                            #land-river link type
            line += str(' %9.3f %9.3f %9.5f %9.4f %9.4f\n' % tuple(data.params[i,0:5]))
            
        elif data.lktype[i] == 0 or data.lktype[i] == 10:    #river-river link type
            line += str(' %9.3f %9.3f %9.5f %9.5f %9.4f %9.4f %9.4f\n' % tuple(data.params[i,0:7]))
            
        elif data.lktype[i] == 1:                            #weir-structure link type without gate
            line += str((11*' %9.4f' + '\n') % tuple(data.params[i,0:11]))
            
        elif data.lktype[i] == 11:                           #weir-structure link type with gate
            line += str((12*' %9.4f' + '\n') % tuple(data.params[i,0:12]))
        
        fct[i] = line
    
    archive = open(archive.name, 'w')
    archive.writelines(fct)
    archive.close()
    
    # Replacing water depths input series files
    with open(file_names['contar'], 'w') as f:
        
        # First line
        ndata = data.inseries['data'].size
        step  = data.inseries['dt']
        nseries = data.inseries['where'].shape[0]
        f.write('%i %i %i\n' % (ndata, step, nseries))
        
        # Water DEPTHS on every HxT boundary-cell
        bd_cells = np.zeros(nseries, dtype=int)
        
        for i in range(nseries):
            bd_cells[i] = data.inseries['where'][i,0] - 1
            
        ht = boundary_depths(data.inseries['data'], data.z[bd_cells])
        
        # Series info/data
        for i in range(nseries):
            f.write('%i %i %i\n' % tuple(data.inseries['where'][i]))
            
            # Data as one value per row
            for j in range(ndata):
                f.write('%.6f\n' % ht[i,j])
                
    # Using water depths in last_state.txt to replace initial condition
    with open(file_names['inicial'], 'r') as f:
        inif = f.readlines()
    
    with open(file_names['last_state'], 'r') as f:
        last = f.readlines()
    
    new_inif = inif[0:2] + last[1:]
    with open(file_names['inicial'], 'w') as f:
        f.writelines(new_inif)


def hydrodynamic(data, runcommand, file_names, tolerance=0.001, maxIteration=5, printAll=False):
    
    #Runing hydrodynamic model until reach equilibrium
    maxDif, it = 9.e9, 0    
    while maxDif > tolerance:

        t, matrix1 = readHydroData(file_names['depth'])
        system(runcommand)
        t, matrix2 = readHydroData(file_names['depth'])
        
        aux = np.absolute(matrix1 - matrix2)
        maxDif = np.nanmax(aux)
        it += 1
        updateInitialCondition(file_names['inicial'], file_names['last_state'])
        
        if it == 1:
            print(' > maximum difference on 1st iteration =', maxDif)
        else:
            if printAll:
                print(' > maximum difference on', it, '-th iteration =', maxDif)
                
        if it == maxIteration:
            #matrix2 = (matrix2 + matrix1) * 0.5
            break
    
    if maxIteration > 1:
        print(' > maximum difference, number of iterations =', maxDif, ',', it)
    
    #Read velocity and discharge files if the hydrodynamic was set to write these files
    f = open(file_names['anpanta'], 'r')
    fc = f.readlines()
    f.close()
    
    if fc[2][1] in ['N', 'n']:
        return t, matrix2
    
    if fc[2][1] in ['S', 's', 'Y','y']:
        nothing, vel = readHydroData(file_names['velocity'])
        nothing, dcg = readHydroData(file_names['discharge'])
        return t, matrix2, vel, dcg


def bathtub(sad, runtime_file):
    ''' Create the water levels matrix, similar to the output matrix of the hydronamic model, but based on the bathtub
    assumption.
    
    INPUTS
    sad |egmdata()| = study area data
    runtime_file |str| = file with time-steps, final steps and interval for record data. Usually is the 'inicial.dat'
        file
    
    OUTPUTS
    tibr |float| = time interval between records of the hydrodynamic model
    h |np.array((NT,NoC), float)| = matrix of water depths from the hydrodynamic model. NT is the number of time step
        of this model and NoC is the number of cells.
    '''
    with open(runtime_file, 'r') as f:
        aux = f.readlines()
        info = aux[1].split()
    
    tibr = int(info[-1])
    last_ts = int(info[-2])    
    intervals = range(0, last_ts, tibr)
    h = np.empty((len(intervals), sad.NoC)) * np.nan
    
    for j, t in enumerate(intervals):
        
        for i in range(sad.inseries['Ndata']):
            
            ti = i * sad.inseries['dt']
            
            if ti > t:
                
                a = (sad.inseries['data'][0,i] - sad.inseries['data'][0,i-1]) / sad.inseries['dt']
                b = sad.inseries['data'][0,i] - a * ti
                wl = a * t + b
                break
            
            if i == sad.inseries['Ndata'] - 1:
                if ti == t:
                    wl = sad.inseries['data'][0,i]
                else:
                    raise IndexError('Not enought data to interpolate water level')
        
        depths = wl - sad.z
        depths[depths < 0] = 0
        h[j,:] = depths[:]
    
    return tibr, h


def computeTidalIndexes(h, Hlimit=0.14, cd=None, ti=None, increaseSteps=True):
    ''' Compute the tidal indexes: hydroperiod, H, and mean depth below high tide, D, using the following inputs:

    INPUTS
    h |np.array((NT,NoC), float)| = matrix of water depths from the hydrodynamic model. NT is the number of time step of
        this model and NoC is the number of cells.
    Hlimit |float| = value above which a cell is considered inundated (used for the hydroperiod)
    cd |np.array((NoC), float)| = array with the channel depth in each cell. If it is None, create a zero-values array
        of dimension NoC. The channel depth is subtracted from the water depth to compute the tidal indexes for the
        land-area of channel cells.
    ti |list of np.array((*), int)| = a list of arrays, where each array stores a range of time indexes of 'h' matrix,
        in a way that h[t,c] return the water depths in the cell c in the time step present in t. This input is used to
        take the maximum values in each period, which is supposed to represent the high tides, and than compute the mean
        depth below high tide. If t is None, a list with just one array containing all time indexes is created and the
        maximum level will be the mean depth b.h.t.
    increaseSteps |bool| = if True create a linear interpolation between every pair of h-data to refine the precision
        of the hydroperiod data.
    
    OUTPUTS
    HP |np.array((NoC), float)| = array with the values of hydroperiod in %
    MD |np.array((NoC), float)| = array with the values of mean depth below high tide in m
    '''    
    if cd is None:
        cd = np.zeros((h.shape[1]), dtype=np.float64)
    
    if ti is None:
        ti = [np.arange(h.shape[0])]
    
    #Computing the hydroperiod
    HP = np.zeros((h.shape[1]), dtype=np.float64)
    
    if increaseSteps:
        
        for c in range(h.shape[1]):           #loop through the cells in the domain
            for t in range(h.shape[0]-1):    #loop throught the time-steps in the hydrodynamic data
                
                h0 = h[t,c] - cd[c]
                h1 = h[t+1,c] - cd[c]
                hint = np.linspace(start=h0, stop=h1, num=10, endpoint=False)
                HP[c] += hint[hint >= Hlimit].size
        
        HP = HP * 10 / (h.shape[0] - 1)
        
    else:
        
        for t in range(h.shape[0]):
            
            wet = np.where( (h[t] - cd) >= Hlimit )
            HP[wet] += 1.
        
        HP = HP * 100 / h.shape[0]
    
    #Computing the mean depth below high tide
    MD = np.zeros((h.shape[1]), dtype=np.float64)
    
    for period in ti:
        
        for i in range(h.shape[1]):
            
            MD[i] += h[period,i].max()
    
    MD = MD / len(ti) - cd
    aux = np.where(MD < 0.0)
    MD[aux] = 0.0
    
    return HP, MD


def vegetation(HP, MD, avgM=None, inp=None):
    ''' Applies the vegetation establishment rules due to the tidal indexes in each cell.
    
    INPUTS
    HP |np.array((NoC), float)| = array with the values of hydroperiod in %. NoC is the number of cells
    MD |np.array((NoC), float)| = array with the values of mean depth below high tide in m
    avgM |list of np.array((NoC,), int)| = matrix with neighboring cells for each cell in domain. If it is different
        of None, than use this matrix to compute the predominant vegetation around each cell.
    inp |np.array((NoC), boolean)| = array to indicate cells with fixed vegetation
    
    OUTPUTS
    V or Vavg |np.array((NoC), int)| = array with vegetation-especific values due to survaibility rules. These
        values are presented below. If avgM is given a spatial filter is used to compute the predominant vegetation.
            0 = No vegetation
            1 = Grassland
            2 = SaltMarsh
            3 = Mangrove
    '''
    NoC = HP.shape[0]
    V = np.zeros(NoC, dtype=np.int32)
    
    for i in range(NoC):
        
        if HP[i] < 80. and MD[i] < 0.25:      #Saltmarsh
            V[i] = 2
            
        if HP[i] == 0.0 and MD[i] < 0.055:    #Grassland
            V[i] = 1
        
        if 10. <= HP[i] <= 50.:               #Mangrove
            if MD[i] >= 0.20:
                V[i] = 3
    
    if avgM is not None:
        
        Vavg = np.zeros((NoC), dtype=np.int32)
        
        for i in range(NoC):
            
            aux = V[avgM[i]]
            predominant = 0
            n = aux.count(predominant)
            
            for j in range(1,4):
                
                m = aux.count(j)
                
                if m >= n:              # The most frequent vegetation-value in 'aux' will be used to represent the
                    predominant = j     #vegetation in the cell i. Using '>=' means that, whenever a draw occurs, the
                    n = m               #highest vegetation-value is used.
            
            Vavg[i] = predominant
        
        V = Vavg
    
    if inp is not None:
        V[inp] = 0
    
    return V


def accretion(Darray, Varray, C=None, fixedC = 37.0, avgM=None, mult=1):
    ''' This function computes the biomass and soil accretion due to vegetation type and tidal indexes.
    Inputs:
        Darray |np.array((NoC,), float)| = array of mean depth below high tide values, in meters, on each element
            (NoE = number of elements).
        Varray |np.array((NoC,), int)| = array of vegetation type (see tideNvegetation function description)
        C |np.array((NoC,), float)| = array of suspended solid concentration, in g/m3, on each element
            (NoE = number of elements). If it is None, use a relationship with Darray.
        fixedC |float| = static sediment input concentration used when C == None is True
        avgM |list of np.array((NoC,), int)| = list of arrays with the indexes of neighbors cells to compute am area
            average
        mult |int| = a constant to multiply the yearly accretion rate to get an accumulated value        
    Outputs:
        B |np.array((NoC,), float)| = array of biomass production, in g/m2, on each cell element.
        Acc |np.array((NoC,), float)| = array of total accreation, in metres, on each cell element.
    '''
    Acc = np.zeros(Darray.shape[0], dtype=np.float64)
    B = np.zeros(Darray.size, dtype=np.float64)

    if C is None:
        C = (0.55 * Darray + 0.32) * fixedC
    
    #Saltmarsh
    spot = np.where(Varray == 2)
    B[spot] = 8384.0*Darray[spot] - 16767*(Darray[spot]**2)
    Acc[spot] = C[spot] * (0.00009 + 6.2e-7*B[spot]) * Darray[spot]
    # Acc[spot] = -0.2063606*(Darray[spot]**4) - 0.02450337*(Darray[spot]**3) + 0.06562834*(Darray[spot]**2) \
    #             + 0.00110538*Darray[spot]
    
    #Mangrove
    spot = np.where(Varray == 3)
    B[spot] = 7848.9*Darray[spot] - 6037.6*(Darray[spot]**2) - 1328.3
    Acc[spot] = C[spot] * (0.00009 + 1.2e-7*B[spot]) * Darray[spot]
    # Acc[spot] = -0.0147438*(Darray[spot]**4) + 0.01058874*(Darray[spot]**3) + 0.009751695*(Darray[spot]**2) \
    #             - 0.00081454*Darray[spot]
    
    if avgM != None:
        
        #Averaging        
        aux = np.zeros(Darray.shape[0], dtype=np.float64)
    
        for i in range(Darray.shape[0]):
            aux[i] = sum(Acc[avgM[i]]) / avgM[i].shape[0]
        
        Acc = aux
        
    return B, Acc * mult


def updateInitialCondition(initFile, refreshFile, old=False):
    ''' This function copy the end water level state of the previous model run, over the initial water levels input
    file.
    Inputs:
        initFile |str| = path/filename of the initial conditions file
        refreshFile |str| = path/filename of the final conditions file
    Outputs:
        This function does not return any output variable.
    '''
    arc = open(refreshFile, 'r')
    fct = arc.readlines()
    arc.close()
    if not old: fct.pop(0)                  #.pop(0) removes the first element
    
    arc = open(initFile, 'r')
    lin1 = arc.readline()
    lin2 = arc.readline()
    arc.close()
    
    fct.insert(0, lin1)         #.insert(position, item)
    fct.insert(1, lin2)
    
    arc = open(initFile, 'w')
    arc.writelines(fct)
    arc.close()
#-----------------------------------------------------------------------------------------------------------------------



# Functions for data save/load actions
#-----------------------------------------------------------------------------------------------------------------------
def saveToPickle(matrix, model_step, pickle_step, prefix='data_', ref=None):
    
    ''' This function stores the data from the input matrix in a pickle file. It might reduce the data frequency if
    the pickle_step is different from model_step.
    
    Inputs:
        matrix |np.array((t,n), float)| = data matrix with 't' time steps and 'n' cells/links.
        model_step |float| = original time step in the hydrodynamic model output files.
        pickle_step |float| = desired time step to save in the pickle file. Linear interpolation will be applied if
            needed to create data in time steps not recorded in the original outputs.
        prefix |str| = string to define the begining of the pickle file name.
        ref |str| = string to define the ending of the pickle file name. If is None the current time will be used.
    
    Outputs:
        This function does not return anything. It just create a pickle file to store the input data in the required
            frequency.
    
    OBS: To load the dumped data, use the following syntax:
        archive = open(filename, 'rb')
        time_stamps, data_matrix = pickle.load(archive)
    '''
    if ref is None:
        aux = datetime.now()
        ref = aux.strftime('%Y%m%d_%H%M%S')
    
    if isclose(model_step, pickle_step, rel_tol=1.e-1):
        #copy the matrix data because the time steps are the same
        matrix2 = matrix
        
    else:
        #have to find data in the required time steps or interpolate if there is not
        model_time = np.arange(matrix.shape[0]) * model_step
        matrix2, ti = [], 0
        
        while ti <= model_time[-1]:            
            idx = np.where(model_time == ti)[0]
            
            if idx.size == 1:
                #there are records at time ti in data matrix
                matrix2.append(matrix[idx])
                
            elif idx.size == 0:
                #there are NOT records at time ti in data matrix. Need to find the closest ones to interpolate.
                i_bef = np.where(model_time < ti)[0][-1]
                i_aft = np.where(model_time > ti)[0][0]
                a = (matrix[i_aft] - matrix[i_bef]) / (model_time[i_aft] - model_time[i_bef])
                b = matrix[i_bef] - a * model_time[i_bef]
                matrix2.append(a*ti + b)
                
            else:
                raise ValueError('Inapropriate time step')
            
            ti += pickle_step
        
        matrix2 = np.array(matrix2, dtype=np.float64)
    
    #writing pickle file
    time_stamps = np.arange(matrix2.shape[0]) * pickle_step
    file = open(prefix+ref+'.pkl', 'wb')
    dump([time_stamps, matrix], file)
    file.close()


def loadFromPickle(data_year, data_directory=''):
    ''' Load data created with the hydrodynamic model
    
    Inputs:
        data_directory |str| = path where the files are located
        data_year |int| = year used as suffix of data files
        
    Outputs:
        t |np.array((N), dtype=np.float64)| = array with time [s] values at each of the N time-steps
        h |np.array((N,nc), dtype=np.float64)| = matrix with water dephts [m] for N time-steps and nc cells
        v |np.array((N,nl), dtype=np.float64)| = matrix with water velocity [m] for N time-steps and nl links
        q |np.array((N,nl), dtype=np.float64)| = matrix with discharge [m] for N time-steps and nl links
    '''
    names = [data_directory + o + str('_%4.4i.pkl' % data_year) for o in ['depth', 'veloc', 'flow']]
    
    with open(names[0], 'rb') as a:
        t, h = load(a)
        
    with open(names[1], 'rb') as a:
        t, v = load(a)
        
    with open(names[2], 'rb') as a:
        t, q = load(a)
        
    return t, h, v, q
#-----------------------------------------------------------------------------------------------------------------------



# Functions for data organization
#-----------------------------------------------------------------------------------------------------------------------
# Creating arrays of neighbouring information
def relationships(ed):
    ''' Find the related cells and links to each cell in the domain. It create arrays with the indexes for
    neighbouring cells, links and also the flow signal, for each cell.
    
    Inputs:
        ed |egm_lib.egmdata()| = variable from class egmdata() with all information of the EGM run
        
    Outputs:
        nbcells |list of np.array((N), dtype=np.int32)| = a list with ed.NoC arrays. Each array stores the index of
            the N neighbouring cells.
        links |list of np.array((N), dtype=np.int32)| = a list with ed.NoC arrays. Each array stores the index of
            the N links (the index of links goes from 0 to ed.NoL-1, while the index of cells goes from 0 to ed.NoC-1)
        signal |list of np.array((N), dtype=np.int32)| = a list with ed.NoC arrays. Each array stores the signal (+1 or -1)
            for each of the N links. If the current cell is in ed.cell1 of the link, a positive velocity/discharge
            means that the flow is going from cell1 to cell2. A negative means that the cell1 is receiving the flow
            from cell2. However, if the current cell is in ed.cell2, than these orientations are the opposite.
    '''
    nbcells = [[] for i in range(ed.NoC)]
    links   = [[] for i in range(ed.NoC)]
    signal  = [[] for i in range(ed.NoC)]
    
    for i in range(ed.NoL):
        
        #First of all, change from cell number to cell index (Python counting)
        cell1 = ed.cell1[i] - 1
        cell2 = ed.cell2[i] - 1
        
        #Adding neighbour cells to the list
        nbcells[cell1].append(cell2)
        nbcells[cell2].append(cell1)
        
        #Adding links
        links[cell1].append(i)
        links[cell2].append(i)
        
        #Setting signals: positive for incoming flow, negative for outcome
        signal[cell1].append(-1)
        signal[cell2].append(1)
    
    #Converting inner lists to arrays of integer type
    for i in range(ed.NoC):
        nbcells[i] = np.array(nbcells[i], dtype=np.int32)
        links[i] = np.array(links[i], dtype=np.int32)
        signal[i] = np.array(signal[i], dtype=np.int32)
    
    return nbcells, links, signal
#-----------------------------------------------------------------------------------------------------------------------