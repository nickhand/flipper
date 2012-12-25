import os
import numpy
import trace, constants
import liteMap, preconditioner, prior


def map2numpy( m, n, accum = False ):
    if accum:
        n[:][:] += m.data[:][:]
    else:
        n[:][:] =  m.data[:][:]

def numpy2map( n, m, accum = False ):
    if accum:
        m.data[:][:] += n[:][:]
    else:
        m.data[:][:] =  n[:][:]

def dot( n1, n2 ):
    return (n1*n2).sum()

class pcg( object ):
    """
    @brief class for implementing the preconditioned conjugate gradient solution

    See (e.g.) http://en.wikipedia.org/wiki/Conjugate_gradient_method for an overview
    """

    def __init__( self, filterClass = None, filterParams = None, root = constants.NO_VALUE ):
        """
        @brief Setup the pcg problem

        @param filterClass a class from scripts/pcgFilters.py
        @param filterParams dictionary of keyword arguments for filter initialization
        @param root if using MPI, the rank of the root node
        """
        self.maps = {}
        self.weights = {}
        self.nmap = 0
        self.datasets = {}
        self.ntod = 0
        self.proj = {}
        self.filterClass = filterClass
        self.filterParams = filterParams
        self.tolerance = 1e-4
        self.maxIter   = 200
        self.filters    = {}        # for noise weighting in Fourier space
                                    # have "applyFilter" methods which take a map
        self.preconditioners = {}   # objects with "applyPreconditioner(map)" methods
        self.priors = {}            # objects with "applyPrior(arrIn, arrOut)" methods
        self.pcParams        = {}   # parameters for preconditioners
        self.priorParams     = {}   # parameters for priors
        self.r               = {}   # r = b - Ax
        self.b               = {}   # b = M^T N^{-1} d
        self.p               = {}   # the step in map space`
        self.q               = {}   # Ap
        self.x               = {}   # The latest map

        #Additional MPI parameters
        if root != constants.NO_VALUE:
            self.root = root
            self.rootPort = MPI.WORLD[self.root]
            self.doMPI = True
        else:
            self.root = constants.NO_VALUE
            self.rootPort = None
            self.doMPI = False

    def _createProjKey( self, todName, mapName ):
        """
        @brief construct a key for the projector dictionary out of the associated map and tod names
        @param todName name of tod
        @param mapName name of map
        @return projector key string
        """
        return "%s.%s" % (todName, mapName)

    def addMap( self, mp, name, pcParams = None, priorParams = None ):
        """
        @brief add a map to the pcg
        @param mp Map.Map object to add
        @param name string identifying the map, must be different from other maps in the pcg
        @param pcParams optional list of preconditioner dictionaries (see flipper.pcgDefault.par)

        This should be done before adding tods which project to the added map.
        """

        self.nmap += 1
        self.maps[name] = mp
        self.nmap += 1
        self.weights[name] = None
        self.r[name] = numpy.zeros((mp.nrow, mp.ncol))
        self.b[name] = numpy.zeros((mp.nrow, mp.ncol))
        self.p[name] = numpy.zeros((mp.nrow, mp.ncol))
        self.q[name] = numpy.zeros((mp.nrow, mp.ncol))
        self.x[name] = numpy.array(mp.data).copy() # initial map
        self.preconditioners[name] = None
        self.priors[name] = None
        self.pcParams[name] = pcParams
        self.priorParams[name] = priorParams

    def getMapNamesForTOD( self, todName ):
        """
        @brief Get all map names associated with a particular TOD
        """
        todMapNames = []
        allMapNames  = self.maps.keys()
        projNames = self.proj.keys()
        for mapName in allMapNames:
            projName = self._createProjKey(todName, mapName)
            if projName in projNames:
                todMapNames += [mapName]
        return todMapNames       

    def addTOD( self, tod, projDict, filter = None ):
        """
        @brief add a tod and associated projectors to the pcg
        @param tod the TOD.TOD to  add to the pcg
        @param projDict a dictionary with map name keys and projector values
        """
        self.ntod += 1
        self.datasets[tod.name] = tod
        for mn in projDict.keys():
            self.proj[ self._createProjKey( tod.name, mn ) ] = projDict[mn]
        if filter != None:
            self.filters[tod.name] = filter
            self.filters[tod.name].applyFilter()
            print "filta ", tod.name,  self.filters[tod.name] 
        elif filter == None and self.filterClass != None:
            trace.issue("flipper.pcg", 3, "Estimating noise for %s." % tod.name)
            data = tod.data.copy()
            mapsZero = True
            for mapName in self.getMapNamesForTOD( tod.name ):
                if self.x[mapName].max() > 0. or self.x[mapName].min() < 0.:
                    mapsZero = False
            if not mapsZero:
                trace.issue("flipper.pcg", 3, "Subtracting initial maps from %s." % tod.name)
                # Remove maps from data before estimating noise filters
                tod.data[:] *= -1
                self.loadMaps( weight=False )
                self.projectMapsToTOD( tod )
                tod.data[:] *= -1
            #now estimate filters
            self.filters[tod.name] = self.filterClass( tod, self.filterParams )
            tod.data[:] = data[:]
            del data
            self.filters[tod.name].setTOD(tod)
            self.filters[tod.name].applyFilter()
            self.filters[tod.name].setTOD( None )
        else:
            self.filters[tod.name] = None
        self.clearMaps()
        self.projectTODToMaps( tod )
        mapNames = self.getMapNamesForTOD( tod.name )
        for mapName in mapNames:
            map2numpy( self.maps[mapName], self.b[mapName], accum = True )

    def projectMapsToTOD( self, tod ):
        """
        @brief project all maps associated with the a tod into that tod
        @param tod TOD.TOD associated with this pcg
        """
        mapNames = self.getMapNamesForTOD( tod.name )
        for mapName in mapNames:
            projName = self._createProjKey(tod.name, mapName)
            m = self.maps[mapName]
            p = self.proj[projName]
            p.projectMapToData( m, tod )

    def projectTODToMaps( self, tod ):
        """
        @brief Project Data into its associated maps
        @param data Data associated with this pcg
        """
        mapNames = self.getMapNamesForTOD( tod.name )
        for mapName in mapNames:
            projName = self._createProjKey(tod.name, mapName)
            m = self.maps[mapName]
            p = self.proj[projName]
            p.projectDataToMap( tod, m )

    def setWeights( self ):
        """
        @brief sets the weights for all maps
        """
        trace.issue("flipper.pcg", 3, "pcg: setting weights")
        self.clearMaps()
        for ts in self.datasets.values():
            tod = ts
            self.projectTODToMaps( tod )
        for mapName in self.maps.keys():
            self.weights[mapName] = numpy.array(self.maps[mapName].weight).copy()

    def setup( self ):
        """
        @brief initialize all PCG vectors and preconditioners
        Execute after you've added your maps and tods.
        """
        # Compute weight maps and 
        self.setWeights()

        # If we're on a cluster, accumulate the b map
        if self.doMPI:
            trace.issue("flipper.pcg", 3, "pcg: Reducing b and weights")
            for mapName in self.maps.keys():
                mbMPI.reduceArray( self.b[mapName], self.root )
                mbMPI.reduceArray( self.weights[mapName], self.root )
                if self.root == MPI.WORLD.rank:
                    self.maps[mapName].weight[:][:] = self.weights[mapName][:][:]

        # setup and apply preconditioners
        if not self.doMPI or self.root == MPI.WORLD.rank: 
            trace.issue("flipper.pcg", 3, "pcg: masking initial maps with the weight maps")
            for mapName in self.maps.keys():
                self.x[mapName][numpy.where(self.weights[mapName] == 0)] = 0.
            self.loadMaps()
            trace.issue("flipper.pcg", 3, "pcg: setting up preconditioners.")
            for mapName in self.maps.keys():
                if self.pcParams[mapName] != None:
                    self.preconditioners[mapName] = preconditioner.preconditionerList()
                    for pcDict in self.pcParams[mapName]:
                        try:
                            self.preconditioners[mapName].append( apply(eval(pcDict['func']), \
                                  [self.maps[mapName]]  , pcDict['keywords']) )
                        except:
                            trace.issue("flipper.pcg", 0, "pcg: Invalid preconditioner specification for %s" %\
                                mapName)
                            raise

            trace.issue("flipper.pcg", 3, "pcg: Applying preconditioners to b.")
            for mapName in self.maps.keys():
                if self.preconditioners[mapName] != None:
                    self.preconditioners[mapName].applyPreconditioner(self.b[mapName])

        # setup priors
        if not self.doMPI or self.root == MPI.WORLD.rank: 
            trace.issue( "flipper.pcg", 3, "pcg: Creating map priors" )
            for mapName in self.maps.keys():
                if self.priorParams[mapName] != None:
                    self.priors[mapName] = prior.priorList()
                    for priorDict in self.priorParams[mapName]:
                        try:
                            self.priors[mapName].append( apply(eval(priorDict['class']), \
                                    [self.maps[mapName]]  , pcDict['keywords']) )
                        except:
                            trace.issue("flipper.pcg", 0, "pcg: Invalid prior specification for %s" %\
                                    mapName)
                            raise

        # get Ax  (stor in q)
        trace.issue("flipper.pcg", 3, "pcg: Applying inverse covariance to initial map")
        computeAx = False
        for mapVector in self.x.values():
            if mapVector.max() != 0. or mapVector.min() != 0.:
                computeAx = True
        if computeAx:
            self.applyInverseCovariance( self.x, self.q )
        else:
            trace.issue("flipper.pcg", 3, "All initial maps are 0.: not computing M^TN^{-1}Mx_0.") 

        if self.doMPI:
            mbMPI.reduceArray( self.q[mapName], self.root )

        if not self.doMPI or self.root == MPI.WORLD.rank: 

            for mapName in self.maps.keys():                    
                if self.preconditioners[mapName] != None:   # Q M^T N^-1 M x + Q K^-1 x
                    if self.q[mapName].max() != 0 or self.q[mapName].min() != 0.:
                        self.preconditioners[mapName].applyPreconditioner(self.q[mapName])

            for mapName in self.maps.keys():
                self.r[mapName] = self.b[mapName] - self.q[mapName]
                self.p[mapName] = self.r[mapName].copy()

        if self.doMPI:
            for mapName in self.maps.keys():
                mbMPI.broadcastArray( self.p[mapName], self.root )

    def clearMaps(self):
        for m in self.maps.values():
            m.clear()

    def applyInverseCovariance( self, ps , qs):
        """
        @brief compute qs = A ps = M^T N^-1 M ps

        Uses TODs and maps as scratch.

        @param ps a list of pcg vectors to apply the inverse covariance to
        @param qs a list of pcg vectors to accumulate the output in
        """

        for q in qs.values():
            q[:][:] = 0.  #We're accumulating here, so start from 0

        for mapName in self.maps.keys():
            if self.priors[mapName] != None:  # q = K^-1 x
                self.priors[mapName].applyPrior(ps[mapName], qs[mapName])

        for ts in self.datasets.values():
            tod = ts
            self.loadMaps( arr = ps, weight = False )
            self.projectMapsToTOD(tod)                            # M x
            print "filta ", tod.name,  self.filters[tod.name] 
            if self.filters[tod.name] != None:
                self.filters[tod.name].setTOD(tod)
                print "applying map filter in covariance"
                self.filters[tod.name].applyFilter()           # N^-1 M x 
                self.filters[tod.name].setTOD(None)
            self.clearMaps()
            self.projectTODToMaps(tod)                            # M^T N^-1 M x
            self.unloadMaps( arr = qs, accum = True )             # q = M^T N^-1 M x + K^-1 x
#             del tod
        

    def convergence( self ):
        """
        @brief Compute the ratio of the residual over the projected data

        @return |r|/|b|
        """
        num = 0.0
        den = 0.0
        for mapName in self.maps.keys():
            r = self.r[mapName]
            b = self.b[mapName]
            num += dot(r,r) 
            den += dot(b,b)
        return (num/den)**0.5

    def step( self ):
        """
        Iterate the solver. 

        @return True if converged
        """
        #First check convergence
        finished = False
        if not self.doMPI or self.root == MPI.WORLD.rank:
            if self.tolerance > self.convergence():
                if self.root == constants.NO_VALUE:
                    return True
                finished = True

        if self.doMPI:
            finished = self.rootPort.Bcast( finished )
            if finished:
                return True

        # q = Ap
        self.applyInverseCovariance( self.p, self.q )

        if self.doMPI:
            for mapName in self.maps.keys():
                mbMPI.reduceArray( self.q[mapName], self.root )

        if not self.doMPI or self.root == MPI.WORLD.rank: 
            for mapName in self.maps.keys(): # Q M^T N^-1 M x + Q K^-1 x
                if self.preconditioners[mapName] != None:
                    self.preconditioners[mapName].applyPreconditioner(self.q[mapName])
            
            #compute alpha and r^T r
            rTr = 0; pTAp = 0
            for mapName in self.maps.keys():
                rTr  += dot(self.r[mapName], self.r[mapName])
                pTAp += dot(self.p[mapName], self.q[mapName])  # q = Ap
            alpha = rTr/pTAp
    
            #compute new x and r
            for mapName in self.maps.keys():
                self.x[mapName] += alpha*self.p[mapName]
                self.r[mapName] -= alpha*self.q[mapName]
    
            #compute beta
            rTrNew = 0
            for mapName in self.maps.keys():
                rTrNew += dot(self.r[mapName], self.r[mapName])
            beta = rTrNew/rTr
    
            #compute the next step
            for mapName in self.maps.keys():
                self.p[mapName] = self.r[mapName] + beta * self.p[mapName]

        if  self.doMPI:
            for mapName in self.maps.keys():
                mbMPI.broadcastArray( self.p[mapName], self.root )

        return False
        
    def loadMaps( self, arr = None, weight = True ):
        """
        @brief load current map (self.x[i]) into the mapset map (self.mapsets[i].map)
        @param arr if specified, load different arrays. for instance loadMaps(pcg.b) will give you 
          raw the filtered maps
        @param weight bool: load weights too
        In general, the maps are used as scratch space.
        """        
        if arr == None:
            arr = self.x
        for mapName in self.maps.keys():
            if weight:
                if str(self.weights[mapName]) == "None":
                    self.setWeights()
                self.maps[mapName].weight[:][:] = self.weights[mapName][:][:]
            numpy2map( arr[mapName], self.maps[mapName] )

    def unloadMaps( self, arr = None, accum = False ):
        """
        @brief Unload maps into an array
        @param dictionary of arrays to unload into (defaults to self.x)
        @param accum bool: add into array instead of using assignment
        """
        if arr == None:
            arr = self.x
        for mapName in self.maps.keys():
            map2numpy( self.maps[mapName], arr[mapName], accum = accum )
    

class pcgFilter( object ):
    """
    @brief A class to describe our pcg time domain filters

    Implements (1-C)N^{-1}(1-C) where 1-C is the correlation mode removal and
    N^{-1} is the independent noise filtering. Both operations must be linear
    symmetric and non-singular.
    """

    def __init__( self, noiseTOD, filterNoise = True ):
        """
        @param noiseTOD TOD.TOD with best estimate of noise
        @param filterNoise bool if False just perform decorrelation
        """
        pass

    def setTOD( self, tod ):
        """
        @brief Set the tod associated with filters 
        @param tod TOD.TOD to point the filters to
        """
        pass

    def removeCorrelations( self ):
        """
        @brief remove correlation modes (1-C)
        """
        pass

    def noiseFilter( self ):
        """
        @brief filter individual detector noises
        """
        pass

    def applyFilter( self ):
        """
        @brief Filter the TOD. Performs the pcg (1-C)^TN^{-1}(1-C) step.
        """
        trace.issue( "flipper.pcg", 2, "Filtering %s" % self.tod.name)
        self.removeCorrelations() # don't need to repeat 1-C b/c (1-C)^T(1-C) = 1-C
        if self.params['filterNoise']:
            trace.issue( "flipper.pcg", 3, "Noise Filtering")
            self.noiseFilter()
        if not(self.params['useNoiseInCorr']): self.removeCorrelations()

