from fftTools import *
import mapPCG

def nInverseMap(liteMap,p2dNoise,noiseFloorAsPercOfMax = 2,\
                showNInverse=False,zoomUptoL=None,\
                kMask=None, saveFig = None):
    """
    m : map to which inverse noise filter is applied
    m1,m2 : maps from which noise map is evaluated (e.g. half season maps)
    saveFig: specify a file name for writing out
    """
    
    noisePower = p2dNoise.powerMap.copy()
    noiseFloor = noisePower.max()*noiseFloorAsPercOfMax/100.
    #print noiseFloor
    idx = numpy.where(noisePower < noiseFloor)
    #print idx
    noisePower[idx] = noiseFloor
    invNoiseFilter = noisePower.copy()
    invNoiseFilter[:,:] = 1./noisePower[:,:]
    if kMask!=None:
        invNoiseFilter *= kMask
    ft = fftFromLiteMap(liteMap)
    filteredMap = ft.mapFromFFT(kFilter=invNoiseFilter)
    filtLitemap = liteMap.copy()
    filtLitemap.data[:,:] = filteredMap[:,:]
    if showNInverse or saveFig:
        im = pylab.matshow((fftshift(invNoiseFilter.copy())),\
                      origin="down",extent=[numpy.min(p2dNoise.lx),\
                                            numpy.max(p2dNoise.lx),\
                                            numpy.min(p2dNoise.ly),\
                                            numpy.max(p2dNoise.ly)])
        if zoomUptoL!=None:
            im.axes.set_xlim(-zoomUptoL,zoomUptoL)
            im.axes.set_ylim(-zoomUptoL,zoomUptoL)
        pylab.colorbar()
        if showNInverse:
            pylab.show()
        if saveFig != None:
            pylab.savefig( saveFig )
        
    return filtLitemap

