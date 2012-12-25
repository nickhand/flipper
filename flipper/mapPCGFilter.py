import noiseInverse
import mapPCG

class mapNoiseFilter( mapPCG.pcgFilter ):
    def __init__( self, m, p2dNoise ):
        self.m = m
        self.p2d = p2dNoise
    def applyFilter( self ):
        filtMap0 = noiseInverse.nInverseMap(self.m,self.p2d,kMask=self.p2d.kMask,showNInverse=False, \
                                            noiseFloorAsPercOfMax = 2.0)
        self.m.data[:] = filtMap0.data[:]

