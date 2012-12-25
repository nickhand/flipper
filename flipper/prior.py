# Module to implement priors on the maps
# the prior is generally an addition to the map covariance of the form K^-1
# s.t. M^T N^-1 M  ->   M^T N^-1 M + K^-1
# These functions give you the K^-1
import numpy
from utilities import mobyUtils

class priorList( list ):
    """
    @brief a class for combining multiple priors
    """
    def applyPrior( self, arrIn, arrOut ):
        """
        Apply, in order, the priors in the list
        """
        for p in self:
            p.applyPrior( arrIn, arrOut )

class prior( object ):
    """
    @brief a superclass to specify general format of priors
    """
    def __init__( self, mp ):
        """
        @param mp associated mapUtils.Map.Map
        """
        pass

    def applyPrior( self, arrIn, arrOut ):
        """
        @brief apply prior weighting on arrIn and add it to arrOut
        @param arrIn array to be weighted (2 dimensional, map sized)
        @param arrOut array to accumulate arrIn into
        """
        pass

class trivial( prior ):
    """
    @brief unity weighting
    """
    def applyPrior( self, arrIn, arrOut ):
        arrOut[:] += arrIn[:]


class multiplyByWeight( prior ):
    """
    @brief multiply by the weight map (counters divide by weight preconditioner)
    """

    def __init__( self, mp, scale = 1.0 ):
        """
        @param mp the Map.Map from which we'll derive the weights
        @param scale overall multiplier
        """
        self.weight = numpy.array(mp.weight).copy() * scale

    def applyPrior( self, arrIn, arrOut ):
        """
        @brief multiplies arrIn by self.weight and accumulates in arrOut
        """
        arrOut[:] += (self.weight*arrIn)[:]


class atmosPrior( prior ):

    def __init__(self, mp, doMask=True, maskSize=35, weightCut=150, keepMask=False, useMask=False, keepIm=False, filterMap=False):
        """
        @brief fit an atmospheric prior to a map.  
        @param mp map from which to compute prior
        @param doMask taper edges to avoid FFT spikes (uses weight mask smoothed with convolution kernel)
        @param maskSize size of 2d smoothing convolution kernel (in pixels)
        @param weightCut defines weight mask: use map pixels with this weight and higher
        @param filterMap filter map as part of initialization step
        @param keepMask keep mask array
        @param useMask use smoothed weight mask during prior application (overrides keepMask if True)
        @param keepIm keep convolved map image
        """
        if doMask:
            kernel=mobyUtils.make2DconvKernel(maskSize)
            mask=mp.weight.copy()*0
            mask[numpy.where(mp.weight>weightCut)]=1
            mask=mobyUtils.conv2(mask,kernel)
            im_to_fit=mp.data*mask
        else:
            im_to_fit=mp.data
    
        prior_fit=fit_atmos_prior_from_im(im_to_fit)
        if keepIm:
            prior_fit['im']=im_to_fit
        self.useMask = useMask
        if useMask or keepMask:
            if doMask:
                prior_fit['mask']=mask
            else:
                print 'warning - output mask requested in fitAtmosPrior, but mask has not been used, hence not calculated.'
        if filterMap:
            filterAtmosFromPrior(mp,prior_fit)
        self.prior = prior_fit

    def applyPrior( self, arrIn, arrOut ):
        """
        applyAtmosPrior(map,prior,useMask=False):
        apply the map prior calculated in prior to the map.data in map and
        return a copy.
        useMask will apply the mask in the prior to the map before applying the prior.
        Shouldn't neet to do this, but might be useful for testing...
        """
        
        im=arrIn
        if self.useMask:
            im=im*self.prior['mask']
        imfilt=apply_atmos_prior_to_im(im,self.prior)
        arrOut[:] += imfilt[:]

####
#Private functions
####

def filterAtmosFromPrior(mp,prior):
    """
    filterAtmosFromPrior(mp,prior):
    Wiener filter the atmosphere mp from the prior passed in.  The
    mp is overwritten.  The prior should containa power-law plus plateau
    fit, plus pre-calculated info on how to apply FFT modes.
    """
    
    im=mp.data.copy()
    imfilt= apply_wiener_filter_from_prior_to_im(im,prior)
    mp.data[:,:]=imfilt

def fit_atmos_prior_from_im(im,index=-3.666667,rmin=0.01,rmax=1.0):
    [dat,r,x1,x2]=mp_ftamps(im)
    params,pred,rr,dd=fit_1over_f(dat,r,ind=index,rmin=rmin,rmax=rmax)
    myprior={}
    myprior['params']=params
    myprior['x1']=x1
    myprior['x2']=x2
    myprior['ind']=index
    return myprior

def apply_atmos_prior_to_im(im,prior):
    imft=numpy.fft.fft2(im)
    rmat=get_rmat(prior['x1'],prior['x2'])
    p=prior['params']
    rmat[numpy.where(rmat==0)]=1
    rmat=(rmat**prior['ind'])
    rmat=rmat*float(prior['params'][1])    
    imft=imft/rmat
    imft[numpy.where(rmat==0)]=0
    imback=numpy.fft.ifft2(imft)
    return numpy.real(imback)

def apply_wiener_filter_from_prior_to_im(im,prior):
    imft=numpy.fft.fft2(im)
    rmat=get_rmat(prior['x1'],prior['x2'])
    p=prior['params']
    rmat[numpy.where(rmat==0)]=1e-10
    rmat=(rmat**prior['ind'])
    rmat=rmat*float(prior['params'][1])
    rmat2=1/(rmat+float(prior['params'][0]))
    rmat3=rmat*rmat2
    imft2=imft*rmat3
    imback=numpy.fft.ifft2(imft2)
    return numpy.real(imback)
    #return rmat


def mp_ftamps(mp):
    #mpft=numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(mp)));
    mpft=numpy.abs((numpy.fft.fft2(mp)));
    cents=mp.shape
    x1=numpy.arange(0,mp.shape[0])-cents[0]/2+0.0
    x1=x1/(numpy.max(numpy.abs(x1)))
    x2=numpy.arange(0,mp.shape[1])-cents[1]/2+0.0
    x2=x2/(numpy.max(numpy.abs(x2)))
    x1=x1**2
    x2=x2**2
    x1=numpy.transpose(numpy.matrix(x1))
    x2=numpy.matrix(x2)
    x1=numpy.fft.ifftshift(x1)
    x2=numpy.fft.ifftshift(x2)
    #rmat=numpy.fft.ifftshift(numpy.sqrt(numpy.array(x1.repeat(x2.size,1)+x2.repeat(x1.size,0))))
    rmat=get_rmat(x1,x2)
    y1=rmat.ravel()
    y2=mpft.ravel()
    return y2,y1,x1,x2

def get_rmat(x1,x2):
    rmat=(numpy.sqrt(numpy.array(x1.repeat(x2.size,1)+x2.repeat(x1.size,0))))
    return rmat

def fit_1over_f(data,r,ind=-3.666667,rmin=None,rmax=None,params=None,maxiter=10):
    if rmin==None:
        print 'Here rmin'
    else:
        ii=r>rmin
        r=r[ii]
        data=data[ii]
    if rmax!=None:
        ii=r<rmax
        r=r[ii]
        data=data[ii]
    vecs=numpy.ones(len(data))    
    vecs=numpy.transpose(numpy.array([vecs,r**ind]))
    fac=numpy.max(vecs[:,1])/100.0
    vecs[:,1]=vecs[:,1]/fac
    if params==None:
        params=mobyUtils.lsqFit(data**2,vecs);
    iter=0;
    while iter<maxiter:
        pred=vecs*params
        #mylike=like_1overf(params,data,vecs)
        #if (iter==0):
        #    l0=mylike
        #mylike=mylike-l0
        #print numpy.transpose(params)
        
        params=mobyUtils.lsqFit(data**2,vecs,numpy.multiply(pred,pred));
        iter=iter+1
    params[1]=params[1]/fac
    return params,pred,r,data

def like_1overf(params,data,vecs):
    pred=numpy.array(vecs*params)
    data=numpy.array(data)
    print data.shape
    data=data**2
    pp=1/pred
    chisq=numpy.dot(data,pp)
    #dd=numpy.product(data,data)
    #chisq=numpy.product(dd,pp)
    #chisq=numpy.sum(chisq)
    logdet=numpy.sum(numpy.log(pred))
    return -0.5*chisq-0.5*logdet

