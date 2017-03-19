from __future__ import division
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import numpy as np
import scipy as sp
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from helper_functions import image_to_matrix, matrix_to_image, flatten_image_matrix, unflatten_image_matrix, image_difference

from random import randint,sample
from functools import reduce
def k_means_cluster(image_values, k=3, initial_means=None):
    """
    Separate the provided RGB values into
    k separate clusters using the k-means algorithm,
    then return an updated version of the image
    with the original values replaced with
    the corresponding cluster values.

    params:
    image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    k = int
    initial_means = numpy.ndarray[numpy.ndarray[float]] or None

    returns:
    updated_image_values = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
    """
    # TODO: finish this function
    if(len(image_values.shape) == 3):
        height, width, depth = image_values.shape
    else:
        height, width = image_values.shape
        depth = 1
    flat_values=flatten_image_matrix(image_values)
    #if flat_values.shape[1]==1:
     #   flat_values.shape=(len(flat_values,),)
    
    if initial_means==None:
        initial_means_index= np.array(sample(range(len(flat_values)),k))
        means=np.array(flat_values[initial_means_index])
    else:
        means=np.array(initial_means)
        
    diff_max=10
    count=0

    while count<100: ###insert while condition here for testing whether the clustering has converged
        dist=[]
        for i in range(k):
            dist.append(np.sqrt(((flat_values-means[i])**2).sum(axis=1)))
            
        dist=np.array(dist)
        cluster_indices=dist.T.argmin(axis=1)
        new_means=[]
        diff=[]
        for i in range(k):
            new_means.append(flat_values[cluster_indices==i].mean(axis=0))
            diff.append(np.sqrt(((new_means[i]-means[i])**2).sum(axis=0)))
        diff_max=max(diff)
        if diff_max<.01:
            count+=1
        means=new_means[:]   
      
    
    for i in range(k):
        flat_values[cluster_indices==i]=flat_values[cluster_indices==i].mean(axis=0).T
    new_values= unflatten_image_matrix(flat_values, width)   
    
    return new_values

def default_convergence(prev_likelihood, new_likelihood, conv_ctr, conv_ctr_cap=10):
    """
    Default condition for increasing
    convergence counter:
    new likelihood deviates less than 10%
    from previous likelihood.

    params:
    prev_likelihood = float
    new_likelihood = float
    conv_ctr = int
    conv_ctr_cap = int

    returns:
    conv_ctr = int
    converged = boolean
    """
    increase_convergence_ctr = (abs(prev_likelihood) * 0.9 <
                                abs(new_likelihood) <
                                abs(prev_likelihood) * 1.1)

    if increase_convergence_ctr:
        conv_ctr+=1
    else:
        conv_ctr =0

    return conv_ctr, conv_ctr > conv_ctr_cap

from random import randint
import math
from scipy.misc import logsumexp
class GaussianMixtureModel:
    """
    A Gaussian mixture model
    to represent a provided
    grayscale image.
    """

    def __init__(self, image_matrix, num_components, means=None):
        """
        Initialize a Gaussian mixture model.
        
        params:
        image_matrix = (grayscale) numpy.nparray[numpy.nparray[float]]
        num_components = int
        """
        

        self.image_matrix = image_matrix
        self.num_components = num_components
        self.flat_values=np.array(flatten_image_matrix(self.image_matrix))
        self.flat_values.shape=(len(self.flat_values),)
        if(means is None):
            self.means = [0]*num_components
        else:
            self.means = means
        self.means=np.array(self.means)
        self.variances = [0]*num_components
        self.mixing_coefficients = [0]*num_components
        
        if(len(self.image_matrix.shape) == 3):
            self.height, self.width, self.depth = self.image_matrix.shape
            
        else:
            self.height, self.width = self.image_matrix.shape
            self.depth = 1
            
    
    def joint_prob(self, val):
        """Calculate the joint 
        log probability of a greyscale
        value within the image.
        
        params:
        val = float
        
        returns:
        joint_prob = float
        """
        # TODO: finish this
        m=np.array(np.exp(self.mixing_coefficients))
        sig=np.array(self.variances)
        mu=np.array(self.means)
        joint_prob=np.log(sum(m*((1/np.sqrt(2*sig**2*np.pi))*np.exp(-((val-mu)**2)/(2*sig)))))
        return joint_prob
    
    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean to a random
        pixel's value (without replacement),
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value 
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        
        NOTE: this should be called before 
        train_model() in order for tests
        to execute correctly.
        """
        # TODO: finish this

        
        self.variances=np.array(self.num_components*[1])
        self.flat_values=np.array(flatten_image_matrix(self.image_matrix))
        self.flat_values.shape=(len(self.flat_values),)
        ##set the mean to a random val in the data set
        initial_means_index= np.array(sample(range(len(self.flat_values)),self.num_components))
        self.means=np.array(self.flat_values[initial_means_index])
        self.mixing_coefficients = np.array(self.num_components*[1/float(self.num_components)])
        self.mixing_coefficients=np.log(self.mixing_coefficients)
    def train_model(self, convergence_function=default_convergence):
        """
        Train the mixture model 
        using the expectation-maximization
        algorithm. Since each Gaussian is
        a combination of mean and variance,
        this will fill self.means and 
        self.variances, plus 
        self.mixing_coefficients, with
        the values that maximize
        the overall model likelihood.
        
        params:
        convergence_function = function that returns True if convergence is reached
        """
        # TODO: finish this
        converged=False
        count=0
        j=0
        N=float(len(self.flat_values))
        #self.mixing_coefficients=np.exp(self.mixing_coefficients)
        while converged==False and j<100:#converged==False and 
        #for j in range(25):
            num=[]
            cur_likelihood=self.likelihood()
            for i in range(self.num_components):
                num.append(-.5*np.log(2*np.pi*self.variances[i])-(((self.flat_values-self.means[i])**2)/(2*self.variances[i])))
                #num.append(scipy.stats.lognorm.pdf(self.flat_values,scale=np.exp(self.means[i]),s=np.sqrt(self.variances[i])))
               
            num=np.array(num).T
            #mixing_numerator=self.mixing_coefficients*np.exp(num)
            mixing_numerator=np.exp(self.mixing_coefficients)*np.exp(num)
            #gamma=mixing_numerator/(self.mixing_coefficients*np.exp(num)).sum(axis=1)[:,None]
            gamma=mixing_numerator/(np.exp(self.mixing_coefficients)*np.exp(num)).sum(axis=1)[:,None]
            self.mixing_coefficients=sum(gamma)/N
            #self.mixing_coefficients=scipy.misc.logsumexp(gamma)-np.log(N)
            #self.mixing_coefficients=np.log(sum(np.exp(gamma)))-np.log(N)
            denom=sum(gamma)
            #denom=scipy.misc.logsumexp(gamma)
            self.means=((gamma.T*self.flat_values).sum(axis=1))/denom
            #self.means=(1/np.exp(denom))*(np.exp(gamma.T)*self.flat_values).sum(axis=1)
            #self.variances=((gamma.T*((self.flat_values-self.means[:,None])*(self.flat_values-self.means[:,None]).T).sum(axis=0)))/denom
            self.variances=(gamma.T*((self.flat_values-self.means[:,None])**2)).sum(axis=1)/denom
           
            new_count,converged=default_convergence(cur_likelihood,self.likelihood(),count)
            count=new_count
            j=j+1


            
    
    def segment(self):
        """
        Using the trained model, 
        segment the image matrix into
        the pre-specified number of 
        components. Returns the original 
        image matrix with the each 
        pixel's intensity replaced 
        with its max-likelihood 
        component mean.
        
        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # TODO: finish this
        num=[]
        for i in range(self.num_components):
            num.append(-.5*np.log(2*np.pi*self.variances[i])-(((self.flat_values-self.means[i])**2)/(2*self.variances[i])))
        num=np.array(num)
        seg_index=num.argmax(axis=0)
        for i in range(self.num_components):
            self.flat_values[seg_index==i]=self.means[i]
        
        segment=unflatten_image_matrix(self.flat_values,self.width)
        
        return segment
    
    def likelihood(self):
        """Assign a log 
        likelihood to the trained
        model based on the following 
        formula for posterior probability:
        ln(Pr(X | mixing, mean, stdev)) = sum((n=1 to N),ln(sum((k=1 to K), mixing_k * N(x_n | mean_k, stdev_k) )))
        
        returns:
        log_likelihood = float [0,1]
        """
        # TODO: finish this
        log_list=[]
        i=0
        for i in range(self.num_components):
            #log_list.append(scipy.stats.lognorm.logpdf(self.flat_values,np.sqrt(self.variances[i]),loc=0,scale=np.exp(self.means[i])))
            log_list.append((-.5*np.log(2*np.pi*self.variances[i])-(((self.flat_values-self.means[i])**2)/2*self.variances[i])))
        log_list=np.array(log_list).T
        log_likelihood= sum(np.log((np.exp(self.mixing_coefficients)*np.exp(log_list)).sum(axis=1)))
             #log_list.append(scipy.stats.norm.pdf(self.flat_values,loc=self.means[i],scale=self.variances[i]))
        #log_list=np.array(log_list).T
        #log_likelihood=sum(np.log((self.mixing_coefficients*log_list).sum(axis=1)))
        return log_likelihood
        
        
        
    def best_segment(self, iters):
        """Determine the best segmentation
        of the image by repeatedly 
        training the model and 
        calculating its likelihood. 
        Return the segment with the
        highest likelihood.
        
        params:
        iters = int
        
        returns:
        segment = numpy.ndarray[numpy.ndarray[float]]
        """
        # finish this
        self.train_model()
        baseline=self.likelihood()
        best_seg=self.segment()
        
        for i in range(iters):
            self.initialize_training()
            self.train_model()
            likelihood_new=self.likelihood()
            if likelihood_new>baseline:
                baseline=likelihood_new
                best_seg=self.segment()
                      
        return best_seg

class GaussianMixtureModelImproved(GaussianMixtureModel):
    """A Gaussian mixture model
    for a provided grayscale image,
    with improved training
    performance."""

    def initialize_training(self):
        """
        Initialize the training
        process by setting each
        component mean using some algorithm that you think might give better means to start with,
        each component variance to 1, and
        each component mixing coefficient
        to a uniform value 
        (e.g. 4 components -> [0.25,0.25,0.25,0.25]).
        [You can feel free to modify the variance and mixing coefficient initializations too if that works well.]
        """
        self.variances=np.array(self.num_components*[1])
        ##set the mean to a random val in the data set
        initial_means_index= np.array(sample(range(len(self.flat_values)),self.num_components))
        #self.means=np.array(self.flat_values[initial_means_index])
        self.means=np.unique(k_means_cluster(self.image_matrix, self.num_components))
        
        self.mixing_coefficients = np.array(self.num_components*[1/float(self.num_components)])
        self.mixing_coefficients=np.log(self.mixing_coefficients)

def new_convergence_function(previous_variables, new_variables, conv_ctr, conv_ctr_cap=10):
    """
    Convergence function
    based on parameters:
    when all variables vary by
    less than 10% from the previous
    iteration's variables, increase
    the convergence counter.

    params:

    previous_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    new_variables = [numpy.ndarray[float]] containing [means, variances, mixing_coefficients]
    conv_ctr = int
    conv_ctr_cap = int

    return:
    conv_ctr = int
    converged = boolean
    """
    # TODO: finish this function
    increase_convergence_ctr=[]
    for i in range(len(previous_variables)):
        
        increase_convergence_ctr.append((abs(previous_variables[i]) * 0.9 < 
                                        abs(new_variables[i]))&(
                                        (abs(previous_variables[i]) * 1.1)>abs(new_variables[i])))
    
        if increase_convergence_ctr[i].all():
            increase_convergence_ctr[i]=1
        else:
            increase_convergence_ctr[i]=0
    if sum(increase_convergence_ctr)==len(previous_variables):
        conv_ctr+=1
        
    
    return conv_ctr, conv_ctr > conv_ctr_cap
    return conv_ctr, converged

class GaussianMixtureModelConvergence(GaussianMixtureModel):
    """
    Class to test the
    new convergence function
    in the same GMM model as
    before.
    """

    def train_model(self, convergence_function=new_convergence_function):
        # TODO: finish this function
        converged=False
        j=0
        count=0
        N=float(len(self.flat_values))
        while converged==False and j<100:
            num=[]
            cur_likelihood=self.likelihood()
            cur_means=self.means[:]
            cur_var=self.variances[:]
            cur_mix=np.exp(self.mixing_coefficients[:])
            init_vars=np.array([cur_means,cur_var,cur_mix])
            
            for i in range(self.num_components):
                num.append(-.5*np.log(2*np.pi*self.variances[i])-(((self.flat_values-self.means[i])**2)/(2*self.variances[i])))               
            num=np.array(num).T
            mixing_numerator=np.exp(self.mixing_coefficients)*np.exp(num)
            gamma=mixing_numerator/(np.exp(self.mixing_coefficients)*np.exp(num)).sum(axis=1)[:,None]
            denom=sum(gamma)
            
            self.mixing_coefficients=denom/N
            self.means=((gamma.T*self.flat_values).sum(axis=1))/denom
            self.variances=(gamma.T*((self.flat_values-self.means[:,None])**2)).sum(axis=1)/denom
            new_vars=np.array([self.means,self.variances,np.exp(self.mixing_coefficients)])
            
            count,converged=new_convergence_function(init_vars,new_vars,count)
            j+=1

def bayes_info_criterion(gmm):
    # TODO: finish this function
    k=3*gmm.num_components
    ll = gmm.likelihood()
    n=len(gmm.flat_values)
    BIC= -2 * ll + np.log(n) *k
    return BIC

def BIC_likelihood_model_test():
    """Test to compare the 
    models with the lowest BIC
    and the highest likelihood.
    
    returns:
    min_BIC_model = GaussianMixtureModel
    max_likelihood_model = GaussianMixtureModel
    """
    # TODO: finish this method
    
    comp_means = [
        [0.023529412, 0.1254902],
        [0.023529412, 0.1254902, 0.20392157],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563],
        [0.023529412, 0.1254902, 0.20392157, 0.36078432, 0.59215689, 0.71372563, 0.964706]
    ]
    image_file = 'images/party_spock.png'
    image_matrix = image_to_matrix(image_file)

    
    # first test original model
    max_likelihood_model = GaussianMixtureModel(image_matrix, len(comp_means[0]))
    max_likelihood_model.initialize_training()
    max_likelihood_model.means = np.copy(comp_means[0])
    max_likelihood_model.train_model()
    max_like=max_likelihood_model.likelihood()
    
    min_BIC_model=GaussianMixtureModel(image_matrix, len(comp_means[0]))
    min_BIC_model.initialize_training()
    min_BIC_model.means = np.copy(comp_means[0])
    min_BIC_model.train_model()
    BIC=bayes_info_criterion(min_BIC_model)
    
    
    for i in range(1,len(comp_means)):
        
        gmm=GaussianMixtureModel(image_matrix,len(comp_means[i]))
        gmm.initialize_training()
        gmm.means=np.copy(comp_means[i])
        gmm.train_model()
        
        if bayes_info_criterion(gmm)<BIC:
            min_BIC_model=gmm
            BIC=bayes_info_criterion(gmm)
        if gmm.likelihood()>max_like:
            max_likelihood_model=gmm
            max_like=gmm.likelihood()
            
        
    
    return min_BIC_model, max_likelihood_model


def BIC_likelihood_question():
    """
    Choose the best number of
    components for each metric
    (min BIC and maximum likelihood).

    returns:
    pairs = dict
    """
    # TODO: fill in bic and likelihood
    bic = 7
    likelihood = 7
    pairs = {
        'BIC' : bic,
        'likelihood' : likelihood
    }
    return pairs


