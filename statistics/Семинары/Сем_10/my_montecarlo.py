import numpy
from typing import Iterable
from scipy import stats
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportion_confint

def run_montecarloAB(
    # the tested criterion (must accept 2dimensional numpy arrays)
    get_pvalues_2sample=None,
    
    latent_distrA=stats.norm(),
    latent_distrB=stats.norm(), 
    
    zeros_shareA=0,
    zeros_shareB=0,
    
    sample_size: int=20,
    sample_sizeA=None,
    sample_sizeB=None,
    alpha: float=0.05,
    conf_int_alpha: float=0.05,
    n_tests: int=100000,
    rseed: int=1337
) -> tuple:
    """Returns the amount of negatives, positives, conf_intL, conf_intR and the p-values in a montecarloed series of AB tests"""
        
    def student(A, B):
        t_stats, p_values, _ = ttest_ind(B, A, alternative='two-sided', usevar='unequal')
        p_values_cleaned = numpy.where(
            numpy.isnan(p_values), 
            0, 
            p_values
        )        
        return p_values_cleaned
    
    if get_pvalues_2sample is None:
        get_pvalues_2sample = student
        
    if sample_sizeA is None:
        sample_sizeA = sample_size
        
    if sample_sizeB is None:
        sample_sizeB = sample_size
    
    
    numpy.random.seed(rseed)      
    
    #                             n_rows,      n_columns
    samplesA = latent_distrA.rvs([sample_sizeA, n_tests]) 
    samplesB = latent_distrB.rvs([sample_sizeB, n_tests])

    if zeros_shareA > 0:
        samplesA *= stats.bernoulli(1 - zeros_shareA).rvs([sample_sizeA, n_tests]) 

    if zeros_shareB > 0:
        samplesB *= stats.bernoulli(1 - zeros_shareB).rvs([sample_sizeB, n_tests]) 
    
    # columns are treated as samples
    p_values = get_pvalues_2sample(A=samplesA, B=samplesB)    
    n_positives = sum(p_values <= alpha)  
    
    conf_intL, conf_intR = proportion_confint(count=n_positives, nobs=n_tests, alpha=conf_int_alpha, method='wilson')    
    
    return(
        n_tests - n_positives, 
        n_positives, 
        conf_intL,
        conf_intR,
        p_values
    )