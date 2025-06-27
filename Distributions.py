import numpy as np
import torch


def sample_stdentt(m, S, df=np.inf, n=1):

    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return torch.tensor(m + z/np.sqrt(x)[:,None])   # same output format as random.multivariate_normal

def samplestudentT_4(mean,dof,N):
    K=0
    S1=sample_stdentt([0+K,mean+K],[[1,0],[0,1]],dof,N//4)
    S2=sample_stdentt([0+K,-1*mean+K],[[1,0],[0,1]],dof,N//4)
    S3=sample_stdentt([mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    S4=sample_stdentt([-1*mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    sample=torch.cat([S1,S2,S3,S4],0)
    return(sample)

def normalise(x): # n x 2
    x=(x-x.mean(0).unsqueeze(0))/x.std(0).unsqueeze(0)
    return(x)

def sample_funnel(n):
    x1=torch.normal(0, math.sqrt(3), size=(1, n))
    x2=torch.normal(x1*0,np.sqrt(np.exp(x1/2)))
    sample=torch.cat([x2,x1],0).T
    sample=normalise(sample)
    return(sample)