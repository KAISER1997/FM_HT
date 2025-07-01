import numpy as np
import torch

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch import Chi2
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class MultivariateStudentT(TorchDistribution):
    """
    Creates a multivariate Student's t-distribution parameterized by degree of
    freedom :attr:`df`, mean :attr:`loc` and scale :attr:`scale_tril`.

    :param ~torch.Tensor df: degrees of freedom
    :param ~torch.Tensor loc: mean of the distribution
    :param ~torch.Tensor scale_tril: scale of the distribution, which is
        a lower triangular matrix with positive diagonal entries
    """

    arg_constraints = {
        "df": constraints.positive,
        "loc": constraints.real_vector,
        "scale_tril": constraints.lower_cholesky,
    }
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, df, loc, scale_tril, validate_args=None):
        dim = loc.size(-1)
        assert scale_tril.shape[-2:] == (dim, dim)
        if not isinstance(df, torch.Tensor):
            df = loc.new_tensor(df)
        batch_shape = torch.broadcast_shapes(
            df.shape, loc.shape[:-1], scale_tril.shape[:-2]
        )
        event_shape = torch.Size((dim,))
        self.df = df.expand(batch_shape)
        self.loc = loc.expand(batch_shape + event_shape)
        self._unbroadcasted_scale_tril = scale_tril
        self._chi2 = Chi2(self.df)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @lazy_property
    def scale_tril(self):
        return self._unbroadcasted_scale_tril.expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @lazy_property
    def covariance_matrix(self):
        # NB: this is not covariance of this distribution;
        # the actual covariance is df / (df - 2) * covariance_matrix
        return torch.matmul(
            self._unbroadcasted_scale_tril,
            self._unbroadcasted_scale_tril.transpose(-1, -2),
        ).expand(self._batch_shape + self._event_shape + self._event_shape)

    @lazy_property
    def precision_matrix(self):
        identity = torch.eye(
            self.loc.size(-1), device=self.loc.device, dtype=self.loc.dtype
        )
        return torch.cholesky_solve(identity, self._unbroadcasted_scale_tril).expand(
            self._batch_shape + self._event_shape + self._event_shape
        )

    @staticmethod
    def infer_shapes(df, loc, scale_tril):
        event_shape = loc[-1:]
        batch_shape = broadcast_shape(df, loc[:-1], scale_tril[:-2])
        return batch_shape, event_shape


    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateStudentT, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        scale_shape = loc_shape + self.event_shape
        new.df = self.df.expand(batch_shape)
        new.loc = self.loc.expand(loc_shape)
        new._unbroadcasted_scale_tril = self._unbroadcasted_scale_tril
        if "scale_tril" in self.__dict__:
            new.scale_tril = self.scale_tril.expand(scale_shape)
        if "covariance_matrix" in self.__dict__:
            new.covariance_matrix = self.covariance_matrix.expand(scale_shape)
        if "precision_matrix" in self.__dict__:
            new.precision_matrix = self.precision_matrix.expand(scale_shape)
        new._chi2 = self._chi2.expand(batch_shape)
        super(MultivariateStudentT, new).__init__(
            batch_shape, self.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new


    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        X = torch.empty(shape, dtype=self.df.dtype, device=self.df.device).normal_()
        Z = self._chi2.rsample(sample_shape)
        Y = X * torch.rsqrt(Z / self.df).unsqueeze(-1)
        return self.loc + self.scale_tril.matmul(Y.unsqueeze(-1)).squeeze(-1)


    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        n = self.loc.size(-1)
        y = torch.linalg.solve_triangular(
            self.scale_tril, (value - self.loc).unsqueeze(-1), upper=False
        ).squeeze(-1)
        Z = (
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
            + 0.5 * n * self.df.log()
            + 0.5 * n * math.log(math.pi)
            + torch.lgamma(0.5 * self.df)
            - torch.lgamma(0.5 * (self.df + n))
        )
        return -0.5 * (self.df + n) * torch.log1p(y.pow(2).sum(-1) / self.df) - Z


    @property
    def mean(self):
        m = self.loc.clone()
        m[self.df <= 1, :] = float("nan")
        return m

    @property
    def variance(self):
        m = self.scale_tril.pow(2).sum(-1) * (self.df / (self.df - 2)).unsqueeze(-1)
        m[(self.df <= 2) & (self.df > 1), :] = float("inf")
        m[self.df <= 1, :] = float("nan")
        return m






def sample_stdentt(m, S, df=np.inf, n=1):
    ob=MultivariateStudentT(torch.tensor(df).float(),torch.tensor(m).float(),torch.tensor(S).float())
    samplesz=ob.rsample(torch.zeros(n).float().shape)
    return(samplesz,ob)





def samplestudentT_4(mean,dof,N):
    K=0
    S1,ob1=sample_stdentt([0+K,mean+K],[[1,0],[0,1]],dof,N//4)
    S2,ob2=sample_stdentt([0+K,-1*mean+K],[[1,0],[0,1]],dof,N//4)
    S3,ob3=sample_stdentt([mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    S4,ob4=sample_stdentt([-1*mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    sample=torch.cat([S1,S2,S3,S4],0)
    ob=[ob1,ob2,ob3,ob4]
    return(sample,ob)


def samplestudentT_4V2(mean,dof,N):
    K=0
    S1,ob1=sample_stdentt([0+K,mean+K],[[1,0],[0,1]],dof,N//4)
    S2,ob2=sample_stdentt([0+K,-1*mean+K],[[1,0],[0,1]],dof,N//4)
    S3,ob3=sample_stdentt([mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    S4,ob4=sample_stdentt([-1*mean+K,0+K],[[1,0],[0,1]],dof,N//4)
    sample=torch.cat([S1,S2,S3,S4],0)
    ob=[ob1,ob2,ob3,ob4]
    return(sample,ob)





def studentT_4_likelihood(st_object,samples):
    total=0
    for i in st_object:
        prob=torch.exp(i.log_prob(torch.tensor(samples)))
        total=total+prob
    likelhd=torch.log(total/4)
    return(likelhd)
    

def normalise(x): # n x 2
    x=(x-x.mean(0).unsqueeze(0))/x.std(0).unsqueeze(0)
    return(x)




def sample_funnel(n,normalises=True):
    x1=torch.normal(0, (3), size=(1, n))
    x2=torch.normal(x1*0,(np.exp(x1/2)))
    sample=torch.cat([x2,x1],0).T
    if normalises:
        sample=(sample-sf_mean)/sf_std
    return(sample)


df=sample_funnel(1000000,False)
sf_mean=df.mean(0).unsqueeze(0)
sf_std=df.std(0).unsqueeze(0)
