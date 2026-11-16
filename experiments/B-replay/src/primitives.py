import numpy as np
C={"control_bits_per_real":16,"receiver_l2":.001,"adam_beta1":.9,"adam_beta2":.99,"adam_eps":1e-8}

def cn(rng,shape): return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)

def mv(a,b): return np.einsum('ij,bj->bi',a,b)

def pb(h,s): return np.einsum('bmn,bm->bn',h.conj(),s)

def pgrad(gx,a): return np.einsum('bn,bi->ni',gx,a)/len(a)

def qreal(v,bound,signed=True):
    levels=2**C['control_bits_per_real']-1
    low=-bound if signed else 0.
    clips=int(np.count_nonzero((v<low)|(v>bound)))
    q=np.rint((np.clip(v,low,bound)-low)/(bound-low)*levels)/levels*(bound-low)+low
    return q,clips

def task(y,lab,w,b):
    u=np.tanh(abs(y)**2)
    z=b+np.einsum('bi,i->b',u,w)
    prob=1/(1+np.exp(-z))
    err=prob-lab
    loss=np.logaddexp(0,z)-lab*z
    seed=2*err[:,None]*w[None,:]*(1-u*u)*y
    gw=np.einsum('b,bi->i',err,u)/len(y)+C['receiver_l2']*w
    gb=err.mean()
    return loss,seed,gw,gb,prob

class Adam:
    def __init__(self,shape,dtype): self.m=np.zeros(shape,dtype=dtype); self.v=np.zeros(shape); self.t=0
    def step(self,x,g,lr):
        self.t+=1
        self.m=C['adam_beta1']*self.m+(1-C['adam_beta1'])*g
        self.v=C['adam_beta2']*self.v+(1-C['adam_beta2'])*abs(g)**2
        return x-lr*(self.m/(1-C['adam_beta1']**self.t))/(np.sqrt(self.v/(1-C['adam_beta2']**self.t))+C['adam_eps'])
