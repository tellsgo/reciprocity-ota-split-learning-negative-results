"""Algebraically identical float64/complex128 contractions; no model/grid change."""
import numpy as np
from itertools import product
PATTERNS=np.array(list(product([-1.,1.],repeat=3)))
class Backend:
 def __getattr__(self,name):return getattr(np,name)
 def einsum(self,subscripts,*args,**kwargs):
  if subscripts=='kni,bi->kbn' and not kwargs:
   W,inp=args
   if W.shape[-1]==3 and np.all((inp==-1)|(inp==1)):
    # There are exactly eight input vectors. Evaluate the same three-term sum once per vector.
    small=W[:,None,:,0]*PATTERNS[None,:,None,0]+W[:,None,:,1]*PATTERNS[None,:,None,1]+W[:,None,:,2]*PATTERNS[None,:,None,2]
    ix=((inp[:,0]>0)*4+(inp[:,1]>0)*2+(inp[:,2]>0)).astype(int)
    return small[:,ix,:]
  if subscripts=='bmn,kbn->kbm' and not kwargs:
   f,x=args
   # Batch b, multiply (k,n) by (n,m); identical contraction with ordinary double arithmetic.
   return np.matmul(x.transpose(1,0,2),f.transpose(0,2,1)).transpose(1,0,2)
  return np.einsum(subscripts,*args,**kwargs)
def install(module):module.np=Backend()
