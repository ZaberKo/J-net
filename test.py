import cntk as C
import numpy as np
a=C.placeholder(1,name='rrr')
def M1():
    return C.layers.identity(a)



M1().eval({a:np.asarray([4],dtype=np.float32)})