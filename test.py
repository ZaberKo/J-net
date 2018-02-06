import cntk as C
import numpy as np



in1 = C.input_variable((4,),name='a')
in2 = C.input_variable((4,),name='b')
in1_data = np.asarray([[1., 2., 3., 4.]], np.float32)
in2_data = np.asarray([[0., 5., -3., 2.]], np.float32)
val=C.combine([in1,in2]).eval({in1: in1_data, in2: in2_data})
print(val.value())
a,b=val
# print(a.value())
# print()
# print(b.values())


