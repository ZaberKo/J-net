import cntk as C
from cntk.layers import *


def BiRecurrence(fwd, bwd):
    Sequential([
        (Recurrence(fwd),
         Recurrence(bwd, go_backwards=True)),
        splice
    ])