from cntk.layers import *


def BiRecurrence(fwd, bwd):
    return Sequential([
        (Recurrence(fwd),
         Recurrence(bwd, go_backwards=True)),
        splice
    ])




