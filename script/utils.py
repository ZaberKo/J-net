from cntk.layers import *


def BiRecurrence(fwd, bwd,name=None):
    if name!=None:
        return Sequential([
            (Recurrence(fwd),
             Recurrence(bwd, go_backwards=True)),
            splice
        ], name=name)
    else:
        return Sequential([
            (Recurrence(fwd),
             Recurrence(bwd, go_backwards=True)),
            splice
        ])




