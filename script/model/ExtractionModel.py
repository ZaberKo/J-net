import math
import numpy as np
import os


import cntk as C
import cntk.tests.test_utils
cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1)
def BiRecurrence(fwd, bwd):
    F1=C.layers.GRU(fwd)
    F2=C.layers.GRU(bwd)
    F = C.layers.Recurrence(F1)
    G = C.layers.Recurrence(F2, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(F(x), G(x))
    return apply_x

def word_to_vector(W):
    with C.layers.default_options(initial_state=0.1):
        W_to_V=C.layers.Sequential([
            C.layers.Embedding(EMB_DIM, name='embed'),
            BiRecurrence(EMB_DIM,EMB_DIM)
        ])
    return W_to_V

def U_passageToV_passage(U_question,U_passage):
    V_passage=C.layers.AttentionModel(len(U_passage),attention_span=None, attention_axis=U_question,
                   init=C.initializer.glorot_uniform(),
                   go_backwards=False,
                   enable_self_stabilization=True, name='V_passage')
    return V_passage

def  U_questionToV_question(U_question,All_U_passage):
    V_question= C.layers.AttentionModel(len(U_question), attention_span=None, attention_axis=All_U_passage,
                                        init=C.initializer.glorot_uniform(), go_backwards=False, enable_self_stabilization=True,
                                        name='V_question')
    return V_question


def attention(source, hidden, attention_dim):
    with C.layers.default_options(bias=None):
        s = C.layers.Dense(1, activation=C.tanh)(
                C.layers.Dense(attention_dim),  # for d_t-1
        )

        a = C.softmax(s)


def FindPosP1(All_V_Passage,NumOfAllPassageWord):
    a=attention(All_V_Passage,h0 ,NumOfAllPassageWord)
    P1=C.reduce_max(a)
    return P1

c1=C.reduce_sum(a,All_V_passage)
h1=C.layers.GRU(h0,c1)

def FindPosP2(All_V_Passage,NumOfAllPassageWord):
    a=attention(All_V_Passage,h1 ,NumOfAllPassageWord)
    P2=C.reduce_max(a)
    return P2

Lap

def V_questionToR_question(U_question,V_question):
    R_question=attention(U_question,V_question,len(V_question))
    return R_question

def V_passageToR_passage(All_U_passage,All_V_passage):
    R_passage=attention(All_U_passage,All_V_passage,len(All_V_passage))
    return R_passage

def PassageRanking(R_passage,R_question):
    g=C.tanh([R_question,R_passage])
    g_hat=C.softmax(g)

Lpr

Le=0.8Lap+0.2Lpr

