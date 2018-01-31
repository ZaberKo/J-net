import cntk as C
from cntk.layers import *


def BiRecurrence(fwd, bwd):
    Sequential([
        (Recurrence(fwd),
         Recurrence(bwd, go_backwards=True)),
        splice
    ])

    hidden_dim = 512
    num_layers = 2
    attention_dim = 128
    use_attention = True
    use_embedding = True
    embedding_dim = 200
    vocab = None  # all lines of vocab_file in a list
    length_increase = 1.5
    label_vocab_dim=None
# create the s2s model
    def create_model():  # :: (history*, input*) -> logP(w)*

        # Embedding: (input*) --> embedded_input*
        embed = C.layers.Embedding(embedding_dim, name='embed') if use_embedding else identity

        # Encoder: (input*) --> (h0, c0)
        # Create multiple layers of LSTMs by passing the output of the i-th layer
        # to the (i+1)th layer as its input
        # Note: We go_backwards for the plain model, but forward for the attention model.
        with C.layers.default_options(enable_self_stabilization=True, go_backwards=not use_attention):
            LastRecurrence = C.layers.Fold if not use_attention else C.layers.Recurrence
            encode = C.layers.Sequential([
                embed,
                C.layers.Stabilizer(),
                C.layers.For(range(num_layers - 1), lambda:
                C.layers.Recurrence(C.layers.LSTM(hidden_dim))),
                LastRecurrence(C.layers.LSTM(hidden_dim), return_full_state=True),
                (C.layers.Label('encoded_h'), C.layers.Label('encoded_c')),
            ])

        # Decoder: (history*, input*) --> unnormalized_word_logp*
        # where history is one of these, delayed by 1 step and <s> prepended:
        #  - training: labels
        #  - testing:  its own output hardmax(z) (greedy decoder)
        with C.layers.default_options(enable_self_stabilization=True):
            # sub-layers
            stab_in = C.layers.Stabilizer()
            rec_blocks = [C.layers.LSTM(hidden_dim) for i in range(num_layers)]
            stab_out = C.layers.Stabilizer()
            proj_out = C.layers.Dense(label_vocab_dim, name='out_proj')
            # attention model
            if use_attention:  # maps a decoder hidden state and all the encoder states into an augmented state
                attention_model = C.layers.AttentionModel(attention_dim,
                                                          name='attention_model')  # :: (h_enc*, h_dec) -> (h_dec augmented)

            # layer function
            @C.Function
            def decode(history, input):
                encoded_input = encode(input)
                r = history
                r = embed(r)
                r = stab_in(r)
                for i in range(num_layers):
                    rec_block = rec_blocks[i]  # LSTM(hidden_dim)  # :: (dh, dc, x) -> (h, c)
                    if use_attention:
                        if i == 0:
                            @C.Function
                            def lstm_with_attention(dh, dc, x):
                                h_att = attention_model(encoded_input.outputs[0], dh)
                                x = C.splice(x, h_att)
                                return rec_block(dh, dc, x)

                            r = C.layers.Recurrence(lstm_with_attention)(r)
                        else:
                            r = C.layers.Recurrence(rec_block)(r)
                    else:
                        # unlike Recurrence(), the RecurrenceFrom() layer takes the initial hidden state as a data input
                        r = C.layers.RecurrenceFrom(rec_block)(
                            *(encoded_input.outputs + (r,)))  # :: h0, c0, r -> h
                r = stab_out(r)
                r = proj_out(r)
                r = C.layers.Label('out_proj_out')(r)
                return r

        return decode
