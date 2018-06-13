import numpy as np

def NumericalGradient(X_inputs, Ys, ConvNet, h):

    Gs= []

    for l in range(len(ConvNet)):

        try_convNet_Fl= np.copy(ConvNet[l])
        Gs_l= np.zeros(ConvNet[i])
        nf= ConvNet[l].shape[2]

        for i in range(nf):

            try_convNet_Fl = np.copy(ConvNet[l])
            F_try= np.squeeze(F_l_copy[:, :, i])

            G= np.zeros(shape=(F_try.size, 1))

            for j in range(F_try.size):

                F_try1= np.copy(F_try)
                F_try1[j]= F_try[j] - h
                try_convNet_Fl[:, :, i]= F_try1





