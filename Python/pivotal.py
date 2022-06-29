import numpy as np

# helper function -- marches through a list of probabilities and
# rounds them to 0 or 1
def pivotal(x):
    y = np.copy(x)
    i = 0
    j = 1
    k = 2
    d = y.size
    a = y[i]
    b = y[j]
    while k < d:
        if k < d and (a < 1e-12 or a > 1 - 1e-12):
            a = y[k]
            i = k
            k += 1
        if k < d and (b < 1e-12 or b > 1 - 1e-12):
            b = y[k]
            j = k
            k += 1
        u = np.random.rand()
        add = a + b
        if (add > 1) and (add < 2):
            if u < (1 - b)/(2 - add):
                b = add - 1
                a = 1
            else:
                a = add - 1
                b = 1
        elif (add > 0) and (add <= 1):
            if u < b / add:
                b = add
                a = 0
            else:
                a = add
                b = 0
        y[i] = a
        y[j] = b
    return(y)