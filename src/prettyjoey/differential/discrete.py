def diff(t,x):
    """
    computes discrete derivative v(t) from a given timeseries x(t)
    following
    v(t) = x(tk) - x(tk-1) / (tk - tk-1)
    input 1: time value tk, k is index of time
    input 2: signal value x

    -must return v(t) as array
    -must check for equal length arrays
    """
    #same length?
    if len(x) != len(t):
        raise ValueError("input arrays must have the same length")
    #store results
    v = []
    #computation
    for k in range(1,len(t)):
        der_v = (x[k] - x[k-1]) / (t[k] - t[k-1])
        v.append(der_v) #append results to list previously created

    return v
