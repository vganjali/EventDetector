import numpy as np

def mexican_hat(scale, window=1, dt=1):
    resolution = scale/dt
    length = int((10*window)*resolution)
    a = resolution/2
    t = np.arange(length)
    s = 2/(np.sqrt(3*a)*np.pi**1/4)*(1-(t-length/2)**2/a**2)\
        *np.exp(-(t-length/2)**2/(2*a**2))
    s_square_norm = np.trapz(s**2, dx=dt)
    s -= np.mean(s)
    return s/np.sqrt(s_square_norm)

def morlet(scale, N, window=1, is_complex=False, dt=1):
    resolution = scale/dt
    length = int(2*(N+2)*window*resolution)
    t = np.arange(length)
    sigma = length/(10*window)
    s_exp = np.exp(-(t-length/2)**2/(2*sigma**2))
    if (is_complex):
        s_sin = np.exp(1j*(2*np.pi/resolution*(t-length/2)-np.pi*(0.75-N%2)))
    else:
        s_sin = np.sin((2*np.pi/resolution*(t-length/2)-np.pi*(0.5-N%2)))
    s = s_exp*s_sin
    s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=dt)
    return s/np.sqrt(s_square_norm)

def mmi_gaussian(scale, N, window=1, weight=1, mod=0.5, shift=1, dt=1):
    resolution = scale/dt
    sigma = resolution/4*mod
    if (type(N) == list):
        if ((type(weight) == list) and (len(N) != len(weight))):
            print('weight and N lists should have equal length')
            return
        elif ((type(weight) == float) or (type(weight) == int)):
            weight = [weight]*len(N)
        N_max = max(N)
    else:
        N_max = N
    length = int(N_max*resolution + shift*resolution + 5*N_max*window*sigma)
    t = np.arange(length)
    s = np.zeros((length,))
    if (type(N) == list):
        for i,n in enumerate(N):
            for m in range(n):
                s += weight[i]*np.exp(-(t-length/2+((n-1)/2-m)*(N_max/n)*resolution)**2/(2*(sigma*N_max/n)**2))
    else:
        for m in range(N):
            s += weight*np.exp(-(t-length/2+((N_max-1)/2-m)*resolution)**2/(2*(sigma)**2))
    s -= np.amax(s)/window*np.exp(-(t-length/2+(N_max/2+shift/2)*resolution)**2/(2*(N_max/2*window*sigma)**2))
    s -= np.amax(s)/window*np.exp(-(t-length/2-(N_max/2+shift/2)*resolution)**2/(2*(N_max/2*window*sigma)**2))
    s -= np.mean(s)
    s_square_norm = np.trapz(s**2, dx=dt)
    s = s/np.sqrt(s_square_norm)
    return s
