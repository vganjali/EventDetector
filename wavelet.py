import numpy as np
from scipy.special import erf

def ricker(scale=10, N=1, pattern=None, window=1, mod=None, shift=None, skewness=None, dt=1):
    resolution = scale/dt
    print(resolution)
    length = int((10*window)*resolution)
    a = resolution/1.25187536
    t = np.arange(length)
    s = 2/(np.sqrt(3*a)*np.pi**1/4)*(1-(t-length/2)**2/a**2)\
        *np.exp(-(t-length/2)**2/(2*a**2))
    s_square_norm = np.trapz(s**2, dx=1)
    s -= np.mean(s)
    return s/np.sqrt(s_square_norm)

def morlet(scale=10, N=6, pattern=None, window=1, mod=None, shift=None, skewness=None, is_complex=False, dt=1):
    resolution = scale/dt
    length = int(2*(N+4)*window*resolution)
    t = np.arange(length)
    sigma = length/(10*window)
    s_exp = np.exp(-(t-length/2)**2/(2*sigma**2))
    if (is_complex):
        s_sin = np.exp(1j*(2*np.pi/resolution*(t-length/2)-np.pi*(0.75-N%2)))
    else:
        s_sin = np.sin((2*np.pi/resolution*(t-length/2)-np.pi*(0.5-N%2)))
    s = s_exp*s_sin
    s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    return s/np.sqrt(s_square_norm)

def morlet_complex(scale=10, N=6, pattern=None, window=1, mod=None, shift=None, skewness=None, is_complex=True, dt=1):
    resolution = scale/dt
    length = int(2*(N+4)*window*resolution)
    t = np.arange(length)
    sigma = length/(10*window)
    s_exp = np.exp(-(t-length/2)**2/(2*sigma**2))
    if (is_complex):
        s_sin = np.exp(1j*(2*np.pi/resolution*(t-length/2)-np.pi*(0.75-N%2)))
    else:
        s_sin = np.sin((2*np.pi/resolution*(t-length/2)-np.pi*(0.5-N%2)))
    s = s_exp*s_sin
    s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    return s/np.sqrt(s_square_norm)

def skew_normal(x, mu, sigma, alpha=0):
    # mean = mu - sigma*alpha/np.sqrt(1+alpha**2)*np.sqrt(2/np.pi)
    delta = alpha/(np.sqrt(1+alpha**2))
    mu_z = np.sqrt(2/np.pi)*delta
    sigma_z = np.sqrt(1-mu_z**2)
    gamma_1 = (4-np.pi)/2*(delta*np.sqrt(2/np.pi))**3/((1-2*delta**2/np.pi)**(3/2))
    if alpha == 0:
        m_0 = 0
    else:
        m_0 = mu_z - gamma_1*sigma_z/2 - np.sign(alpha)/2*np.exp(-2*np.pi/np.abs(alpha))
    mode = mu + sigma*m_0
    xi = mu - sigma*m_0
    phi = 1/np.sqrt(2*np.pi)*np.exp(-((x-xi)**2)/(2*sigma**2))
    _PHI = 1/2*(1+erf(alpha*(x-xi)/sigma/np.sqrt(2)))
    return 2/sigma*phi*_PHI

def msg(scale=10, N=6, pattern='6', window=1, mplx_ratio=1, weight=1, mod=0.5, shift=1, skewness=1, is_complex=False, dt=1, mf=False):
    N = int(N)
    if (type(N) != list):
        N = [N]
    if ((type(mplx_ratio) == float) or (type(mplx_ratio) == int)):
        mplx_ratio = [mplx_ratio]*len(N)
    elif (len(mplx_ratio) != len(N)):
        print('multiplex ratio should have same length as N')
        return
    if ((type(weight) == float) or (type(weight) == int)):
        weight = [weight]*max(N)
    elif (len(weight) != N):
        print('weight ratio should have a length equal to N')
        return
    N_max = max(N)
#     mod *= (4.29193/2.35482)
    skewness *= 1.2*mod
    resolution = scale/dt
    sigma = resolution/4.29193*mod*N_max/8
#     amp = np.array([3,4,3,2,4,3,3,3])
#     amp = amp[::-1]/np.sum(amp)*N_max
    # sigma = resolution/4*mod
    length = int(N_max*resolution + shift*resolution + (1+4*skewness)*5*N_max*window*sigma)
    t = np.arange(length)
    if is_complex:
        s = np.zeros((length,),dtype=np.complex)
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*weight[m]*skew_normal(t,length/2+((n-1)/2-m+0.125)*(N_max/n)*resolution,sigma, alpha=0)
                s += 1j*mplx_ratio[i]*skew_normal(t,length/2+((n-1)/2-m-0.125)*(N_max/n)*resolution,sigma, alpha=0)
        s -= np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1j*np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= 1j*np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
    else:
        s = np.zeros((length,))
        for i,n in enumerate(N):
            for m in range(n):
                s += mplx_ratio[i]*weight[m]*skew_normal(t,length/2-((n-1)/2-m)*(N_max/n)*resolution,sigma, alpha=0)
    if not mf:
        s -= np.sum(weight)/2*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(weight)/2*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
#     s = s/np.sqrt(scale)
    return s

def msg_encoded(scale=10, N=4, pattern='1201', window=1, mod=0.5, shift=1, skewness=1, is_complex=False, dt=1, mf=False):
    if (type(N) != list):
        N = [N]
    if (type(pattern) != list):
        pattern = [pattern]
    for i,n in enumerate(pattern):
        pattern[i] = [float.fromhex('0x'+p) for p in pattern[i]]
    # if (len(N) < 2):
    #     print('Please define code by Hex numbers: i.e. 1a30e')
    #     return
    N_max = max(N)
#     mod *= (4.29193/2.35482)
    skewness *= 1.2*mod
    resolution = scale/dt
    sigma = resolution/4.29193*mod*N_max/8
#     amp = np.array([3,4,3,2,4,3,3,3])
#     amp = amp[::-1]/np.sum(amp)*N_max
    # sigma = resolution/4*mod
    length = int(N_max*resolution + shift*resolution + (1+4*skewness)*5*N_max*window*sigma)
    t = np.arange(length)
    if is_complex:
        s = np.zeros((length,),dtype=np.complex)
        for i,n in enumerate(N):
            for m in range(n):
                s += float.fromhex('0x'+n[m])*skew_normal(t,length/2+((n-1)/2-m+0.125)*(N_max/n)*resolution,sigma, alpha=0)
                s += 1j*skew_normal(t,length/2+((n-1)/2-m-0.125)*(N_max/n)*resolution,sigma, alpha=0)
        s -= 1/2*skew_normal(t, length/2+(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1j/2*skew_normal(t, length/2+(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= 1/2*skew_normal(t, length/2-(N_max/2+shift/2-0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= 1j/2*skew_normal(t, length/2-(N_max/2+shift/2+0.125)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
    else:
        s = np.zeros((length,))
        for i,n in enumerate(N):
            for m in range(n):
                s += pattern[i][m]*skew_normal(t,length/2-((n-1)/2-m)*(N_max/n)*resolution,sigma, alpha=0)
    if not mf:
        s -= np.sum(pattern)/2*skew_normal(t, length/2+(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=4*skewness*N_max/2*window)
        s -= np.sum(pattern)/2*skew_normal(t, length/2-(N_max/2+shift/2)*resolution, (1+4*skewness)*N_max/2*window*sigma, alpha=-4*skewness*N_max/2*window)
        s -= np.mean(s)
    s_square_norm = np.trapz(np.abs(s)**2, dx=1)
    s = s/np.sqrt(s_square_norm)
#     s = s/np.sqrt(scale)
    return s
