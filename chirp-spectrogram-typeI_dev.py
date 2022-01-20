# Short-time Linear Chirp Transform Enhanced Spectrogram dev 
# aka. Type I Joint-Chirp-Rate-Time-Frequency Transform (Type I JCTFT)
# Dev version, not final
# Linear Chirp Transform (LCT) reference - O, A, Alkaishriwo & L.F. Chaparro (2012)

import numpy as np
import scipy.signal
import scipy
import matplotlib.pyplot as plt
import TFchirp

rate = 4096                         # sampling frequency 
dt = 1/rate                         # time interval between sample points
N = rate * 5                        # in seconds
n = np.arange(0, N)                 # time bin array (bin number)
t = n*dt                            # time array (actural time value)

x1 = np.exp(1j*2*np.pi*(20*t**2))   # wave component 1
x2 = np.exp(1j*2*np.pi*(10*t**2))   # wave component 2

noise = np.random.rand(N)           # Gaussian noise with length N
noise_amp = 5                       # noise amplitude

ts = x1 + x2 + noise*noise_amp      # combined signal

# Display the time series
plt.plot(ts)
plt.title('2-component Linear Chirp Signal')
plt.xlabel('Time bin #')
plt.ylabel('Amplitude')
plt.show()

# DLCT Discrete Linear Chirp Transform reference - O, A, Alkaishriwo & L.F. Chaparro (2012)
L = 50; C = 0.001                   # chirp-rate parameters;

def dlct(ts, fs, C, L, frange=[0, 500], padding=False):
    '''Discrete Linear Chirp Transform
    Input:
                    ts              signal
                    fs              sample rate
                    C               chirp-rate scaling constant
                    L               chirp-rate parameter        
                    frange          frequency range; [0, 500] by default
                    padding         0 padding
    Output:
                    table           chirp-rate frequency distribution table
    Note:
                    chirp-rate parameter beta = C*L
    '''
    
    if padding:
        pad = fs-len(ts)
        ts = np.concatenate((ts, np.zeros(pad)))

    length = len(ts)
    n = np.arange(0, length)            # time bin array
    if fs//2 <= frange[1]:              # determine the upper frequency limit
        frange[1] = fs//2               # Preliminary: assuming ts => 1s in length
    
    table = np.zeros((frange[1]-frange[0], L))

    for l in range(0, L):               # go through all positive chirp-rates
        hs = ts*np.exp(-1j*2*np.pi*C*l*n**2/length)   
        table[:, l] = np.fft.fft(hs)[int(frange[0]):int(frange[1])].real          # desired frequency range

    return table

# Short Time Discrete Linear Chirp Transform with Thresholding
def stlct(ts, fs, C, L, N, frange=[0, 500], padding=True, overlap=None, percentile_l=[99, 100, 0, 0]):
    '''Short Time Linear Chirp Transform
    Input:  
                    x               time series
                    fs              sample rate
                    C               chirp-rate constant 
                    l               chirp-rate component
                    N               bin size
                    frange          frequency range; default set to [0, 500]
                    padding         0 padding; default set to True
                    overlap         number of data points to overlap
                    percentile_l    lower percentile cut-off
    Output: 
                    table           the result table of the short time linear chirp transform
    '''
    
    if padding:
        pad = N - len(ts)%N
        ts = np.concatenate((ts, np.zeros(pad)))
    length = len(ts)
    lap = N
    if overlap is not None:
        lap = int(N-overlap)                    
  
    if length <= frange[1]:
        frange[1] = length//2                   # determine the upper frequency

    table = np.zeros((frange[1]-frange[0], (length-N)//lap+1))          # shape (frequency, time)
    win = np.ones(N)                                                    # will add window function

    for i in range(table.shape[1]):                                     # for i in range time segments
        ts_bin = ts[i*lap:i*lap+N]*win
        dlct_table = dlct(ts_bin, fs, C, L, frange=frange, padding=True)
       
        masked_1 = dlct_table[frange[0]:frange[1]]                      # freq cut-off
        if frange[0] <= 10:
            masked_1[:10-frange[0]] = 0                                 # block low freq noise 
        # first percentile thresholding
        percentile_lower1 = np.percentile(masked_1, percentile_l[0])
        masked_1[masked_1<percentile_lower1]=0
        percentile_higher1 = np.percentile(masked_1, percentile_l[1])
        masked_1[masked_1>percentile_higher1]=0
        masked_2 = masked_1                                             # another thresholding position
        # sum
        chirp_rate_sum = np.sum(masked_2, 1)
        # second percentile thresholding
        percentile_lower2 = np.percentile(chirp_rate_sum, percentile_l[2])
        chirp_rate_sum[chirp_rate_sum<percentile_lower2]=0
        table[:, i] = chirp_rate_sum

    return table

def normalize(table, ax_origin=0, factor=1):
    '''Normalizes (Rescale) table values (Simple)
    Input:
                    table               table of values
                    ax_origin           the origin of the normalized value; default set to 0
                    factor              the scaling factor to normalization; default set to 1
    Output:
                    table/norm + ax_origin              the normalized table with shifted origin
    Note:
                    when ax_origin = 0, the range of value = [0,1]
                    when ax_origin = n, the range of value = [n,n+1]
    '''

    norm =  np.amax(table)
    if norm == 0:
        return table*factor + ax_origin

    return table/norm*factor + ax_origin

def enhance_factor(xs, a=0.5, norm=True, bias='limits'):
    '''Calculates the enhancement factor. Acts like a mask over STLCT result
    Input:
                    xs              input array
                    a               enhancement function tuning parameter
                    norm 
                    bias            'limits, center' where the distortion should focus on
    Output:
                    factor          enhancement factor array, same shape as the input
    Note:
                    Expected input range [0, 1] inclusive
    '''
    if norm:
        xs = normalize(xs, ax_origin=-0.5, factor=1)            # normalized range = [-.5, .5]

    denomi = np.tan((np.pi/2)*a)
    
    if bias == 'limits':
        factor = np.tan(a*np.pi*xs)/denomi + 1

        return factor
    else:
        xs_copy = np.copy(xs)
        xs_p = 0.5 - np.clip(xs_copy, 0, 1)
        factor_p =  2 - (np.tan(a*np.pi*xs_p)/denomi + 1)

        xs_n = -0.5 - np.clip(xs_copy, -1, 0)
        factor_n = 1 - (np.tan(a*np.pi*xs_n)/denomi + 1)
        factor = factor_p+factor_n

    return factor

bin_size = 1000
stlct_u = stlct(ts, rate, C, L, bin_size, overlap=(bin_size*1)//2, percentile_l=[75,100,0,0])

# Display the STLCT 
plt.imshow(abs(stlct_u), origin='lower', aspect='auto')
plt.title('STLCT Spectrogram')
plt.xlabel('Time bin #')
plt.ylabel('Frequency')
plt.show()

stlct_u_normalized = enhance_factor(stlct_u, a = 0.5, norm = True, bias='center')

# effects of different enhancement factor a
fig, axs = plt.subplots(2,2)
fig.set_size_inches(14.5, 10.5)
stlct_u_normalized00 = enhance_factor(stlct_u, a = 0.001, norm = True, bias='center')
im00=axs[0,0].imshow(abs(stlct_u_normalized00), origin='lower', aspect='auto')
axs[0,0].set_xlabel('Time bin number')
axs[0,0].set_ylabel('Frequency (Hz)')
axs[0,0].set_title(r'$f(x)$ with $a = 0.001$')
fig.colorbar(im00, ax=axs[0,0])

stlct_u_normalized01 = enhance_factor(stlct_u, a = 0.5, norm = True, bias='center')
im01=axs[0,1].imshow(abs(stlct_u_normalized01), origin='lower', aspect='auto')
axs[0,1].set_xlabel('Time bin number')
axs[0,1].set_ylabel('Frequency (Hz)')
axs[0,1].set_title(r'$f(x)$ with $a = 0.5$')
fig.colorbar(im01, ax=axs[0,1])

stlct_u_normalized10 = enhance_factor(stlct_u, a = 0.75, norm = True, bias='center')
im10=axs[1,0].imshow(abs(stlct_u_normalized10), origin='lower', aspect='auto')
axs[1,0].set_xlabel('Time bin number')
axs[1,0].set_ylabel('Frequency (Hz)')
axs[1,0].set_title(r'$f(x)$ with $a = 0.75$')
fig.colorbar(im10, ax=axs[1,0])

stlct_u_normalized11 = enhance_factor(stlct_u, a = 0.99, norm = True, bias='center')
im11=axs[1,1].imshow(abs(stlct_u_normalized11), origin='lower', aspect='auto')
axs[1,1].set_xlabel('Time bin number')
axs[1,1].set_ylabel('Frequency (Hz)')
axs[1,1].set_title(r'$f(x)$ with $a = 0.99$')
fig.colorbar(im11, ax=axs[1,1])
plt.show()

# S transform 

# 0 padding; to match the time scale of STLCT as padding was necessary in STLCT
ts_copy = np.copy(ts)
ts_copy = np.concatenate((ts_copy, np.zeros(bin_size - len(ts)%bin_size)))

spectrogram = TFchirp.sTransform(ts, sample_rate=rate, frange=[0, 499])         # need to redefine S transform freq range (currently: 499+1 = 500)
spectrogram = abs(spectrogram)

plt.imshow(spectrogram, origin='lower', aspect='auto')
plt.title('S Transform Spectrogram')
plt.xlabel('Time bin #')
plt.ylabel('Frequency bin #')
plt.show()

# STLCT interpolation
stlct_u_interpolated = scipy.ndimage.zoom(stlct_u_normalized, 
                                            (spectrogram.shape[0]/stlct_u_normalized.shape[0],
                                                spectrogram.shape[1]/stlct_u_normalized.shape[1]), order=2)

# enhance
enhanced = spectrogram*stlct_u_interpolated
plt.imshow(enhanced, origin='lower', aspect='auto')
plt.title('Enhanced')
plt.colorbar()
plt.clim(np.average(enhanced),1.80)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()

# Short-time Fourier Transform (STFT) -  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html
f, t, stft_u = scipy.signal.stft(ts)
plt.imshow(abs(stft_u), origin='lower', aspect='auto')
plt.title('STFT Spectrogram')
plt.xlabel('Time bin #')
plt.ylabel('Frequency bin #')
plt.ylim(0, 20)
plt.show()