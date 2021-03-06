# EventDetector
This package of scripts are developed to use with data recorded using PicoQuant TimeHarp-260P TCSPC instrument. It consists of three main parts:
- reading and converting photon time-tags binned data trace
- applying continuous wavelet transform with user defined mother wavelet(s)
- finding and selecting events above a preset threshold value

Continuous Wavelet Transform (CWT) are widely used to analyze temporal information with interested behaviors happening in different frequencies. Comparing to windowed Fourier transform, CWT utilizes varying window size such that provides a good balance between time and frequency/scale localization.
In cytometry, due to the fact that particles are flowing at different speeds depending on the location of particle in fluidic channel's cross-section, a multi-scale signal analysis approach is usually desired. CWT provide such tool, by taking a mother wavelet (can assume as a filter), scales it by compressing and stretching in time (while keeping the square norm value = 1) and convolving with the time trace. Whenever a localized enhancement is detected in time-scale graph, it markes it as detected and stores time and scale information into a HDF file. For the cases with multiple wavelets used for analysis, it compares the intensity of the enhancement for each and choose the strongest one indicating the highest similarity of the event with one of the wavelets. The HDF file includes detected events information as well as binned time traces.

# Reference
  This program was introduced in the paper:
  (will be great if you cite it if you find this program helpful in your study)
  
  1. [**Ganjalizadeh, V.**, G. G. Meena, M. A. Stott, H. Schmidt, and A. R. Hawkins. "Single Particle Detection Enhancement with Wavelet-based Signal Processing Technique." In CLEO: Science and Innovations, pp. STu3H-4. Optical Society of America, 2019.](https://doi.org/10.1364/CLEO_SI.2019.STu3H.4)
