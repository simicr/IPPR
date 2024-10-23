import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''

    se = skf.gaussian(im, sigma=sigma_e)
    sf = skf.gaussian(im, sigma=np.sqrt(1.6) * sigma_e)

    diff = se - tau * sf    
    mask = diff < 0
    
    dog = np.ones_like(im)
    dog[mask] = 1 + np.tanh(phi_e * diff[mask]) 

    return dog
  

def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''
    
    lmax = 1 # Zasto je napisao u opisu zadatka da je 100?
    deltaq = lmax / n_bins 
    q = np.linspace(0, lmax, n_bins)

    diff = np.abs((im[..., np.newaxis] - q.reshape(1, -1)))
    argmin = np.argmin(diff, axis=-1)
    closest_qs = q[argmin]

    ss_phiq = closest_qs + (deltaq/2) * np.tanh(phi_q*(im - closest_qs))
    return ss_phiq


def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''


    return im


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    edges = edge_detection(filtered[:, :, 0])

    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    '''Get the final image by merging the channels properly'''
    combined = filtered  # Todo
    return skc.lab2rgb(combined)


if __name__ == '__main__':
    # Algorithm
    n_e = 2
    n_b = 4
    # Bilateral Filter
    sigma_r = 4.25  # "Range" sigma
    sigma_s = 3.5  # "Spatial" sigma
    # Edge Detection
    sigma_e = 1
    tau = 0.98
    phi_e = 5
    # Luminance Quantization
    n_bins = 10
    phi_q = 0.7

    im = imageio.imread('./girl.png') / 255.
    
    abstracted = bilateral_gaussian(im)
    
    imageio.imsave('abstracted.png', np.clip(abstracted, 0, 1))
