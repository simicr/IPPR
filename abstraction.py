import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''

    se = skf.gaussian(im, sigma=sigma_e) 
    sf = skf.gaussian(im, sigma=np.sqrt(1.6) * sigma_e) # given formula

    diff = se - tau * sf  # substract
    mask = diff < 0 # find all that are smaller then 0
    
    dog = np.ones_like(im) 
    dog[mask] = 1 + np.tanh(phi_e * diff[mask]) # apply formula to all that are smaller then 0, rest stays 1

    return dog
  

def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''
    
    lmax = 100.0 
    deltaq = lmax / n_bins # calculate step
    q = np.linspace(0, lmax, n_bins) # get an array of all steps

    diff = np.abs((im[..., None] - q.reshape(1, -1))) # difference between (i, i+deltaq) for every step
    argmin = np.argmin(diff, axis=-1) # find positions of the closest ones
    closest_qs = q[argmin] # find the closest ones

    ss_phiq = closest_qs + (deltaq/2) * np.tanh(phi_q*(im - closest_qs)) # given formula
    return ss_phiq


def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''

    windows = ns.sliding_window_view(padded , (2*r+1, 2*r+1, 3)) # makes an array (400, 254, 1, 17,17,3), first two numbers are for pixel position, the rest is the window itself. 
    chnanel_diff = windows - im[:, :, None, None, None, :] # substract central pixel intensity from every window 
    intensity_magnitude = np.sqrt(np.sum((chnanel_diff ** 2), axis=-1)) # calculate L2 norm of difference 
    intensity_magnitude = np.exp(-intensity_magnitude**2 / (2.0*sigma_r**2)) # results in (400, 254, 1, 17, 17); |F(p) - F(q)|_f. 

    indices = np.indices((2*r+1, 2*r+1)) + 1  
    pixels = np.stack(indices, axis=-1) 
    pixels = pixels - pixels[r,r,:] # for pixel difference every window is the same, therefore we calculate it only once.

    pixels_magnitude = np.linalg.norm(pixels, axis=-1, ord=2, keepdims=False) # calculate L2 norm
    pixels_magnitude = np.exp(-pixels_magnitude**2/ (2.0*sigma_s**2)) 

    weights =  pixels_magnitude * intensity_magnitude # b(|p-q|_inf)b(|F(p) - F(q)|_f) for every window. 
    sum_weights = np.sum(weights, axis=(-1,-2, -3)) # detominator. 
    
    filtered = weights[..., None] * windows # b(|p-q|_inf)b(|F(p) - F(q)|_f) * F(q)
    filtered = np.sum(filtered, axis=(-2, -3, -4)) / sum_weights[..., None] # summ all windows

    return filtered 


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for i in range(n_e):
        filtered = bilateral_gaussian(filtered)

    edges = edge_detection(filtered[:, :, 0])
 
    for i in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)

    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    '''Get the final image by merging the channels properly'''
    filtered[:, :, 0] = luminance_quantized * edges 
    combined = np.clip(filtered, [0., -128., -128.], [100., 127., 127.]) 
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

    abstracted = abstraction(im)
    
    imageio.imsave('abstracted.png', (np.clip(abstracted, 0, 1) * 255.).astype(np.uint8))
