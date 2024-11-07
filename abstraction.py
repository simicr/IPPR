import numpy as np
import numpy.lib.stride_tricks as ns
import imageio
import skimage.color as skc
import skimage.filters as skf


def edge_detection(im):
    '''Implement DoG smooth edge detection (Eq. 6)'''

    se = skf.gaussian(im, sigma=sigma_e) 
    sf = skf.gaussian(im, sigma=np.sqrt(1.6) * sigma_e) # Po formuli dao.

    diff = se - tau * sf  # Oduzmmeo
    mask = diff < 0 # Nadjemo sve manje od 0
    
    dog = np.ones_like(im) 
    dog[mask] = 1 + np.tanh(phi_e * diff[mask]) # Sve manje od 0 provucemo kroz formulu, ostalo ostaje 1.

    return dog
  

def luminance_quantization(im):
    '''Implement luminance quantization (Eq. 8)'''
    
    lmax = 100 # Zasto je napisao u opisu zadatka da je 100?
    deltaq = lmax / n_bins # Izracunam razmak koji ce biti
    q = np.linspace(0, lmax, n_bins) # Dobijem niz svih razmaka

    diff = np.abs((im[..., None] - q.reshape(1, -1))) # Od svakog piksela izrazunamo razmak od dva granicna broja (i, i+deltaq) 
    argmin = np.argmin(diff, axis=-1) # Nadjemo pozicije najblizih
    closest_qs = q[argmin] # Nadjemo najblize

    ss_phiq = closest_qs + (deltaq/2) * np.tanh(phi_q*(im - closest_qs)) # Data formula
    return ss_phiq


def bilateral_gaussian(im):
    # Radius of the Gaussian filter
    r = int(2 * sigma_s) + 1
    padded = np.pad(im, ((r, r), (r, r), (0, 0)), 'edge')
    '''
    Implement the bilateral Gaussian filter (Eq. 3).
    Apply it to the padded image.
    '''


    windows = ns.sliding_window_view(padded , (2*r+1, 2*r+1, 3)) # Napravice array (400, 254, 1, 17,17,3), u sustini prva dva broja su mesto piksela, ostatak je sam prozor. 
    chnanel_diff = windows - im[:, :, None, None, None, :] # Oduzmem od svakog prozora centralni piksel, ovo bi trebalo da radi. 
    luminance_magnitude = np.sqrt(np.sum((chnanel_diff ** 2), axis=-1)) # Izracunam L2 normu te razlike. 
    luminance_magnitude = np.exp(-luminance_magnitude**2 / (2.0*sigma_s**2)) # Ukoliko ovo prethodno radi imacemo (400, 254, 1, 17, 17). tj onaj deo |F(p) - F(q)|_f u zadatku. 

    indices = np.indices((2*r+1, 2*r+1)) + 1  
    pixels = np.stack(indices, axis=-1) 
    pixels = pixels - pixels[r,r,:] # Meni nesto deluje da svaki prozor kada je razlika piksela u pitanju je uvek isti prozor. I onda mi je ideja da ovo izracunamo samo jednom 
                                    # i posle da ne moramo uopste. 

    pixels_magnitude = np.linalg.norm(pixels, axis=-1, ord=2, keepdims=False) # Izracunam L2 normu, nisam siguran za ovaj keep dims, ali dobije se taman (17,17) sto mozemo samo 
                                                                                   # da pomnozimo sa luminance magnitude.
    pixels_magnitude = np.exp(-pixels_magnitude**2/ (2.0*sigma_r**2)) # Po formuli opet fali /2

    weights =  pixels_magnitude * luminance_magnitude # Pomnozimo ih ovo bi trebalo da bude b(|p-q|_inf)b(|F(p) - F(q)|_f) za svaki prozor. 
    sum_weights = np.sum(weights, axis=(-1,-2, -3)) # Sumiramo, to je delioc u onoj formuli. 
    
    filtered = weights[..., None] * windows # b(|p-q|_inf)b(|F(p) - F(q)|_f) * F(q)
    filtered = np.sum(filtered, axis=(-2, -3, -4)) / sum_weights[..., None] # Sumiramo sve prozore, i izbacimo onu jedinicu u (400, 254, 1 ...)

    return filtered 


def abstraction(im):
    filtered = skc.rgb2lab(im)
    for _ in range(n_e):
        filtered = bilateral_gaussian(filtered)
    edges = edge_detection(filtered[:, :, 0])
 
    for _ in range(n_b - n_e):
        filtered = bilateral_gaussian(filtered)
    luminance_quantized = luminance_quantization(filtered[:, :, 0])

    filtered[:, :, 0] = luminance_quantized * edges # Valjda je ovo dobro
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

    abstracted = abstraction(im)
    
    imageio.imsave('abstracted.png', np.clip(abstracted, 0, 1))
