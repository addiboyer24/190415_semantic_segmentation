import numpy as np
import scipy.ndimage as snd
def class_activation_mapping(weights, feature_index, final_conv, train_img):
       '''Generalized function to perform class activation mapping'''
       
    num_iters = final_conv.shape[3]
    w_index = 0
    fm_0 = np.zeros((final_conv.shape[1], final_conv.shape[2]))

    # Sum the weighted feature maps into fm_0
    for x in range(num_iters):
        final_conv[0,:,:,x] *= np.absolute(weights[w_index][feature_index])
        fm_0 += final_conv[0,:,:,x]
        w_index+=1

    # get the integer upscale factor and perform upscaling
    upscale_factor = train_img.shape[0] // final_conv.shape[1]
    fm_0_upscaled = snd.zoom(fm_0,upscale_factor)
    
    return fm_0_upscaled
