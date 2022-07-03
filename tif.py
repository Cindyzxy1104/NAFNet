from libtiff import TIFF
#from scipy import misc
import matplotlib.pyplot as plt
import numpy as np


##TIFF files are parsed into image sequences
##tiff_ image_ Name: TIFF file name;
##out_ Folder: the folder where the image sequence is saved
##out_ Type: the type of saved image, such as. JPG,. PNG,. BMP, etc
def tiff_to_image_array(tiff_image_name, out_folder, out_type):
    tif = TIFF.open(tiff_image_name, mode = "r")
    idx = 0
    im_folder=[]
    for im in list(tif.iter_images())[:1]:
        im_name = out_folder + str(idx) + out_type
        # add one dimension for rgb
        im = np.dstack([im,im,im]).astype(np.float32)

        #misc.imsave(im_name, im)
        #print(im_name, 'successfully saved!!!')
        idx = idx +1

        #crop the image from 2048*2048*3 to 256*256*3
        #tiles = [im[x:x+256,y:y+256,:] for x in range(0,im.shape[0],256) for y in range(0,im.shape[1],256)]
        tiles=[]
        for x in range(0,im.shape[0],256):
            for y in range(0,im.shape[1],256):
                tiles.append(im[x:x+256,y:y+256,:])
        im_folder.append(tiles)
    return im_folder




