# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

import matplotlib.pyplot as plt
import numpy as np
from tif import tiff_to_image_array
from PIL import Image
# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    #img_path = opt['img_path'].get('input_img')
    #output_path = opt['img_path'].get('output_img')


    ## 1. read image

    '''
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))
    breakpoint()
    img = img2tensor(img, bgr2rgb=True, float32=True)
    '''
    #Get images from TIF file using function from tif.py
    im_folder = tiff_to_image_array("sample1.tif", "NAFNet", ".PNG")
    #img_original = im_folder[0][0]
    imgfolder = im_folder.copy()
    #print(len(imgfolder))
    #breakpoint()
    for pic in range(len(imgfolder)):
        idx = 0
        to1pic=[]
        for img in range(len(imgfolder[pic])):
            imgfolder[pic][img] = imgfolder[pic][img]/255

        # list of numpy array to list of tensor
        imgfolder[pic] = img2tensor(imgfolder[pic], bgr2rgb=True, float32=True)

            ## 2. run inference
        opt['dist'] = False
        model = create_model(opt)
        for tile in imgfolder[pic]:
            model.feed_data(data={'lq': tile.unsqueeze(dim=0)})

            if model.opt['val'].get('grids', False):
                model.grids()

            model.test()

            if model.opt['val'].get('grids', False):
                model.grids_inverse()

            visuals = model.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            to1pic.append(sr_img)
                # put 64 tiles back to 1 pic
        #for image in im_folder:
        line = []
        for j in range(0,len(to1pic),8):
            line.append(np.hstack(to1pic[j:j+8]))
        picture = np.vstack(line)
        imwrite(sr_img, "./result/denoise_img_%d.png"%(idx))
        idx += 1


    #print(f'inference {img_path} .. finished. saved to {output_path}')


if __name__ == '__main__':
    main()


