# -*- coding: utf-8 -*-\
import numpy as np
from skimage import data
from skimage import color
from skimage import io
from matplotlib import pyplot as plt
import get_glcm
import time
from PIL import Image
import cv2
import glob
import os
import Adaptive

def main():
    pass

if __name__ == '__main__':
    
    main()
     
    start = time.time()

    print('---------------0. Parameter Setting-----------------')
    nbit = 64 # gray levels
    mi, ma = 0, 255 # max gray and min gray
    slide_window = 7 # sliding window
    # step = [2, 4, 8, 16] # step
    # angle = [0, np.pi/4, np.pi/2, np.pi*3/4] # angle or direction
    step = [2]
    angle = [0]
    print('-------------------1. Load Data---------------------')
    dir=os.listdir('../picture/')
    for location in dir:
        picture_dir=os.listdir('../picture/'+location+'/')
        for picture in picture_dir:
            aa=picture.split('.')
            print(aa[0])
            image=cv2.imread('../picture/'+location+'/'+str(picture),0)
            image=Adaptive.auto_median_filter(image, 7)
            img_array = np.array(image) # If the image has multi-bands, it needs to be converted to grayscale image
            img_array = np.uint8(255.0 * (img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))) # normalization
            h, w = img_array.shape
            print('------------------2. Calcu GLCM---------------------')
            glcm = get_glcm.calcu_glcm(img_array, mi, ma, nbit, slide_window, step, angle)
            print('-----------------3. Calcu Feature-------------------')
            # 
            for i in range(glcm.shape[2]):
                for j in range(glcm.shape[3]):
                    glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
                    glcm_cut = glcm[:, :, i, j, :, :]
                    
                    mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)
                    variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
                    homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
                    contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
                    dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
                    entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)
                    energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
                    correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
                    Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
                    
            
            print('---------------4. Display and Result----------------')
           
            
            plt.imsave('../GLCM/Mean/'+location+'/'+aa[0]+'.png',arr=mean,dpi=300,cmap ='gray')
            """
            plt.imsave('../GLCM/Variance/'+location+'/'+aa[0]+'.png',arr=variance,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Entropy/'+location+'/'+aa[0]+'.png',arr=entropy,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Energy/'+location+'/'+aa[0]+'.png',arr=energy,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Homogeneity/'+location+'/'+aa[0]+'.png',arr=homogeneity,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Contrast/'+location+'/'+aa[0]+'.png',arr=contrast,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Dissimilarity/'+location+'/'+aa[0]+'.png',arr=dissimilarity,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Correlation/'+location+'/'+aa[0]+'.png',arr=correlation,dpi=300,cmap ='gray')
            plt.imsave('../GLCM/Auto_correlation/'+location+'/'+aa[0]+'.png',arr=Auto_correlation,dpi=300,cmap ='gray')
            """
            
            
            
            """
            plt.figure(figsize=(10, 4.5))
            font = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 12,
            }
            
            plt.subplot(2,5,1)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(img_array, cmap ='gray')
            plt.title('Original', font)
        
            plt.subplot(2,5,2)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(mean, cmap ='gray')
            plt.title('Mean', font)
        
            plt.subplot(2,5,3)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(variance, cmap ='gray')
            plt.title('Variance', font)
        
            plt.subplot(2,5,4)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(homogeneity, cmap ='gray')
            plt.title('Homogeneity', font)
        
            plt.subplot(2,5,5)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(contrast, cmap ='gray')
            plt.title('Contrast', font)
        
            plt.subplot(2,5,6)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(dissimilarity, cmap ='gray')
            plt.title('Dissimilarity', font)
        
            plt.subplot(2,5,7)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(entropy, cmap ='gray')
            plt.title('Entropy', font)
        
            plt.subplot(2,5,8)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(energy, cmap ='gray')
            plt.title('Energy', font)
        
            plt.subplot(2,5,9)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(correlation, cmap ='gray')
            plt.title('Correlation', font)
        
            plt.subplot(2,5,10)
            plt.tick_params(labelbottom=False, labelleft=False)
            plt.axis('off')
            plt.imshow(Auto_correlation, cmap ='gray')
            plt.title('Auto Correlation', font)
        
            plt.tight_layout(pad=0.5)
            plt.savefig('./results/GLCM_results.jpg'
                        , format='png'
                        , bbox_inches = 'tight'
                        , pad_inches = 0
                        , dpi=300)
            plt.show()
            """
            
            end = time.time()
            print('Code run time:', end - start)
        
    