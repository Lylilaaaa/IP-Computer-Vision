import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import median_filter
import notch_filter

img_path_1 = './pro1_radiograph_1.jpg'
img_path_2 = './pro1_radiograph_2.jpg'
img1 = cv.imread(img_path_1,0)
img2 = cv.imread(img_path_2,0)

#(a)
outputs = []

kernelInput = [3,7,11,15]
for k in kernelInput:
    temp = median_filter.median_filter(img1, k)
    outputs.append(temp)

plt.figure()
for i in range(len(outputs)):
    plt.subplot(2,3,i+1)
    plt.imshow(outputs[i],'gray')
    plt.title('N = '+str(kernelInput[i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('./img1_a.jpg')
plt.show()

#处理第二张图
outputs = []

kernelInput = [5,9,13,17]
for k in kernelInput:
    temp = median_filter.median_filter(img2, k)
    outputs.append(temp)

plt.figure()
for i in range(len(outputs)):
    plt.subplot(2,3,i+1)
    plt.imshow(outputs[i],'gray')
    plt.title('N = '+str(kernelInput[i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('./img2_a.jpg')
plt.show()

#(b)
dft = np.fft.fft2(img1)
dftshift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(np.abs(dftshift)) #更加清楚展示
plt.subplot(121)
plt.imshow(img1, cmap = 'gray')
plt.title('original')
plt.axis('off')
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('frequency')
plt.axis('off')
plt.savefig('./img1_b_DFT.jpg')
plt.show()

plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.savefig('./img1_b_DFT_pure.jpg')
plt.show()

dft = np.fft.fft2(img2)
dftshift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(np.abs(dftshift))
plt.subplot(121)
plt.imshow(img2, cmap = 'gray')
plt.title('original')
plt.axis('off')
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('frequency')
plt.axis('off')
plt.savefig('./img2_b_DFT.jpg')
plt.show()

plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.savefig('./img2_b_DFT_pure.jpg')
plt.show()

#(c)
f = np.fft.fft2(img1)
_dftinput = np.fft.fftshift(f)

f2 = np.fft.fft2(img2)
_dftinput2 = np.fft.fftshift(f2)

img_output1 = notch_filter.IdealNotchFilter(_dftinput, [
    [234,259], 
    [245,213], 
    [261,123],[269,80],
], 70)

f_output = np.fft.fft2(img_output1)
f_output_shift = np.fft.fftshift(f_output)
f_output_magnitude_spectrum = 20*np.log(np.abs(f_output_shift))
plt.imshow(f_output_magnitude_spectrum)
plt.savefig('./img1_c_INF_fre.jpg')
plt.show()

plt.imshow(img_output1, cmap = "gray")
plt.savefig('./img1_c_INF.jpg')
plt.show()

img_output2 = notch_filter.IdealNotchFilter(_dftinput2, [
    [274,440], 
    [208,451], 
], 100)

f_output2 = np.fft.fft2(img_output2)
f_output_shift2 = np.fft.fftshift(f_output2)
f_output_magnitude_spectrum2 = 20*np.log(np.abs(f_output_shift2))
plt.imshow(f_output_magnitude_spectrum2)
plt.savefig('./img2_c_INF_fre.jpg')
plt.show()

plt.imshow(img_output2, cmap = "gray")
plt.savefig('./img2_c_INF.jpg')
plt.show()