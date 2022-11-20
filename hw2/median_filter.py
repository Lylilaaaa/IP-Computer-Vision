import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def median_filter(imgInput, kernelSize):
    temp = []
    bounder = kernelSize // 2 #取整，获得要从中间向两边拓展要被考虑比大小的pix，单边的长度
    imgHeight = len(imgInput)
    imgWidth = len(imgInput[0])
    output = np.zeros((imgHeight,imgWidth))
    for row in range(imgHeight):  #一行一行遍历，到一列一列遍历
        for col in range(imgWidth):

            for ker in range(bounder):  #比较自己和周围pix的大小
                if row + ker - bounder < 0 or row + ker - bounder > imgHeight - 1:  #height超出边界
                    for c in range(kernelSize):
                        temp.append(0) #padding with 0 用0值替代超出来的值
                else:
                    if col + ker - bounder < 0 or col + bounder > imgWidth - 1:
                        temp.append(0)
                    else:
                        for k in range(kernelSize):
                            temp.append(imgInput[row + ker - bounder][col + k - bounder])

            temp.sort()
            output[row][col] = temp[len(temp) // 2]
            temp = []
    return output


#outputs and shown
