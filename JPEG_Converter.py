import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#Convert BGR image to YCbCr image
def cvtBGR2YCbCr(img):
    B, G, R = cv.split(img)
    Y = (0.299*R + 0.587*G + 0.114*B).astype(int)
    Cb = (-0.1687*R - 0.3313*G + 0.5*B + 128*np.ones_like(B)).astype(int)
    Cr = (0.5*R - 0.4187*G - 0.0813*B + 128*np.ones_like(B)).astype(int)
    return cv.merge([Y, Cb, Cr])

#Utility functions to visualize YCbCr color channels
def cvtY2RGB(Y):
    viewY = cvtYCbCr2RGB(cv.merge([Y, 128*np.ones_like(Y), 128*np.ones_like(Y)]))
    return viewY
def cvtCb2RGB(Cb):
    viewCb = cvtYCbCr2RGB(cv.merge([128*np.ones_like(Cb), Cb, 128*np.ones_like(Cb)]))
    return viewCb
def cvtCr2RGB(Cr):
    viewCr = cvtYCbCr2RGB(cv.merge([128*np.ones_like(Cr), 128*np.ones_like(Cr), Cr]))
    return viewCr

#Convert YCbCr image to RGB image
def cvtYCbCr2RGB(img):
    Y, Cb, Cr = cv.split(img)
    R = (Y + 1.402*(Cr - 128*np.ones_like(Y))).astype(int)
    G = (Y - 0.34414*(Cb - 128*np.ones_like(Y)) - 0.71414*(Cr - 128*np.ones_like(Y))).astype(int)
    B = (Y + 1.772*(Cb - 128*np.ones_like(Y))).astype(int)
    return cv.merge([R, G, B])

#Remove one row or column to make sure the channel has even dimensions (or mod-2^depth dim)
def evenDim(channel, depth=1):
    h, w = channel.shape
    newh = h - h%(2**depth)
    neww = w - w%(2**depth)
    newchannel = channel[:newh, :neww]
    return newchannel

#Subsample the channel (1/4x resolution)
def subsample2(channel):
    channel = evenDim(channel)
    h, w = channel.shape
    subchannel = np.zeros((h//2, w//2), dtype=int)
    for i in range(h//2):
        for j in range(w//2):
            subchannel[i, j] = channel[2*i, 2*j]
    return subchannel

#4:2:0 Sub-sample YCbCr image (full resolution for Y, 1/4 resolution for Cb and Cr)
def YCbCrSubsample(img, depth=1):
    Y, Cb, Cr = cv.split(img)
    Y, Cb, Cr = evenDim(Y, depth), evenDim(Cb, depth), evenDim(Cr, depth)
    for i in range(depth):
        Cb, Cr = subsample2(Cb), subsample2(Cr)
    return Y, Cb, Cr

#Oversample the channel (4x resolution)
def oversample2(channel):
    h, w = channel.shape
    overchannel = np.zeros((h*2, w*2), dtype=int)
    for i in range(h):
        for j in range(w):
            value = channel[i, j]
            overchannel[2*i, 2*j] = value
            overchannel[2*i, 2*j+1] = value
            overchannel[2*i+1, 2*j] = value
            overchannel[2*i+1, 2*j+1] = value
    return overchannel

#Reconstruct the image from the 4:2:0 subsampled YCbCr image
def YCbCrOversample(Y, Cb, Cr, depth=1):
    for i in range(depth):
        Cb, Cr = oversample2(Cb), oversample2(Cr)
    return cv.merge([Y, Cb, Cr])

#Shift color from 0:255 to -128:127
def shift(channel, k):
    return channel + k*np.ones_like(channel)

#Calculates the DCT (Discrete Cosine Transform) matrix of order N
def DCTMatrix(N):
    return np.array([ [ np.cos( np.pi * (2*i+1) * j / (2*N)) for i in range(N)] for j in range(N)])

#Calculates the inverse DCT matrix of order N
def invDCTMatrix(N):
    return np.array([ [1/2] + [ np.cos( np.pi * (2*j+1) * i / (2*N)) for i in range(1, N)] for j in range(N)]) * 2 / N

#Calculates the DCT of a channel
def DCT8(channel):
    h, w = channel.shape
    DCTMat = DCTMatrix(8)
    rowDCTchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            for k in range(8):
                rowDCTchannel[8*i+k, 8*j:8*j+8] = DCTMat.dot(channel[8*i+k, 8*j:8*j+8].T)
    DCTchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            for k in range(8):
                DCTchannel[8*i:8*i+8, 8*j+k] = DCTMat.dot(rowDCTchannel[8*i:8*i+8, 8*j+k])
    return DCTchannel

#Calculates the DCT of the YCbCr image
def YCbCrDCT(Y, Cb, Cr):
    Y = evenDim(Y, 4)
    Cb, Cr = evenDim(Cb, 3), evenDim(Cr, 3)
    DCTY = DCT8(shift(Y, -128))
    DCTCb = DCT8(shift(Cb, -128))
    DCTCr = DCT8(shift(Cr, -128))
    return DCTY, DCTCb, DCTCr

#Calculates the inverse DCT of a channel
def invDCT8(channel):
    h, w = channel.shape
    invDCTMat = invDCTMatrix(8)
    colinvDCTchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            for k in range(8):
                colinvDCTchannel[8*i:8*i+8, 8*j+k] = invDCTMat.dot(channel[8*i:8*i+8, 8*j+k])
    invDCTchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            for k in range(8):
                invDCTchannel[8*i+k, 8*j:8*j+8] = invDCTMat.dot(colinvDCTchannel[8*i+k, 8*j:8*j+8].T)
    return invDCTchannel

#Utility function to see the spatial frequencies
def baseSpatial():
    h, w = 8*8 + 7*8, 8*8 + 7*8
    base = np.zeros((h, w), dtype=int)
    for i in range(8):
        for j in range(8):
            base[16*i+i, 16*j+j] = 256
    return base

#Calculates the inverse DCT of the YCbCr image
def YCbCrInvDCT(DCTY, DCTCb, DCTCr):
    Y = shift(invDCT8(DCTY), 128)
    Cb = shift(invDCT8(DCTCb), 128)
    Cr = shift(invDCT8(DCTCr), 128)
    return Y, Cb, Cr

#Quantization table of Y channel
def YquantizationTable(quality): #Quality in ]0, 1]
    QTable = np.array([ [2 , 1 , 1 , 1 , 1 , 1 , 2 , 1 ],
                        [1 , 1 , 2 , 2 , 2 , 2 , 2 , 4 ],
                        [3 , 2 , 2 , 2 , 2 , 5 , 4 , 4 ],
                        [3 , 4 , 6 , 5 , 6 , 6 , 6 , 5 ],
                        [6 , 6 , 6 , 7 , 9 , 8 , 6 , 7 ],
                        [9 , 7 , 6 , 6 , 8 , 11, 8 , 9 ],
                        [10, 10, 10, 10, 10, 6 , 8 , 11],
                        [12, 11, 10, 12, 9 , 10, 10, 10] ])
    QTable = QTable / quality
    return QTable.astype(int)

#Quantization table of Cb and Cr channel
def CquantizationTable(quality): #Quality in ]0, 1]
    QTable = np.array([ [2 , 2 , 2 , 2 , 2 , 2 , 5 , 3 ],
                        [3 , 5 , 10, 7 , 6 , 7 , 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10],
                        [10, 10, 10, 10, 10, 10, 10, 10] ])
    QTable = QTable / quality
    return QTable.astype(int)

#Quantization using a table
def quantize(channel, Qtable):
    h, w = channel.shape
    Qchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            Qchannel[8*i:8*i+8, 8*j:8*j+8] = channel[8*i:8*i+8, 8*j:8*j+8] / Qtable
    return Qchannel.astype(int)

#Quantization of YCbCr DCT image
def YCbCrQuantization(Y, Cb, Cr, quality):
    YTable = YquantizationTable(quality)
    CTable = CquantizationTable(quality)
    QY = quantize(Y, YTable)
    QCb = quantize(Cb, CTable)
    QCr = quantize(Cr, CTable)
    return QY, QCb, QCr

#Inverse quantization using a table
def invQuantize(channel, Qtable):
    h, w = channel.shape
    invQchannel = np.zeros_like(channel)
    for i in range(h//8):
        for j in range(w//8):
            invQchannel[8*i:8*i+8, 8*j:8*j+8] = channel[8*i:8*i+8, 8*j:8*j+8] * Qtable
    return invQchannel.astype(int)

#Inverse quantization of YCbCr DCT image
def YCbCrInvQuantization(Y, Cb, Cr, quality):
    YTable = YquantizationTable(quality)
    CTable = CquantizationTable(quality)
    invQY = invQuantize(Y, YTable)
    invQCb = invQuantize(Cb, CTable)
    invQCr = invQuantize(Cr, CTable)
    return invQY, invQCb, invQCr
"""
plt.imshow(invDCT8(baseSpatial()), cmap='gray')
plt.show()
"""

img = cv.imread("IMAGE FILE NAME AND PATH")
#Original image
RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(3, 2, 3), plt.title("Original image"), plt.imshow(RGB)
#YCbCr split
YCbCr = cvtBGR2YCbCr(img)
Y, Cb,Cr = cv.split(YCbCr)
plt.subplot(3, 2, 2), plt.title("Original Y channel"), plt.imshow(cvtY2RGB(Y))
plt.subplot(3, 2, 4), plt.title("Original Cb channel"), plt.imshow(cvtCb2RGB(Cb))
plt.subplot(3, 2, 6), plt.title("Original Cr channel"), plt.imshow(cvtCr2RGB(Cr))
plt.show()

plt.subplot(3, 2, 1), plt.title("Original Y channel"), plt.imshow(cvtY2RGB(Y))
plt.subplot(3, 2, 3), plt.title("Original Cb channel"), plt.imshow(cvtCb2RGB(Cb))
plt.subplot(3, 2, 5), plt.title("Original Cr channel"), plt.imshow(cvtCr2RGB(Cr))
#4:2:0 Subsample
subY, subCb, subCr = YCbCrSubsample(YCbCr, 1)
plt.subplot(3, 2, 2), plt.title("Original Y channel"), plt.imshow(cvtY2RGB(Y))
plt.subplot(3, 2, 4), plt.title("Subspampled Cb channel"), plt.imshow(cvtCb2RGB(subCb))
plt.subplot(3, 2, 6), plt.title("Subsampled Cr channel"), plt.imshow(cvtCr2RGB(subCr))
plt.show()

plt.subplot(3, 2, 1), plt.title("Original Y channel"), plt.imshow(cvtY2RGB(Y))
plt.subplot(3, 2, 3), plt.title("Subspampled Cb channel"), plt.imshow(cvtCb2RGB(subCb))
plt.subplot(3, 2, 5), plt.title("Subsampled Cr channel"), plt.imshow(cvtCr2RGB(subCr))
#DCT
DCTY, DCTCb, DCTCr = YCbCrDCT(subY, subCb, subCr)
plt.subplot(3, 2, 2), plt.title("DCT of Y channel"), plt.imshow(DCTY)
plt.subplot(3, 2, 4), plt.title("DCT of Cb channel"), plt.imshow(DCTCb)
plt.subplot(3, 2, 6), plt.title("DCT of Cr channel"), plt.imshow(DCTCr)
plt.show()

plt.subplot(3, 2, 1), plt.title("DCT of Y channel"), plt.imshow(DCTY)
plt.subplot(3, 2, 3), plt.title("DCT of Cb channel"), plt.imshow(DCTCb)
plt.subplot(3, 2, 5), plt.title("DCT of Cr channel"), plt.imshow(DCTCr)
#Quantization
quality = 0.5
QY, QCb, QCr = YCbCrQuantization(DCTY, DCTCb, DCTCr, quality)
plt.subplot(3, 2, 2), plt.title("Quantized Y channel"), plt.imshow(QY)
plt.subplot(3, 2, 4), plt.title("Quantized Cb channel"), plt.imshow(QCb)
plt.subplot(3, 2, 6), plt.title("Quantized Cr channel"), plt.imshow(QCr)
plt.show()

plt.subplot(3, 2, 1), plt.title("Quantized Y channel"), plt.imshow(QY)
plt.subplot(3, 2, 3), plt.title("Quantized Cb channel"), plt.imshow(QCb)
plt.subplot(3, 2, 5), plt.title("Quantized Cr channel"), plt.imshow(QCr)
#Inverse quantization
invQY, invQCb, invQCr = YCbCrInvQuantization(QY, QCb,QCr, quality)
plt.subplot(3, 2, 2), plt.title("Reconstrcucted DCT of Y channel"), plt.imshow(invQY)
plt.subplot(3, 2, 4), plt.title("Reconstrcucted DCT of Cb channel"), plt.imshow(invQCb)
plt.subplot(3, 2, 6), plt.title("Reconstrcucted DCT of Cr channel"), plt.imshow(invQCr)
plt.show()

plt.subplot(3, 2, 1), plt.title("DCT of Y channel"), plt.imshow(DCTY)
plt.subplot(3, 2, 3), plt.title("DCT of Cb channel"), plt.imshow(DCTCb)
plt.subplot(3, 2, 5), plt.title("DCT of Cr channel"), plt.imshow(DCTCr)
plt.subplot(3, 2, 2), plt.title("Reconstrcucted DCT of Y channel"), plt.imshow(invQY)
plt.subplot(3, 2, 4), plt.title("Reconstrcucted DCT of Cb channel"), plt.imshow(invQCb)
plt.subplot(3, 2, 6), plt.title("Reconstrcucted DCT of Cr channel"), plt.imshow(invQCr)
plt.show()

print(DCTCb[0:8,0:8])
print("--------------------------------------------------")
print(invQCb[0:8,0:8])

plt.subplot(3, 2, 1), plt.title("Reconstrcucted DCT of Y channel"), plt.imshow(invQY)
plt.subplot(3, 2, 3), plt.title("Reconstrcucted DCT of Cb channel"), plt.imshow(invQCb)
plt.subplot(3, 2, 5), plt.title("Reconstrcucted DCT of Cr channel"), plt.imshow(invQCr)
#Inverse DCT
invDCTY, invDCTCb, invDCTCr = YCbCrInvDCT(invQY, invQCb, invQCr)
plt.subplot(3, 2, 2), plt.title("Reconstructed Y channel"), plt.imshow(cvtY2RGB(invDCTY))
plt.subplot(3, 2, 4), plt.title("Reconstructed Cb channel"), plt.imshow(cvtCb2RGB(invDCTCb))
plt.subplot(3, 2, 6), plt.title("Reconstructed Cr channel"), plt.imshow(cvtCr2RGB(invDCTCr))
plt.show()

plt.subplot(3, 2, 1), plt.title("Reconstructed Y channel"), plt.imshow(cvtY2RGB(invDCTY))
plt.subplot(3, 2, 3), plt.title("Reconstructed Cb channel"), plt.imshow(cvtCb2RGB(invDCTCb))
plt.subplot(3, 2, 5), plt.title("Reconstructed Cr channel"), plt.imshow(cvtCr2RGB(invDCTCr))
#Reconstruction
reconstructedYCbCr = YCbCrOversample(invDCTY, invDCTCb, invDCTCr, 1)
reconstructedRGB = cvtYCbCr2RGB(reconstructedYCbCr)
plt.subplot(3, 2, 4), plt.title("Reconstructed image"), plt.imshow(reconstructedRGB)
plt.show()

plt.subplot(2, 1, 1), plt.title("Original image"), plt.imshow(RGB)
plt.subplot(2, 1, 2), plt.title("Reconstructed image"), plt.imshow(reconstructedRGB)
plt.show()