#!/usr/bin/env python
# coding: utf-8

# # PDI - Projeto Individual 1
# 
# - Data: 11/2020
# - Nome: Bruna Medeiros da Silva
# - Matrícula: 16/0048711
# 
# - Professor: Renan Utida
# - Matéria: Processamento Digital de Imagens

# ## Imports - Importando Bibliotecas

# In[1]:


from scipy import signal
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import statistics 
import numpy as np
import cv2


# ## 1. Ajuste de intensidade
# 
# **Método:** Equalização de Histograma
# 
# **Banco de Imagens:** [Home Objects](http://www.vision.caltech.edu/pmoreels/Datasets/Home_Objects_06/#Download)

# In[2]:


def my_histogram(image, plot = True, amax = 256, norm = False):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get number of lines and columns
    qntI, qntJ = image.shape
    
    # Creating Histogram manually
    histogram = np.zeros(amax)
    color = 0
    for i in range(qntI):
        for j in range(qntJ):
            color = image[i][j]
            # print(color)
            histogram[color] += 1
            
    if(norm):
        histogram = (histogram - np.amin(histogram)) /  (np.amax(histogram) - np.amin(histogram))
    
    if(plot):
        plt.figure()
        plt.stem(histogram, use_line_collection = True)
        plt.title('Original Image Histogram $p_r(r)$')
        plt.savefig('hist_original_fig')
        plt.show()
    return histogram

def cdf_pdf(pdf, plot = True):
    cdf = np.zeros(len(pdf))
    for h in range(len(pdf)):
        cdf[h] = np.sum(pdf[0: h + 1])
    if(plot):
        plt.figure()
        plt.stem(pdf, use_line_collection = True)
        plt.title('Probability Distribuition function (PDF) $p_z(z)$')
        plt.show()

        plt.figure()
        plt.stem(cdf, use_line_collection = True)
        plt.plot(cdf, 'k')
        plt.title('Cumulative Distribution Function (CDF) $G(z)$')
        plt.show()
    return cdf

def cdf_2D(img, plot = True, amax = 256, norm = False):
    cdf = np.zeros(amax)
    # sdf = np.zeros(amax)
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get number of lines and columns
    qntI, qntJ = img.shape
    # Number of pixels
    qnt_pixels = qntI * qntJ
   
    histogram = my_histogram(img, amax = amax, norm = norm, plot = False)
    # print(len(histogram))
    for h in range(amax):
        cdf[h] = np.sum(histogram[0: h + 1]) / qnt_pixels 
        # sdf[h] = histogram[h] / qnt_pixels 
        
    
    if(plot):
        plt.figure()
        plt.stem(cdf, use_line_collection = True)
        plt.plot(cdf, 'k')
        plt.title('Cumulative Distribution Function (CDF) $G(s)$')
        plt.show()
    return cdf

def inv_cdf(img, required_pdf = None, amax = 256):
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y = img.shape
    new_img = np.copy(img)

    if(required_pdf is None):
        required_pdf = np.divide(np.ones(amax), amax)
    s = np.round(np.multiply((amax - 1), cdf_2D(img, plot = False)))
    G = np.round(np.multiply((amax - 1), cdf_pdf(required_pdf, plot = False)))

    # Example
    # original_pdf = [0.19, 0.25, 0.21, 0.16, 0.08, 0.06, 0.03, 0.02]
    # required_pdf = [0.0, 0.0, 0.0, 0.15, 0.20, 0.30, 0.20, 0.15]
    # amax = 8
    # s = np.round(np.multiply((amax - 1), cdf_pdf(original_pdf, plot = False)))
    # G = np.round(np.multiply((amax - 1), cdf_pdf(required_pdf, plot = False)))

    s = s.astype(np.uint8)
    new_z = np.zeros(amax)
    G_s = np.zeros(amax)
    diffs = []

    for k in range(amax):
        diffs = np.abs(np.subtract(G, s[k]))
        new_z[k] = np.argmin(diffs)
        G_s[s[k]] = np.argmin(diffs)

    plt.figure()
    plt.stem(G, linefmt = 'k', use_line_collection = True)
    plt.plot(s, '-or')
    plt.legend(['$s_k$', '$G(z_k)$'])
    plt.savefig('cdf_images')
    plt.show()

    plt.figure()
    markerline, stemlines, baseline = plt.stem(s, G_s[s], linefmt = 'k', markerfmt = '-oc', use_line_collection = True)
    plt.setp(baseline, color='k', linewidth=2)
    plt.setp(markerline, linewidth=3)
    plt.title('Mapping s in z')
    plt.xlabel('original values (s)')
    plt.ylabel('new values ($z = G^{-1}(s)$)')
    plt.savefig('transform_graph')
    plt.show()

    for i in range(x):
        for j in range(y):
            new_img[i][j] = new_z[img[i][j]]
    new_img = new_img.astype(np.uint8)
    return new_img


# In[114]:


img = cv2.imread("images_1/int_adj_1.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_img = inv_cdf(img)

fig, ax = plt.subplots(1, 2, figsize = [15, 15])
ax[0].imshow(img, cmap='gray', vmin = 0, vmax = 255)
ax[1].imshow(new_img, cmap='gray', vmin = 0, vmax = 255)
plt.savefig('images_1/result1_1')
plt.show()


# In[115]:


img = cv2.imread("images_1/int_adj_3.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_img = inv_cdf(img)

fig, ax = plt.subplots(1, 2, figsize = [15, 15])
ax[0].imshow(img, cmap='gray', vmin = 0, vmax = 255)
ax[1].imshow(new_img, cmap='gray', vmin = 0, vmax = 255)
plt.savefig('images_1/result1_2')
plt.show()


# ## 2. Realce de imagem de baixa resolução
# 
# 
# **Banco de Imagens:** [Caltech-UCSD Birds 200](http://www.vision.caltech.edu/visipedia/CUB-200.html)

# In[187]:


def img_filtering(image, D_0 = 80, n = 1):
    fft_image = fftshift(fft2(image))

    M, N = fft_image.shape
    P = M
    Q = N
    H = np.zeros(fft_image.shape)
    
    for u in range(M):
        for v in range(N):
            D = ((u - (P/2))**2 + (v - (Q/2))**2)**(1 / 2)
            H[u][v] = np.divide(1, 1 + (D_0/D)**(2 * n))
            
    
    filt_fft_img = fft_image * H
    filtered_img = ifft2(ifftshift(filt_fft_img))
    return filtered_img, H

def butter_filter(image, D_0 = 80, tol = 1e-3, n = 1):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    new_image = np.zeros(image.shape).astype(np.float)
    freq_filter = np.zeros(image.shape)
    
    log_image = image + tol
    log_image = np.log(log_image)    
    
    img_filtered, freq_filter = img_filtering(log_image, D_0 = D_0, n = n)
    img_filtered = np.exp(np.real(img_filtered))
    img_filtered = img_filtered * 255 / img_filtered.max()
    new_image = image + img_filtered
    new_image = np.floor(new_image * 255 / new_image.max())

#new_image.astype(np.uint8)
# img_filtered
    return new_image.astype(np.uint8) , freq_filter, img_filtered.astype(np.uint8)


# In[188]:


def img_laplace(image):
    fft_image = fftshift(fft2(image))

    M, N = fft_image.shape
    P = M
    Q = N
    H = np.zeros(fft_image.shape)
    
    for u in range(M):
        for v in range(N):
            D = ((u - (P/2))**2 + (v - (Q/2))**2)**(1/2)
            H[u][v] = - 4 * (np.pi**2) * D**2
    
    filt_fft_img = fft_image * H
    filtered_img = ifft2(ifftshift(filt_fft_img))

    return filtered_img

def lapl_filter(image):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    new_image = np.zeros(image.shape).astype(np.float)
    freq_filter = np.zeros(image.shape) 
    
    img_filtered = img_laplace(image)
    img_filtered = np.abs(img_filtered)
    img_filtered = (np.subtract(img_filtered, img_filtered.min())) * 255 / img_filtered.max()
    new_image = image + img_filtered
    new_image = np.subtract(new_image, new_image.min()) * 255 / new_image.max()

#new_image.astype(np.uint8)
# img_filtered
    return new_image.astype(np.uint8), img_filtered.astype(np.uint8)


# ### Imagem #1

# In[189]:


img = cv2.imread("images_2/dogs.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap = 'gray')


# In[190]:


filtered, freq_filter, img_filter = butter_filter(img, D_0 = 20, n = 5)
# filtered, freq_filter, img_filter = butter_filter(filtered, D_0 = 20)
plt.imshow(freq_filter, cmap = 'gray')


# In[192]:


fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][0].set_title('Original Image')
axs[1][0].imshow(20*np.log10(np.abs(fftshift(fft2(img)))), cmap = 'gray', vmin = 0, vmax = 255)


# axs[0][1].imshow(img_filter, cmap = 'gray', vmin = 0, vmax = 255)
# axs[0][1].set_title('Filtered Image')
# axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(img_filter)))), cmap = 'gray', vmin = 0, vmax = 255)
# plt.tight_layout()
# plt.show()

axs[0][1].imshow(filtered, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][1].set_title('Filtered Image')
axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(filtered)))), cmap = 'gray')
plt.tight_layout()

plt.savefig('images_2/result2_1_butter')
plt.show()


# In[193]:


filtered, img_filter = lapl_filter(img)

fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][0].set_title('Original Image')
axs[1][0].imshow(20*np.log10(np.abs(fftshift(fft2(img)))), cmap = 'gray', vmin = 0, vmax = 255)


# axs[0][1].imshow(img_filter, cmap = 'gray', vmin = 0, vmax = 255)
# axs[0][1].set_title('Filtered Image')
# axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(img_filter)))), cmap = 'gray', vmin = 0, vmax = 255)
# plt.tight_layout()
# plt.show()

axs[0][1].imshow(filtered, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][1].set_title('Filtered Image')
axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(filtered)))), cmap = 'gray')
plt.tight_layout()

plt.savefig('images_2/result2_1_laplace')
plt.show()


# ### Imagem #2

# In[194]:


img = cv2.imread("images_2/sharpening_4.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap = 'gray')


# In[195]:


filtered, freq_filter, img_filter = butter_filter(img, D_0 = 20, n = 5)
# filtered, freq_filter, img_filter = butter_filter(filtered, D_0 = 20)
plt.imshow(freq_filter, cmap = 'gray')


# In[196]:


fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][0].set_title('Original Image')
axs[1][0].imshow(20*np.log10(np.abs(fftshift(fft2(img)))), cmap = 'gray', vmin = 0, vmax = 255)


# axs[0][1].imshow(img_filter, cmap = 'gray', vmin = 0, vmax = 255)
# axs[0][1].set_title('Filtered Image')
# axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(img_filter)))), cmap = 'gray', vmin = 0, vmax = 255)
# plt.tight_layout()
# plt.show()

axs[0][1].imshow(filtered, cmap = 'gray')
axs[0][1].set_title('Filtered Image')
axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(filtered)))), cmap = 'gray', vmin = 0, vmax = 255)
plt.tight_layout()

plt.savefig('images_2/result2_2_butter')
plt.show()


# In[197]:


filtered, img_filter = lapl_filter(img)

fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][0].set_title('Original Image')
axs[1][0].imshow(20*np.log10(np.abs(fftshift(fft2(img)))), cmap = 'gray', vmin = 0, vmax = 255)


# axs[0][1].imshow(img_filter, cmap = 'gray', vmin = 0, vmax = 255)
# axs[0][1].set_title('Filtered Image')
# axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(img_filter)))), cmap = 'gray', vmin = 0, vmax = 255)
# plt.tight_layout()
# plt.show()

axs[0][1].imshow(filtered, cmap = 'gray', vmin = 0, vmax = 255)
axs[0][1].set_title('Filtered Image')
axs[1][1].imshow(20*np.log10(np.abs(fftshift(fft2(filtered)))), cmap = 'gray')
plt.tight_layout()

plt.savefig('images_2/result2_2_laplace')
plt.show()


# ## 3. Filtragem de Ruído
# 
# **Banco de imagens:** [Real-world Noisy Images ](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
# 
# **Método:** Filtro de média aritimética

# In[13]:


def avg_filter(img, window = [5, 5]):
    if(len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row, col = img.shape

    new_img = np.zeros(img.shape)
    rectx = int(np.floor(window[0] / 2))
    recty = int(np.floor(window[1] / 2))

    for i in np.arange(rectx, row - np.floor(rectx/2 + 1)):
        for j in np.arange(recty, col - np.floor(recty + 1)):
            i = int(i)
            j = int(j)
#             print(i, j)
            avg_filter = img[((i) - rectx): (i + rectx + 1), (j - recty): (j + recty + 1)]
#             avg = np.sum(np.asarray(avg_filter)) / (rectx * recty)
            avg = cv2.mean(avg_filter)
#             print(avg[0])
            new_img[i][j] = np.floor(avg[0])

    return new_img.astype(np.uint8)


# In[153]:


img = cv2.imread("images_3/avg_filter_4.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_avg = avg_filter(img, window = [5, 5])

fig, axs = plt.subplots(1, 2, figsize = [10, 10])

axs[0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0].set_title('Original Image')

axs[1].imshow(img_avg, cmap = 'gray', vmin = 0, vmax = 255)
axs[1].set_title('Filtered Image')

plt.tight_layout()
plt.savefig("images_3/results3_1")
plt.show()


# In[154]:


img = cv2.imread("images_3/avg_filter_5.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_avg = avg_filter(img, window = [7, 7])

fig, axs = plt.subplots(1, 2, figsize = [10, 10])

axs[0].imshow(img, cmap = 'gray', vmin = 0, vmax = 255)
axs[0].set_title('Original Image')

axs[1].imshow(img_avg, cmap = 'gray', vmin = 0, vmax = 255)
axs[1].set_title('Filtered Image')

plt.tight_layout()
plt.savefig("images_3/results3_2")
plt.show()


# ## 4. Granulometria
# 
# 
# **Banco de imagens:** [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
# 
# **Método:** Segmentação por textura

# In[177]:


def smoothing(img, kernel, background = 0):
    if(background):
        return cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cv2.MORPH_OPEN, kernel)    
    else:
        return cv2.morphologyEx(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

def pixels_diff(img, max_size = 40, plot = True, smooth = True, 
                smooth_k_size = (4, 4), kernel_type = cv2.MORPH_RECT,
               background = 0):
    
    if(smooth):
        kernel = cv2.getStructuringElement(kernel_type, smooth_k_size)
        img = smoothing(img, kernel, background = background)
    
    diff_smooth = np.zeros(max_size)
    pixel_sum = np.zeros(max_size)
    last_pixel = 0

    for k_size in range(1, max_size):
        kernel = cv2.getStructuringElement(kernel_type,(k_size,k_size))   
        pixel_sum[k_size] = np.sum(np.asarray(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)))    

    for k_size in range(1, max_size):
        diff_smooth[k_size] = np.abs(np.subtract(pixel_sum[k_size], pixel_sum[k_size-1]))
    diff_smooth[1] = 0

    if(plot):
        plt.figure()
#         fig, axs = plt.subplots(1, 2, figsize = [10, 10])
        plt.stem(diff_smooth)
        plt.savefig("images_4/result4_2_graph")
        plt.show()
        plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.show()
        
#     return diff_smooth


# ### Image #1

# In[175]:


img = cv2.imread("images_4/gradient_7.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[176]:


pixels_diff(img, kernel_type = cv2.MORPH_ELLIPSE, max_size = 45)


# Como os picos foram detectados nos pontos 32, 34 e 36, estimasse que diâmetro aproximado dos raios serão esses. Com isso, serão feitos processos de fechamentos com kernels desses valores para tentar separar os componentes por seus tamanhos.

# In[173]:


plt.figure()
fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(32, 32))
axs[0][1].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(36, 36))
axs[1][0].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45, 45))
axs[1][1].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')

plt.tight_layout()
plt.savefig("images_4/result4_1")
plt.show()


# In[81]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(34, 34))
plt.imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))


# In[82]:


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(36, 36))
plt.imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel))


# ### Image #2

# In[181]:


img = cv2.imread("images_4/gradient_1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap = 'gray')


# In[182]:


pixels_diff(img, kernel_type = cv2.MORPH_ELLIPSE, background = 1)


# Raios: 14, 24, 31

# In[183]:


plt.figure()
fig, axs = plt.subplots(2, 2, figsize = [10, 10])
axs[0][0].imshow(img, cmap = 'gray')

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(14, 14))
axs[0][1].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(24, 24))
axs[1][0].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(31, 31))
axs[1][1].imshow(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel), cmap = 'gray')

plt.tight_layout()
plt.savefig("images_4/result4_2")
plt.show()


# In[ ]:




