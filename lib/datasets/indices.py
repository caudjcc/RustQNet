import numpy as np

EPS = 1e-32

  
def RSIndex(image):
    if image.ndim == 3 and image.shape[0] in range(1,20):
        #print("CHW format")
        image=image.transpose((1, 2, 0))
    elif image.ndim == 3 and image.shape[2] in range(1,20):
        image=image
    else:
        raise ValueError("Unknown format for VI calculate.")
        #print("Unknown format")
        
    PRI_560nm_531nm=NDVI(image[..., 3], image[..., 2])
    PRI2_560nm_531nm_705nm=PRI2(image[..., 3], image[..., 2], image[..., 6])        
    NPCI_668nm_444nm = NDVI(image[..., 5], image[..., 0])     
    YROI_650nm_444nm_560nm= YROI(image[..., 4], image[..., 0], image[..., 3])        
    DBSI_531nm_560nm_740nm_840nm = DBSI(image[..., 2], image[..., 3], image[..., 8], image[..., 9],)
    FCVI_560nm_444nm_705nm_531nm = FCVI(image[..., 3], image[..., 0], image[..., 6], image[..., 2],)
    EMBI_717nm_668nm_650nm_475nm = EMBI(image[..., 7], image[..., 5], image[..., 4], image[..., 1],)
    DBI_705nm_560nm_717nm_531nm = DBI(image[..., 6], image[..., 3], image[..., 7], image[..., 2],)
    ARI2_560nm_531nm_650nm = ARI2(image[..., 3], image[..., 2], image[..., 4])
    SWI_668nm_560nm_531nm = SWI(image[..., 5], image[..., 3], image[..., 2])
    PSRI_560nm_531nm_740nm = PSRI(image[..., 3], image[..., 2], image[..., 8])
    EBBI_560nm_531nm_840nm = EBBI(image[..., 3], image[..., 2], image[..., 9])
    DBSI_560nm_717nm_705nm_531nm = DBSI(image[..., 3], image[..., 7], image[..., 6], image[..., 2])
    
    VI_ALL = np.stack([PRI_560nm_531nm,PRI2_560nm_531nm_705nm,NPCI_668nm_444nm,
                       YROI_650nm_444nm_560nm, DBSI_531nm_560nm_740nm_840nm, FCVI_560nm_444nm_705nm_531nm,
                       EMBI_717nm_668nm_650nm_475nm, DBI_705nm_560nm_717nm_531nm, ARI2_560nm_531nm_650nm, 
                       SWI_668nm_560nm_531nm, PSRI_560nm_531nm_740nm, EBBI_560nm_531nm_840nm, DBSI_560nm_717nm_705nm_531nm,]).astype('float32')

    return VI_ALL.transpose((1, 2, 0))

#test
"""
from osgeo import gdal
dataset = gdal.Open(r"E:\Multimodal1-2023final\RGB_50m\ALL-CUT-Final3-20231013\MUL\20220423_RGB_W_50M_1.8cm_final2_45_2.tif")
im_data = dataset.ReadAsArray()
img_vi=RSIndex(im_data)
"""


def NDVI(band1, band2):
    return (band1 - band2) / (band1 + band2 + EPS)

def PRI2(band1, band2, band3):
    return (band1 - band2) / (band1 + band3 + EPS)

def YROI(band1, band2, band3):
    return (band1 - band2) / (band3 + EPS)    

def DBI(b, r, n, t1):
    index = (b - t1) / (b + t1 + EPS)
    index -= (n - r) / (n + r + EPS)
    return index

def DBSI(g, r, n, s1):
    index = (s1 - g) / (s1 + g + EPS)
    index -= (n - r) / (n + r + EPS)
    return index

def FCVI(b, g, r, n):
    return n - ((r + g + b) / 3.0)

def EMBI(g, n, s1, s2):
    item1 = NDVI(s1, s2 + n)
    item1 += 0.5
    item2 = NDVI(g, s1)
    return (item1 - item2 - 0.5) / (item1 + item2 + 1.5 + EPS)

def ARI2(g, re1, n):
    index = 1 / (g + EPS)
    index -= 1 / (re1 + EPS)
    index = index * n
    return index

def SWI(g, n, s1):
    num = g * (n - s1)
    denom = (g + n) * (n + s1)
    return num / (denom + EPS)

def PSRI(b, r, re2):
    return (r - b) / (re2 + EPS)

def EBBI(n, s1, t):
    num = s1 - n
    denom = (10.0 * ((s1 + t)**0.5))
    return num / (denom + EPS)



