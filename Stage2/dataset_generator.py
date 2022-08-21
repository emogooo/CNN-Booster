import pandas as pd
import shutil
import os

def klasorleriOlustur(hastalik):
        try:
            os.makedirs("train")
        except FileExistsError:
            pass
        try:
            os.makedirs("test")
        except FileExistsError:
            pass
        try:
            os.makedirs("validation")
        except FileExistsError:
            pass
        try:
            os.makedirs("train/" + hastalik)
        except FileExistsError:
            pass
        try:
            os.makedirs("test/" + hastalik)
        except FileExistsError:
            pass
        try:
            os.makedirs("validation/" + hastalik)
        except FileExistsError:
            pass
        

def ayikla(tumResimler, csv, hastalik, train, test, validation):
    toplamResimAdedi = train + test + validation
    data = pd.read_csv(csv, usecols = ['Image Index','Finding Labels'])
    hastaliklar = data["Finding Labels"].tolist()
    resimler = data["Image Index"].tolist()
    resimList = list()
    idx = 0
    for i in hastaliklar:
        if i == hastalik:
            resimList.append(resimler[idx])
            if len(resimList) == toplamResimAdedi:
                break
        idx += 1
    klasorleriOlustur(hastalik)     
    klasor = "train/" + hastalik
    sayac = 0
    for i in resimList:
        if sayac == train:
            klasor = "test/" + hastalik
        elif sayac == train + test:
            klasor = "validation/" + hastalik
        shutil.copy((tumResimler +'/'+ i), klasor)
        sayac += 1

ayikla("E:/Github/NIH/images", "E:/Github/NIH/Data_Entry_2017.csv", "Infiltration", 2500, 1000, 2500)
ayikla("E:/Github/NIH/images", "E:/Github/NIH/Data_Entry_2017.csv", "No Finding", 2500, 1000, 2500)