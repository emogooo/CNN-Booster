import random
import math

dizi = [2, 5, 8, 13]

sayi = 13

mid = int(math.ceil(len(dizi) / 2)) - 1
bot = 0
top = len(dizi)

while True:
    if sayi < dizi[mid]:
        top = mid
        last = mid
    elif sayi > dizi[mid]:
        bot = mid
    else:
        print(dizi[mid])
        break

    lastMid = mid
    mid = int((top + bot) / 2)

    if lastMid == mid:
        print(dizi[last])
        break