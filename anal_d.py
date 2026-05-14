txt = """
Gęstość (g/cm³)    | Objętość (Å³)      | Energia (eV)
0.2000             | 251561.45          | -1556.5731
0.3483             | 144461.03          | -2079.8463
0.4966             | 101323.36          | -3354.3243
0.6448             | 78024.41           | -5447.0666
0.7931             | 63437.24           | -8339.1107
0.9414             | 53445.29           | -11855.9401
1.0897             | 46172.67           | -15785.9402
1.2379             | 40642.24           | -19858.5020
1.3862             | 36294.94           | -23848.4487
1.5345             | 32787.78           | -27630.7275
1.6828             | 29898.70           | -31121.7865
1.8310             | 27477.52           | -34283.1335
1.9793             | 25419.10           | -37044.2845
2.1276             | 23647.59           | -39240.2711
2.2759             | 22106.92           | -41085.3844
2.4241             | 20754.71           | -42606.0922
2.5724             | 19558.40           | -43816.6751
2.7207             | 18492.48           | -44724.6176
2.8690             | 17536.74           | -45376.0005
3.0172             | 16674.93           | -45766.1673
3.1655             | 15893.86           | -45881.3267
3.3138             | 15182.69           | -45704.0352
3.4621             | 14532.43           | -45239.8556
3.6103             | 13935.59           | -44452.8135
3.7586             | 13385.84           | -43295.4134
3.9069             | 12877.81           | -41722.4250
4.0552             | 12406.94           | -39736.8452
4.2034             | 11969.29           | -37228.7318
4.3517             | 11561.46           | -34135.7726
4.5000             | 11180.51           | -30521.0408
"""

# MAKE PLOT

import matplotlib.pyplot as plt
import numpy as np

lines = txt.strip().split("\n")[2:]  # Pomijamy nagłówki
gestosci = []
objetosci = []
energie = []
for line in lines:
    parts = line.split("|")
    gestosc = float(parts[0].strip())
    objetosc = float(parts[1].strip())
    energia = float(parts[2].strip())
    
    gestosci.append(gestosc)
    objetosci.append(objetosc)
    energie.append(energia)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(gestosci, energie, marker='o')
plt.xlabel("Gęstość (g/cm³)")
plt.ylabel("Energia (eV)")
plt.title("Energia w funkcji gęstości")
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(gestosci, objetosci, marker='o')
plt.xlabel("Gęstość (g/cm³)")
plt.ylabel("Objętość (Å³)")
plt.title("Objętość w funkcji gęstości")
plt.grid()
plt.tight_layout()
plt.show()