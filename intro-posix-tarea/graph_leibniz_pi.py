import matplotlib.pyplot as plt


#times = [ 5.8527165050 , 7.8516028150,  8.6071964440, 5.9859088070, 4.9128250660, 5.6361765930, 6.2860867770, 5.4280511780, 4.8233634940]
threads = [1 , 2 ,4 , 6 , 8, 10, 12 , 14, 16]
times = [7.8047333390, 8.4789334530, 7.5133136250, 6.3303724820, 5.7309287740, 5.2104889620, 6.2329269720, 5.2478158220, 5.2697622350]



plt.scatter(threads, times)

#plt.title('Tiempo vs número de hilos: calculo de pi mediante serie de leibniz, 2 mil millones de iteraciones. Procesador Ryzen 3 5300U(4 cores- 8 hilos)-8gb ram')
plt.title('Tiempo vs número de hilos: calculo de pi mediante serie de leibniz, 2 mil millones de iteraciones. Procesador core i3 7100u(2 cores- 4 hilos)-8gb ram')
plt.xlabel('Número de hilos')
plt.ylabel('Tiempo en segundos')

plt.show()