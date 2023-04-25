import matplotlib.pyplot as plt

sequentialResults = {}
with open("sequentialOutput.txt", "r") as f:
    for line in f:
      row = line.split(" ")
      sequentialResults[row[0]] = float(row[1])

# print(sequentialResults)

ompResults = {}
with open("OmpOutput.txt", "r") as f:
    for line in f:
      row = line.split(" ")
      if not (row[1] in ompResults.keys()):
        ompResults[row[1]] = {str(row[0]) : float(row[2])}
      else:
        if not (row[0] in ompResults[row[1]].keys()):
          ompResults[row[1]][row[0]] = float(row[2])        

#print(ompResults)

speedUpOMP = {key: {} for key in ompResults.keys()}
for n in ompResults.keys():
  speedUpOMP[n][1] = 1.0
  for threads in ompResults[n].keys():
    speedUp = float(sequentialResults[n]) / float(ompResults[n][threads])
    speedUpOMP[n][int(threads)] = float(speedUp)
  speedUpOMP[n] = dict(sorted(speedUpOMP[n].items()))

print(speedUpOMP)    

'''
x = []
y = []


for n, seconds in sequentialResults.items():
    x.append(int(n))
    y.append(float(seconds))

plt.plot(x, y, 'o', label="Secuencial")
plt.xlabel("n")
plt.ylabel("Segundos")
plt.title("Tiempo de respuesta")
plt.legend()
plt.show()




plot_types = {"32":{"style": "b-"}, "64":{"style": "r--"}, "128":{"style": "g-."}, "256":{"style": "y-"}, "512":{"style":"m--"}, "1024":{"style":"k-."}}
for n in ompResults.keys():
  data = ompResults[n]
  threads = list(int(a) for a in data.keys())
  threads.insert(0,1)
  seconds = list(float(b) for b in data.values())
  seconds.insert(0,float(sequentialResults[n]))
  
  plt.plot(threads, seconds, plot_types[n]["style"])

  plt.xlabel("Hilos")
  plt.ylabel("Segundos")
  plt.title(f"OMP Response times for n={n}")
  plt.legend()
  plt.show()

'''


plot_types = {"32":{"style": "b-"}, "64":{"style": "r--"}, "128":{"style": "g-."}, "256":{"style": "y-"}, "512":{"style":"m--"}, "1024":{"style":"k-."}}
for n in speedUpOMP.keys():
  data = speedUpOMP[n]
  threads = list(data.keys())
  seconds = list(data.values())
  
  plt.plot(threads, seconds, plot_types[n]["style"])

  plt.xlabel("Hilos")
  plt.ylabel("Speed up")
  plt.title(f"OMP Speed Ups para n={n}")
  plt.show()


