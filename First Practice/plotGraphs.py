import matplotlib.pyplot as plt

sequentialResults = {}
with open("sequentialOutput.txt", "r") as f:
    for line in f:
      row = line.split(" ")
      sequentialResults[row[0]] = float(row[1])

#print(sequentialResults)

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

x = []
y = []

for n, seconds in sequentialResults.items():
    x.append(int(n))
    y.append(float(seconds))

plt.plot(x, y, label="Sequential")
plt.xlabel("n")
plt.ylabel("Seconds")
plt.title("Sequential Results")
plt.legend()
plt.show()



for n in ompResults.keys():
  data = ompResults[n]
  for threads_number, seconds in data.items():
      plt.plot(int(threads_number), float(seconds), 'o', label=f"{threads_number} threads")

  plt.xlabel("Threads")
  plt.ylabel("Seconds")
  plt.title(f"OMP Results for n={n}")
  plt.legend()
  plt.show()


