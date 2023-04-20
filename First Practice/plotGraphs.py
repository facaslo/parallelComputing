import matplotlib.pyplot as plt

sequentialResults = {}
with open("sequentialOutput.txt", "r") as f:
    for line in f:
      row = line.split(" ")
      sequentialResults[row[0]] = float(row[1])

print(sequentialResults)

ompResults = {}
with open("OmpOutput.txt", "r") as f:
    for line in f:
      row = line.split(" ")
      if not (row[1] in ompResults.keys()):
        ompResults[row[1]] = {str(row[0]) : float(row[2])}
      else:
        if not (row[0] in ompResults[row[1]].keys()):
          ompResults[row[1]][row[0]] = float(row[2])        
print(ompResults)

