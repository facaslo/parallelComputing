import subprocess
sequentialExecutable = "sequential_matrix_mult.out"
ompExecutable = "omp_matrix_mult.out"

dimension = [32 , 64 , 128 , 256 , 512 , 1024]
threads = [2, 4, 6, 8, 16, 32]

# Sequential
def outputSequential():
  subprocess.run(["touch", "sequentialOutput.txt"])
  for n in dimension:
      command = ["./" + sequentialExecutable, str(n)]
      with open("sequentialOutput.txt", "a") as outfile:
          subprocess.run(command, stdout=outfile, text=True)

def outputOpenMp():
   subprocess.run(["touch", "OmpOutput.txt"])
   for n in dimension:
      for threadsNumber in threads:
        command = ["./" + ompExecutable, str(threadsNumber) , str(n)]
        with open("OmpOutput.txt", "a") as outfile:
            subprocess.run(command, stdout=outfile, text=True)
    

# outputSequential()
# outputOpenMp()