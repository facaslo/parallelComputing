import subprocess
import matplotlib.pyplot as plt

threads = [1, 2, 4, 6, 8, 10, 12, 14, 16]
iterations = int(1e9)

executables = ["pi_omp_for_reduce.out", "pi_omp_fs.out", "pi_omp_nfs.out", "pi_posix.out"]

results = {}
for executable in executables:
    executable_results = {}
    for thread in threads:
        command = ["./" + executable, str(thread), str(iterations)]
        output = subprocess.check_output(command)
        output = output.decode('utf-8').strip()
        outputLines = output.split('\n')        
        threadsNumber = int(outputLines[1].split(':')[1].strip())
        pi_value = float(outputLines[2].split(':')[1].strip())
        elapsed_time = float(outputLines[3].split(':')[1].strip().split(' ')[0])       
        executable_results[threadsNumber] = {'pi': pi_value, 'time': elapsed_time}
    results[executable] = executable_results

speedups = {}
for executable in executables:
    speeds = [1]
    times = [results[executable][t]['time'] for t in threads]    
    for i in range(1, len(times) ):
        speedup = times[0] / times[i]
        speeds.append(speedup)
    speedups[executable] = speeds



# plot the data for each executable
colors = ['red', 'blue', 'green', 'orange']
markers = ['o', 's', 'v', '^']
# fig, ax= plt.subplots()
fig, bx = plt.subplots()
for i, executable in enumerate(executables):
    '''
    times = [results[executable][t]['time'] for t in threads]
    ax.plot(threads, times, marker=markers[i], color=colors[i], label=executable)
    '''
    speeds = speedups[executable]
    bx.plot(threads, speeds, marker=markers[i], color=colors[i], label=executable)

# set plot title and axis labels
'''
ax.set_title(f"Tiempo de respuesta vs. Número de hilos para el cálculo de Pi usando la serie de Leibniz. Iteraciones:{iterations}")
ax.set_xlabel("Número de hilos")
ax.set_ylabel("Tiempo de respuesta (segundos)")
'''

bx.set_title(f"Speed up vs. Número de hilos para el cálculo de Pi usando la serie de Leibniz. Iteraciones:{iterations}")
bx.set_xlabel("Número de hilos")
bx.set_ylabel("Speed up")


# add legend and show plot
#ax.legend()
bx.legend()
plt.show()
