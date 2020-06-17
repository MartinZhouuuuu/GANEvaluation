import matplotlib.pyplot as plt
x = [i*0.1 for i in range(1,10)]
y = [50,50, 61.95, 75.1, 82.05, 87.7, 90.65, 92.55, 93.9]

plt.plot(x, y)
plt.xlabel('Perturbation size')
plt.ylabel('Best val acc')
plt.savefig('acc-against-perturbation.png')