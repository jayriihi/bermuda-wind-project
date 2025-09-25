import numpy as np
import matplotlib.pyplot as plt

def load_grid(path):
    return np.loadtxt(path)

hs  = load_grid('../cases/hello/hs.out')
tm  = load_grid('../cases/hello/tm01.out')
dire= load_grid('../cases/hello/dir.out')

plt.figure()
plt.title('Hello SWAN — Significant Wave Height (m)')
plt.imshow(hs, origin='lower')
plt.colorbar(label='m')
plt.tight_layout()
plt.show()
