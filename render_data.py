import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from process_data import create_house_from_name

data = np.load('data_bank/train_123.npy', allow_pickle=True).tolist()

reso = 64

rep = data['x']
meta = data['meta']
filename = data['filename']


fig, ax = plt.subplots()
ax.set_xlim(0,reso-1)
ax.set_ylim(reso-1,0)
norm = mcolors.TwoSlopeNorm(vmin=-32, vcenter=0, vmax=10)
ax.imshow(rep[0], cmap='RdBu', norm=norm)
plt.savefig('abc.png')

file_path = f"datasets/rplan/{filename}.json"
house = create_house_from_name(file_path, reso=reso)
house.render(fig_path='abc2.png', reso=reso)

