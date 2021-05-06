import numpy as np
import matplotlib.pyplot as plt
import imageio

def window_plot(reward_list, WINDOW_SIZE=10000, INC=1):
  np_r = np.array(reward_list)
  r_w = []
  for idx in range(0, len(reward_list), INC):
    r_w.append(np_r[idx:idx+WINDOW_SIZE].mean())

  plt.plot(r_w)
  plt.show()

# this function expects a folder ./render_folder with files vis_step_[1...n]
# I_STEP takes files ./render_folder/vis_step_[1....I_STEP], converts them
# into a gif and stores them in ./render_gif/ as render_movie_{I_STEP}.gif
def create_gif(I_STEP):
  images = []

  with imageio.get_writer('./render_gif/render_movie_{}.gif'.format(I_STEP), mode='I') as writer:
    for filename in ['./render_folder/vis_step_{}.png'.format(i) for i in range(1, I_STEP+1)]:
        image = imageio.imread(filename)
        writer.append_data(image)
