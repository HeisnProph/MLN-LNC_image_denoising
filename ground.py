import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.special import expit
import time
import os
import psutil

# make noise
def pixel_value(x, y, img, noise_rate=0.1):
  if np.random.rand() < noise_rate:
    return 1 - img[x, y]
  else:
    return img[x, y]



# markov network
def mln(img, noise_rate=0.1):
  start_time = time.time()
  size = img.shape[0]
  img = img*2-1
  # factor graph initial
  factors = {}
  for x in range(size):
    for y in range(size):
      # every pixel is a node
      factors[(x, y)] = {}
      # connect neighbor nodes
      for dx in [-1, 1]:
        if 0 <= x + dx < size:
          factors[(x, y)][(x + dx, y)] = 1
      for dy in [-1, 1]:
        if 0 <= y + dy < size:
          factors[(x, y)][(x, y + dy)] = 1
      # connect self node
      factors[(x, y)][(x, y)] = 2.1
  # add random noise
  noisy_img = np.array([[pixel_value(x, y, img, noise_rate) for y in range(size)] for x in range(size)])
  end_time = time.time()
  print(f"Time for constructing the factor graph: {end_time - start_time:.4f} sec")
  process = psutil.Process(os.getpid())
  print(f"RAM used for constructing the factor graph: {process.memory_info().rss / 1024 / 1024:.2f} MB")  # MB
  return factors, noisy_img


# define belief propagation algorithm
def belief_propagation(factors, noisy_img, tolerance=1e-3):
  """
  implement belief propagation algorithm, when belief change 
  smaller than tolerance, stop iteration

  Args:
    factors: factor graph。
    noisy_img: image with noise
    tolerance: tolerance

  Returns:
    beliefs: out put image after BP。
  """
  start_time = time.time()
  size = noisy_img.shape[0]
  noisy_img = np.where(noisy_img<0.5,-1,1)
  # initial belief
  beliefs = np.zeros((size, size))
  prev_beliefs = np.zeros((size, size))
  while(True):
    # update belief
    for x in range(size):
      for y in range(size):
        # calculate message from neighbor
        messages = np.array([factors[(x, y)][neighbor] * expit(beliefs[neighbor[0], neighbor[1]])*noisy_img[neighbor[0],neighbor[1]] for neighbor in factors[(x, y)]])
        # update belief
        beliefs[x, y] = expit(np.sum(messages))
    
    # check if converge
    diff = np.max(np.abs(beliefs - prev_beliefs))
    if diff < tolerance:
      break

    # update old belief
    prev_beliefs = np.copy(beliefs)
    
  end_time = time.time()
  print(f"Belief Propagation time cost: {end_time - start_time:.4f} sec")
  process = psutil.Process(os.getpid())
  print(f"BP process RAM cost: {process.memory_info().rss / 1024 / 1024:.2f} MB")  # MB unit
  
  return beliefs

# main
if __name__ == "__main__":
  # original image
  img = np.array(Image.open("final_project/dog_grey.jpg"))
  img = img/255
  # create factor graph and image with noise
  factors, noisy_img = mln(img)
  # belief propagation
  beliefs = belief_propagation(factors, noisy_img)
  # show result
  plt.subplot(1, 3, 1)
  plt.imshow(img, cmap="gray")
  plt.title("Original Image")
  plt.subplot(1, 3, 2)
  plt.imshow(noisy_img, cmap="gray")
  plt.title("Noisy Image")
  plt.subplot(1, 3, 3)
  plt.imshow(beliefs>0.75, cmap="gray")
  plt.title("Denoised Image")
  plt.show()