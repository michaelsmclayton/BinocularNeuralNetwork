import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Stereogram dimensions
stereogramWidth = 30
stereogramHeight = 30

# Generate random stereogram (zeros and ones)
right = np.random.randint(2, size=[stereogramHeight,stereogramWidth])
divider = np.zeros((stereogramHeight, 3))
left = right.copy()

#add pop-out
x, y, w, h = stereogramWidth/4, stereogramHeight/4, stereogramWidth/2, stereogramHeight/2
shift =  - w/20 + 1
print shift
shading = 0
right[y:y+h,x-shift:x+w-shift] = left[y:y+h,x:x+w] + shading
left[y:y+h,x:x+w] += shading
# left[y:y+h,x:x+w] +=1
#  = np.ones((30,30)) #np.random.randint(2, size=[10,20])

combo = np.concatenate((left, divider, right), axis = 1)
plt.figure()
plt.imshow(combo, cmap='Greys',  interpolation='nearest', aspect='equal')
plt.axis('off')
plt.show()