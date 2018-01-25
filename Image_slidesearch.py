
### 8. Feature detection in the entire image tile

#import libraries
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im_age=cv2.imread('path')
im_age=cv2.cvtColor(im_age, cv2.COLOR_BGR2GRAY)

height=len(im_age[0])
width=len(im_age[1])

step = 10
AOI_coordinates = []

image_array = np.array(im_age).astype('uint8')
plt.figure(1, figsize = (15, 30))
plt.imshow(image_array, cmap='gray')
plt.show()

#clipping image to predict
def clipped(x, y):
    #Area Of Interst (AOI)
    AOI = np.arange(1*80*80).reshape(1, 80, 80)
    for i in range(80):
        for j in range(80):
            AOI[0][i][j] = '''picture_tensor'''[y+i][x+j]
    AOI = AOI.reshape([-1, 1, 80, 80])
    AOI = AOI.transpose([0,1,2,3])
    AOI = AOI / 255
    sys.stdout.write('\rX_coordinate:{0} Y_coordinate:{1}  '.format(x, y))
    return AOI

#to check distance
def dist(x, y, size, coordinates):
    result = True
    for point in coordinates:
        if x+size > point[0][0] and x-size < point[0][0] and y+size > point[0][1] and y-size < point[0][1]:
            result = False
    return result


try:
    for y in range(int((height-(80-step))/step)):
        for x in range(int((width-(80-step))/step) ):
            area = clipped(x*step, y*step)
            result = model.predict(area)
            if result[0][1] > 0.90 and dist(x*step,y*step, 80, coordinates):
                AOI_coordinates.append([[x*step, y*step], result])
                print(result)
                plt.imshow(area[0].squeeze())
                plt.show()
except IndexError:
    pass

#test image
im = np.array(Image.open('path2'), dtype=np.uint8)

def bounding_boxe(coordinates,im):
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    for v in coordinates:
        rect = patches.Rectangle((v[0][0],v[0][1]),80,80,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    return plt.show()

bounding_boxe(AOI_coordinates,im)
