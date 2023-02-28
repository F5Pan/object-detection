from keras.preprocessing.image import ImageDataGenerator
import cv2
from matplotlib import pyplot as plt
# Plot inline
from pylab import rcParams
rcParams['figure.figsize'] = 15, 15



folderName = 'B'
grayfolderName= 'BB'
picnum =11
namestr = 'b'

# 第一步：高斯
# oranginalimg = cv2.imread("C:\\Users\\wayne\\testImage\\drug14.jpg")
oranginalimg = cv2.imread("D:\\master\\images\\b3.jpg")
# oranginalimg = cv2.imread("D:\\master\\images\\1\\A1.jpg")
img = cv2.cvtColor(oranginalimg, cv2.COLOR_BGR2RGB)
#
plt.imshow(img)
print(img.shape)
img = img.reshape((1,) + img.shape)
print(img.shape)


#藥物的增生
# datagen = ImageDataGenerator(
#     zca_whitening=False,
#     rotation_range=30,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest')
#藥物的增生



datagen = ImageDataGenerator(
    zca_whitening=False,
    rotation_range=0.0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    shear_range=0.0,
    zoom_range=0.0,
    horizontal_flip=False,
    vertical_flip=True,
    fill_mode='nearest')


i = 0
for batch in datagen.flow(img, batch_size=100,
                          save_to_dir='D:\\master\\images\\background\\0', save_prefix='s', save_format='jpg'):
    plt.subplot(5, 4, 1 + i)
    plt.axis("off")

    augImage = batch[0]
    # augImage = augImage.astype('float32')
    # augImage /= 255
    plt.imshow(augImage)

    i += 1
    if i > 19:
        break  # otherwise the generator would loop indefinitely

print('end')