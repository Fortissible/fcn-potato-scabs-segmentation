import os
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import glob

'''
images_rgb = np.zeros((12, 480, 640, 3))
images_annot = np.zeros((12, 480, 640, 3))
for idx, img_path in enumerate("PotatoInGas/rgb/rgb_train"):
    img = cv2.imread(img_path)
    cv2.imshow(img,"coek")
    images_rgb[idx, :, :, :] = img

for idx, img_path in enumerate("PotatoInGas/rgb/annot_train"):
    img = cv2.imread(img_path)
    images_annot[idx, :, :, :] = img

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
])'''

augmentation1 = iaa.Sequential([
    iaa.Flipud(1.0)
])

def augment_seg(imgs_rgb, imgs_annot, augmentation1):
    images_rgb_aug = []
    images_annot_aug = []
    for idx in range(len(imgs_rgb)):
        images_rgb_aug.append(augmentation1(images=imgs_rgb[idx]))
        images_annot_aug.append(augmentation1(images=imgs_annot[idx]))
    return images_rgb_aug, images_annot_aug


images_rgb = []
images_annot = []

images_rgb_path = glob.glob("PotatoInGas/rgb/rgb_train/*.png")
images_annot_path = glob.glob("PotatoInGas/rgb/annot_train/*.png")

for image_path in images_rgb_path:
    images_rgb.append(cv2.imread(image_path))
    # images_rgb.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

for image_path in images_annot_path:
    images_annot.append(cv2.imread(image_path))
    # images_annot.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

imgs_rgb_aug, imgs_annot_aug = augment_seg(images_rgb, images_annot, augmentation1)

'''
for idx in range(len(imgs_annot_aug)):
    # cv2.imshow("Image", imgs_rgb_aug[idx])
    cv2.imwrite("PotatoInGas/rgb/aug_rgb/crop{}-aug.png".format(idx), imgs_rgb_aug[idx])
    # cv2.imshow("Image-augmented", imgs_annot_aug[idx])
    cv2.imwrite("PotatoInGas/rgb/aug_annot/crop{}-aug.png".format(idx), imgs_annot_aug[idx])
    # cv2.waitKey(0)


for idx in range(len(imgs_annot_aug)):
    cv2.imwrite("PotatoInGas/rgb/aug_rgb/crop{}-aug.png".format(idx+15), images_rgb[idx])
    cv2.imwrite("PotatoInGas/rgb/aug_annot/crop{}-aug.png".format(idx+15), images_annot[idx])
'''

for idx in range(len(imgs_annot_aug)):
    cv2.imwrite("PotatoInGas/rgb/aug_rgb/crop{}-aug.png".format(idx), imgs_rgb_aug[idx])
    (a, b, c) = cv2.split(imgs_annot_aug[idx])

    c_arr = np.array(c)
    kentang = np.array(b)
    new_arr = np.array(a)
    zeros = np.zeros((480,640,3),dtype=np.uint8)
    scabs = cv2.bitwise_and(kentang,kentang,dst=None,mask=c_arr)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            if (kentang[i][j] >= 1):
                zeros[i][j] = 1
            if (scabs[i][j] >= 1):
                zeros[i][j] = 2
            '''
            if (kentang[i][j] >= 1) :
                new_arr[i][j] = 2
            if (scabs[i][j] >= 1):
                new_arr[i][j] = 3
            if (new_arr[i][j] == 0):
                new_arr[i][j] = 1   '''
    cv2.imwrite("PotatoInGas/rgb/aug_annot/crop{}-aug.png".format(idx), zeros)


for idx in range(len(imgs_annot_aug)):
    cv2.imwrite("PotatoInGas/rgb/aug_rgb/crop{}-aug.png".format(idx+15), images_rgb[idx])
    (a, b, c) = cv2.split(images_annot[idx])
    c_arr = np.array(c)
    kentang = np.array(b)
    new_arr = np.array(a)
    zeros = np.zeros((480, 640, 3), dtype=np.uint8)
    scabs = cv2.bitwise_and(kentang, kentang, dst=None, mask=c_arr)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            if (kentang[i][j] >= 1):
                zeros[i][j] = 1
            if (scabs[i][j] >= 1):
                zeros[i][j] = 2
                '''
            if (kentang[i][j] >= 1) :
                new_arr[i][j] = 2
            if (scabs[i][j] >= 1):
                new_arr[i][j] = 3
            if (new_arr[i][j] == 0):
                new_arr[i][j] = 1'''

    cv2.imwrite("PotatoInGas/rgb/aug_annot/crop{}-aug.png".format(idx+15), zeros)
