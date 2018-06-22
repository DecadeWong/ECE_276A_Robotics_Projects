from roipoly import roipoly
from collections import defaultdict
import pylab as pl
import os
import cv2
import time
import numpy as np
import itertools
dict_of_coordinates = defaultdict(list)
l = []
mask_name = "_Other"
curr_directory = os.curdir
direct_to_images = curr_directory + "/trainset/"
direct_to_masked_images = curr_directory + "/mask" + mask_name + "/"
images_names = os.listdir(direct_to_images)
train_image_names = images_names[0:int(len(images_names)*0.8)]
test_image_names = images_names[int(len(images_names)*0.8):-1]
def image_to_features(image, mask):
    # np.bool_(mask)
    x, y = np.nonzero(mask)
    feature = image[x,y,:]
    feature = np.transpose(feature)
    # r_channel = image[:,:,0]* np.bool_(mask)
    # g_channel = image[:,:,1]* np.bool_(mask)
    # b_channel = image[:,:,2]* np.bool_(mask)
    # test = img = cv2.merge([r_channel, g_channel, b_channel])
    # pl.imshow(test)
    return feature
    # time.sleep(10)

# features_all = [[0],[0],[0]]

flag = 0;
for image_name in train_image_names:
    l = []
    print(direct_to_images+image_name)
    image = cv2.imread(direct_to_images+image_name)
    image = image[:,:,::-1]
    # cv2.imshow('jj',image)
    # cv2.waitKey(1)
    count = 0
    while 1:
        l2 = []
        pl.imshow(image, interpolation='nearest')
        pl.colorbar()
        pl.title("left click: line segment         right click: close region")
        MyROI = roipoly(roicolor='r')
        # i = input()

        if len(MyROI.allxpoints) < 4:
            break
        mask = MyROI.getMask(image[:,:,1])
        feature = image_to_features(image, mask)
        if flag == 0:
            features_all = feature
            flag = 1
        else:
            features_all = np.concatenate((features_all,feature),axis=1)
        r_channel = image[:,:,0]* np.bool_(mask)
        g_channel = image[:,:,1]* np.bool_(mask)
        b_channel = image[:,:,2]* np.bool_(mask)

        # l2.append(MyROI.allxpoints)
        print feature
        print features_all
        mask = 1*mask
        np.save(direct_to_masked_images + image_name.replace(".png", "") +  "_" + str(count), mask)
        count += 1
    img_masked = cv2.merge([r_channel, g_channel, b_channel])
    # image_name_split = image_name.split(".")[0]
    cv2.imwrite(image_name.replace(".png","") + mask_name + ".png", img_masked)
    # np.save(direct_to_masked_images+image_name.replace(".png",""), mask)


        # print list(itertools.chain(*features_all))
        # l2.append(MyROI.allypoints)
        # l.append(l2)
    # dict_of_coordinates[image_name].append(l)



    # dict_of_coordinates[image_name].append(MyROI.allypoints)

    # dict_of_coordinates[image].append(l)
    # dict_of_coordinates[image].append(MyROI.allypoints)allypoints
    # time.sleep(0.5)


np.save(direct_to_masked_images+ "features_train" + mask_name, features_all)

