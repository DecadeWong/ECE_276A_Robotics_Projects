from myfunctions import *

name_vs_predicted_dist_and_bbox = []
distances_estimated = []

direct_to_images = curr_directory + "/testset/"

image_list = os.listdir(direct_to_images)
if ".DS_Store" in image_list:
    image_list.remove(".DS_Store")


for image_name in image_list:
    dist = -1
    image = cv2.imread(direct_to_images+image_name)
    image = scipy.misc.imresize(image, (900, 1200))
    print (image_name)

    bl_x,bl_y,tr_x,tr_y,dist = my_algorithm(image,image_name)

    if len(dist) > 0:
        dist_estimated = dist[0]

        for z in range(0, len(bl_x)):
            name_vs_predicted_dist_and_bbox.append([image_name, str(dist_estimated), str(bl_x[z]), str(bl_y[z]), str(tr_x[z]), str(tr_y[z])])
    else:
        name_vs_predicted_dist_and_bbox.append([image_name, "Didn't Detect"])


    # For collecting data for regression
    # area_regression.append(area)
    # dist_inv_regression.append(1.0 / (np.float(image_name.replace(".png", ""))))

txt_file = open("output.txt","w")

for item in name_vs_predicted_dist_and_bbox:

    line = "Image Name: " + item[0] + "  " + "BottomLeftX: " + item[2] +", "+ "BottomLeftY: " + item[3] + ", " + "TopRightX: "+ item[4] + ", " + "TopRightY: "+ item[5]+ ", " + "Distance: " + item[1] + "\n"
    txt_file.write(line)
txt_file.close()