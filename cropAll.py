'''
 This Code Crops all the images in the folder (Project_images) that has two sub folders (train & validation)
 and saves it into another folder named (Project_images_1) having sub folders (train1 & validation1).

 All the images in the folder (Project_images) are appropriately cropped and copied to the corresponding
 folder paths in (Project_images_1).
 '''

import cv2
import glob
import imutils
import os


def crop_contour_brain_img(imgpath):
    # Creating the new folder for saving cropped images
    path = os.getcwd()
    os.chdir(path)
    new_folder = 'Project_images_1'
    os.makedirs(new_folder)

    # Creating folder inside "project_images_1" folder
    path2 = path+'\\'+new_folder
    os.chdir(path2)
    for i in range(1,3):
        if i == 1:
            new_folder1 = 'train1'
            os.makedirs(new_folder1)
        else:
            new_folder2 = 'validation1'
            os.makedirs(new_folder2)

    # Creating folders inside "train1" folder
    path3 = path2 + '\\' + new_folder1
    os.chdir(path3)
    for j in range(1,4):
        if j == 1:
            os.makedirs('Hamorrhage Stroke')
        elif j == 2:
            os.makedirs('Ischemic Stroke')
        else:
            os.makedirs('Normal')

    # Creating folders inside "train1" folder
    path4 = path2 + '\\' + new_folder2
    os.chdir(path4)
    for k in range(1,4):
        if k == 1:
            os.makedirs('Hamorrhage Stroke')
        elif k == 2:
            os.makedirs('Ischemic Stroke')
        else:
            os.makedirs('Normal')

    os.chdir(path)
    


    # Cropping all the images in the given diretory path and saving to newly created folder

    for directory_path in glob.glob(imgpath):
        img_number = 1
        print(directory_path)
        for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
            # print(img_path)
            image = cv2.imread(img_path)
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grayscale = cv2.GaussianBlur(grayscale, (5, 5), 0)
            threshold_image = cv2.threshold(grayscale, 45, 255,cv2.THRESH_BINARY)[1]
            threshold_image = cv2.erode(threshold_image,None,iterations=2)
            threshold_image = cv2.dilate(threshold_image, None, iterations=2)

            contour = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = imutils.grab_contours(contour)
            c = max(contour, key=cv2.contourArea)

            extreme_pnts_left = tuple(c[c[:, :, 0].argmin()][0])
            extreme_pnts_right = tuple(c[c[:, :, 0].argmax()][0])
            extreme_pnts_top = tuple(c[c[:, :, 1].argmin()][0])
            extreme_pnts_bottom = tuple(c[c[:, :, 1].argmax()][0])

            new_image = image[extreme_pnts_top[1]:extreme_pnts_bottom[1], extreme_pnts_left[0]:extreme_pnts_right[0]]

            if directory_path == r"Project_images\train\Hamorrhage Stroke":
                cv2.imwrite("Project_images_1/train1/Hamorrhage Stroke/" + str(img_number) + ".jpg", new_image)

            elif directory_path == r"Project_images\train\Ischemic Stroke":
                cv2.imwrite("Project_images_1/train1/Ischemic Stroke/" + str(img_number) + ".jpg", new_image)

            elif directory_path == r"Project_images\train\Normal":
                cv2.imwrite("Project_images_1/train1/Normal/" + str(img_number) + ".jpg", new_image)

            elif directory_path == r"Project_images\validation\Hamorrhage Stroke":
                cv2.imwrite("Project_images_1/validation1/Hamorrhage Stroke/" + str(img_number) + ".jpg", new_image)

            elif directory_path == r"Project_images\validation\Ischemic Stroke":
                cv2.imwrite("Project_images_1/validation1/Ischemic Stroke/" + str(img_number) + ".jpg", new_image)

            elif directory_path == r"Project_images\validation\Normal":
                cv2.imwrite("Project_images_1/validation1/Normal/" + str(img_number) + ".jpg", new_image)

            else:
                print('Cannot Copy Images')

            img_number +=1
        print('___________________________________________________________')

    print('New Directory created as Project_images_1 saving all the cropped images.')

if __name__ == '__main__':
     crop_contour_brain_img("fldrpath")  # providing the folder path as parameter to function
