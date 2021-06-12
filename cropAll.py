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


def crop_contour_brain_img(path):

    for directory_path in glob.glob(path):
        img_number = 1
        #print(directory_path)
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
        #print('______________________________END_____________________________')

    return None

if __name__ == '__main__':
    
     crop_contour_brain_img(fldrPath)  # providing the folder path as parameter to function
