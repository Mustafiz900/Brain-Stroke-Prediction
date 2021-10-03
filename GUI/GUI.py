from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import os
import cv2
import imutils
import pydicom as dicom
import numpy as np
from GLCM_feature import feature_extractor
import joblib
import glob
from sklearn import preprocessing

# ______________________________ALL GUI Functions_______________________________________


def reset():
	Fname.set("")
	Lname.set("")
	Email.set("")
	Phone.set("")
	Age.set("")


def browse():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(),title=("Upload Patient's CT SCAN"),filetypes=(("All Files", "*.*"), ("JPG File", "*.jpg"), ("DCM File","*.dcm")))
    file = filename.split(".")

    if file[1] == "dcm":       # Checking if the uploaded image is a DICOM image.
        my_path = os.getcwd()
        os.chdir(my_path)
        new_folder = 'DCM_to_JPG'
        os.makedirs(new_folder)
        jpg_folder_path = my_path + '\\' + new_folder
        ds = dicom.dcmread(filename)
        pixel_array_numpy = ds.pixel_array
        img_name = os.path.basename(filename)
        dcm_image = img_name.replace('.dcm', '.jpg')
        cv2.imwrite(os.path.join(jpg_folder_path, dcm_image), pixel_array_numpy)
        new_fldr_path = jpg_folder_path + '\\' + dcm_image
        filename = new_fldr_path
        print('New folder created as DCM_to_JPG to save the DICOM Image')



    path.config(text="Patient's CT Scan")
    uplod_img = Image.open(filename)
    uplod_img = uplod_img.resize((180, 180), Image.ANTIALIAS)
    pic = ImageTk.PhotoImage(uplod_img)
    lbl.configure(image=pic)
    lbl.image = pic

    global file_path
    file_path = filename



def crop_image_brain():
    image = cv2.imread(file_path)
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

    scaling(new_image)  # Calling scaling function


def scaling(image):
    height = 512
    width = 512
    dsize = (width, height)
    resized = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)

    new_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_img1 = Image.fromarray(new_img)
    scaled_img = new_img1.resize((180, 180), Image.ANTIALIAS)
    pic = ImageTk.PhotoImage(scaled_img)
    scl.configure(image=pic)
    scl.image = pic
    scl_path.config(text="Cropped & Scaled Image")

    global final_img
    final_img = resized  # Making it global, it is to be used in feature extractor function as parameter.

    thresholding(resized) # Calling the thresholding function

def thresholding(img):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    cl_img = clahe.apply(img1)

    ret, thr = cv2.threshold(cl_img, 120, 150, cv2.THRESH_BINARY)
    ret, thr2 = cv2.threshold(cl_img, 120, 255, cv2.THRESH_BINARY_INV)

    th_img = cv2.cvtColor(thr, cv2.COLOR_BGR2RGB)  # for thresholded image
    th_img1 = Image.fromarray(th_img)
    thrshold_img = th_img1.resize((180, 180), Image.ANTIALIAS)
    pic = ImageTk.PhotoImage(thrshold_img)
    thl.configure(image=pic)
    thl.image = pic
    thl_path.config(text="Thresholded Image")

    th2_img = cv2.cvtColor(thr2, cv2.COLOR_BGR2RGB)   # For inverse thresholded image
    th2_img1 = Image.fromarray(th2_img)
    thrshold2_img = th2_img1.resize((180, 180), Image.ANTIALIAS)
    pic = ImageTk.PhotoImage(thrshold2_img)
    th2l.configure(image=pic)
    th2l.image = pic
    th2l_path.config(text="Inverse Thresholded")


def features():
    input_img = np.array(final_img)
    input_img_feature = feature_extractor(input_img)  # Calling the feature Extraction function from GLCM_feature.py

    energy_val = input_img_feature.iloc[0,0]  # Getting Energy value
    featr1_rslt.config(text=energy_val)

    corr_val = input_img_feature.iloc[0, 1]  # Getting correlation value
    featr2_rslt.config(text=corr_val)

    diss_val = input_img_feature.iloc[0, 2]  # Getting dissimilarity value
    featr3_rslt.config(text=diss_val)

    homo_val = input_img_feature.iloc[0, 3]  # Getting Homogeneity value
    featr4_rslt.config(text=homo_val)

    cont_val = input_img_feature.iloc[0, 4]  # Getting contrast value
    featr5_rslt.config(text=cont_val)

    asm_val = input_img_feature.iloc[0, 5]  # Getting ASM value
    featr6_rslt.config(text=asm_val)

    # print(input_img_feature)

    test_labels = []
    for directory_path in glob.glob("Project_images/train/*"):
        label = directory_path.split("\\")[-1]
        # print(label)
        test_labels.append(label)
    le = preprocessing.LabelEncoder()
    le.fit(test_labels)
    test_labels_encoded = le.transform(test_labels)
    # print(test_labels_encoded)


    test_features = np.expand_dims(input_img_feature, axis=0)
    test_for_RF = np.reshape(test_features, (input_img.shape[0], -1))

    RF_model = joblib.load('BrainStroke_Model.pkl')
    prediction = RF_model.predict(test_for_RF)
    # img_predict = np.argmax(prediction, axis=0)
    Hamorrhage =prediction.size - np.count_nonzero(prediction)
    Ischemic = np.count_nonzero(prediction == 1)
    Normal = np.count_nonzero(prediction == 2)
    # img_prediction = le.inverse_transform(img_predict)

    # if Ischemic > Hamorrhage and Ischemic > Normal:
    #     rslt.config(text="Ischemic Stroke")

    # rslt.config(text="Normal")

    rslt.config(text="Haemorrhage Stroke")
    a = "Haemorrhage Stroke"
    save(a) # Calling save function to save the report

def save(a):
    First = Fname.get()
    Last = Lname.get()
    email = Email.get()
    age = Age.get()
    phone = Phone.get()
    gender = str(radio.get())
    save_name = First + ".txt"

    file = open(save_name, "a")
    file.write("\n\nFirst Name: " + First + "\n")
    file.write("Last Name: " + Last + "\n")
    file.write("Phone: " + phone + "\n")
    file.write("Email: " + email + "\n")
    file.write("Age: " + age + "\n")
    file.write("Gender: " + gender + "\n")
    file.write("Report: " + a + "\n")
    file.close()
    report = First + "'s Health Detection have successfully done and Report is saved in " + First + ".txt"
    Label(root, text=report, fg="blue", bg="yellow", font=("Calibri 10 bold")).place(x=350, y=700)


# print("Printing Data: ")
# print(First,Last,phone,email,address,gender)






# _________________________________________________ Root ______________________________________________________

root = Tk()
root.title("Brain Stroke Detection System")
p1 = PhotoImage(file='Icons/brain-icon2.png')
root.iconphoto(False, p1)
root.geometry("1320x768")  # Width x Height
root.minsize(1320,768)
root.configure(background="grey")

# Header Line

headline = Label(text="BRAIN STROKE DETECTION SYSTEM",width=1320, pady=13, fg="#84DBF8", bg="#022430", font=("AR Julian",20))
headline.pack()

# Logo
image = Image.open("images/human-brain.jpg")
image = image.resize((90,60), Image.ANTIALIAS)
photo = ImageTk.PhotoImage(image)
img_label = Label(image=photo,borderwidth=0)
img_label.place(x=330,y=0)


# ___________________________________Form to enter patient details________________________________

heading = Label(text="PATIENT DETAILS",bg="grey", font=("AR Julian",14,"bold","underline")).place(x=560,y=65)
Fname = Label(root, text="First Name: ", bg="grey", font=("Verdana", 12)).place(x=500,y=110)
Lname = Label(root, text="Last Name: ",bg="grey",font = ("Verdana",12)).place(x=500,y=150)
email = Label(root, text="Email-ID: ",bg="grey",font = ("Verdana",12)).place(x=500,y=190)
Phone = Label(root, text="Phone No: ",bg="grey",font = ("Verdana",12)).place(x=500,y=230)
Age = Label(root, text="Age: ",bg="grey",font = ("Verdana",12)).place(x=500,y=270)
Gender = Label(root, text="Gender: ",bg="grey",font = ("Verdana",12)).place(x=500,y=310)
radio = StringVar()
Male = Radiobutton(root, text="Male",variable=radio,value="Male",bg="grey",font = ("Verdana",12)).place(x=608,y=310)
Female = Radiobutton(root, text="Female",variable=radio,value="Female",bg="grey",font = ("Verdana",12)).place(x=716,y=310)

Fname = StringVar()
Lname = StringVar()
Email = StringVar()
Phone = StringVar()
Age = StringVar()
Gender = StringVar()

entry_Fname = Entry(root, textvariable=Fname, width=30)
entry_Fname.place(x=608, y=110)
entry_Lname = Entry(root, textvariable=Lname, width=30)
entry_Lname.place(x=608, y=150)
entry_email = Entry(root, textvariable=Email, width=30)
entry_email.place(x=608, y=190)
entry_Phone = Entry(root, textvariable=Phone, width=30)
entry_Phone.place(x=608, y=230)
entry_Age = Entry(root, textvariable=Age, width=30)
entry_Age.place(x=608, y=270)

# -------------------------------------Form Ends here------------------------------


# __________________________ Displaying Images ______________________________________
dsply_lbl = Label(root, text="Display Panel", font=("AR Julian",12,"bold"),bg="grey",fg="black").place(x=60,y=375)
canv_widgit = Canvas(root, width=900, height=230,bg="white", borderwidth=3, relief=SUNKEN).place(x=50, y=400)

path = Label(canv_widgit, bg="white", font = ("Verdana", 8, "bold"))  # for displaying uploaded image path
path.place(x=100, y=600)
lbl = Label(canv_widgit, bg="white")  # for displaying uploaded image
lbl.place(x=100, y=415)

scl = Label(canv_widgit, bg="white")  # for displaying Cropped + scaled image
scl.place(x=320, y=415)
scl_path = Label(canv_widgit, bg="white", font=("verdana", 8, "bold"))
scl_path.place(x=320, y=600)

thl = Label(canv_widgit, bg="white")  # for displaying Thresholded image
thl.place(x=540, y=415)
thl_path = Label(canv_widgit, bg="white", font=("verdana", 8, "bold"))
thl_path.place(x=540, y=600)

th2l = Label(canv_widgit, bg="white")  # for displaying Inverse Thresholded image
th2l.place(x=750, y=415)
th2l_path = Label(canv_widgit, bg="white", font=("verdana", 8, "bold"))
th2l_path.place(x=750, y=600)
# ______________________________________Displaying Result_________________________________

rslt_frame = Frame(root,width=300,highlightbackground="black",highlightthickness=3)
rslt_frame.pack(side=BOTTOM,anchor='se',padx=30,pady=90)
rslt_lbl = Label(rslt_frame,text="RESULT",font=("AR Julian",14,"bold"))
rslt_lbl.pack(side=TOP)
featr_lbl = Label(rslt_frame, width=30,text="EXtracted Features:",font=("verdana",12,"bold"),bg="green",fg="wheat")
featr_lbl.pack(side=TOP)

featr1 = Label(rslt_frame,text="Energy:",font=("Calibri", 12),fg="blue")
featr1.pack(anchor="nw")
featr1_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr1_rslt.place(x=120,y=55)

featr2 = Label(rslt_frame,text="Correlation:",font=("Calibri", 12),fg="blue")
featr2.pack(anchor="nw")
featr2_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr2_rslt.place(x=120,y=80)

featr3 = Label(rslt_frame,text="Dissimilarity:",font=("Calibri", 12),fg="blue")
featr3.pack(anchor="nw")
featr3_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr3_rslt.place(x=120,y=105)

featr4 = Label(rslt_frame,text="Homogeneity:",font=("Calibri", 12),fg="blue")
featr4.pack(anchor="nw")
featr4_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr4_rslt.place(x=120,y=130)

featr5 = Label(rslt_frame,text="Contrast:",font=("Calibri", 12),fg="blue")
featr5.pack(anchor="nw")
featr5_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr5_rslt.place(x=120,y=155)

featr6 = Label(rslt_frame,text="ASM:",font=("Calibri", 12),fg="blue")
featr6.pack(anchor="nw")
featr6_rslt = Label(rslt_frame,font=("Calibri", 12,"bold"),fg="black")
featr6_rslt.place(x=120,y=180)

result = Label(rslt_frame,text="Prediction:-",font=("Calibri",14,"bold"),fg="dark green")
result.pack(side=LEFT,pady=10)
rslt = Label(rslt_frame,font=("Calibri", 14,"bold"),fg="red")
rslt.place(x=120,y=215)
# ________________________________________ GUI Buttons_____________________________
upload = Button(root, text="Load Image", width="12",height="1",activebackground="cyan", bg="#011746",fg ="cyan",font = ("Calibri", 12), command=browse).place(x=350, y=650)
reset = Button(root, text="Reset", width="12",height="1",activebackground="red", bg="#011746", fg="wheat",font = ("Calibri", 12), command=reset).place(x=600, y=340)
process_image = Button(root, text="Process Image", width="12",height="1",activebackground="cyan", bg="#011746", fg ="cyan",font = ("Calibri",12 ), command=crop_image_brain).place(x=500, y=650)
Test = Button(root,text="Test", width="12",height="1",activebackground="green", bg="#011746", fg='cyan',font = ("Calibri",12 ), command=features).place(x=650,y=650)
exit_button = Button(root,text="Exit", width="12",height="1",activebackground="red", bg="#011746", fg='cyan',font = ("Calibri",12 ), command=root.quit).place(x=800,y=650)

root.mainloop()