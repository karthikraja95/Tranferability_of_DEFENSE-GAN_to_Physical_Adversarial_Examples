import os
import cv2 
import autocrop
from PyQt5 import QtGui


cap = cv2.VideoCapture(0)
top = 150  # shape[0] = rows
bottom = top
left = 150  # shape[1] = cols
right = left
borderType = cv2.BORDER_CONSTANT
value = [0,0,0]


        


# cv2.namedWindow('image',WINDOW_NORMAL)
# app = QtGui.QApplication([])
# screen_resolution = app.desktop().screenGeometry()
# width, height = screen_resolution.width(), screen_resolution.height()
# # screen = screeninfo.get_monitors()[0]
# width, height = screen.width, screen.height

def generate_examples(input_path, output_path):
    for file in os.listdir(input_path):
        print (file)
        filepath = os.path.join(input_path, file)
        image = cv2.imread(filepath,1)

        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.moveWindow('image', 70,70)
        cv2.resizeWindow('image', 278,318)
        dst = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, value)
        cv2.imshow("image", dst)
        # ret, frame = cap.read()
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        # outputpath = os.path.join(output_path, file)
        # out = cv2.imwrite(outputpath, frame)
        print ("do something")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

generate_examples("/Users/developer/Desktop/Female", "/Users/developer/Desktop/output")



        

