import cv2
import pytesseract
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Read image from which text needs to be extracted
img = cv2.imread("wal3.jpg")

# Preprocessing the image starts

# Convert the image to gray scale 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold    TRUNC TOZERO
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 
  
# Finding contours 
image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

for cnt in contours:
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

while True:
    cv2.imshow("title", cv2.resize(img, (540,960)))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
