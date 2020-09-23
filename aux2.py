import cv2
import numpy as np
import pytesseract

img = cv2.imread('C:/Users/mdycz/Desktop/Programming/Projects/OpenCVReceiptReader/wal3.jpg')

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

image = img
image = remove_noise(image)
image = erode(image)
bgray = get_grayscale(image)
blurred1 = cv2.medianBlur(bgray, 7)
blurred2 = cv2.medianBlur(bgray, 71)
divided = np.ma.divide(blurred1, blurred2).data
normed = np.uint8(255*divided/divided.max())
th, thresh = cv2.threshold(normed, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)


# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Convert the image to gray scale 
#gray = get_grayscale(image)

# Performing OTSU threshold    TRUNC TOZERO
#thresh = thresholding(gray)

# Specify structure shape and kernel size.  
# Kernel size increases or decreases the area  
# of the rectangle to be detected. 
# A smaller value like (10, 10) will detect  
# each word instead of a sentence. 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Appplying dilation on the threshold image 
dilation = cv2.dilate(thresh, rect_kernel, iterations = 9) 
  
# Finding contours 
image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 


for cnt in contours:
    img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

'''
while True:
    cv2.imshow("title", cv2.resize(img, (540,960)))
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break





'''
# Creating a copy of image 
im2 = img.copy() 
  
# A text file is created and flushed 
file = open("recognized.txt", "w+") 
file.write("") 
file.close() 
  
# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w] 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = pytesseract.image_to_string(cropped) 
      
    # Appending the text into file 
    file.write(text) 
    file.write("\n") 
      
    # Close the file 
    file.close 
