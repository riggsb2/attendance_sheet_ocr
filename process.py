import cv2
import re
import numpy as np
import pandas as pd
import pytesseract
from datetime import datetime


def gen_markers():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    markerImage = cv2.aruco.generateImageMarker(dictionary, 23, 200, 1)
    cv2.imwrite("marker23.png", markerImage)
    markerImage = cv2.aruco.generateImageMarker(dictionary, 21, 200, 1)
    cv2.imwrite("marker21.png", markerImage)


def find_corners_and_crop(img):
    ## Add rotation using the symbols as orientations
    
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    
    def get_markers_and_angle(image):
        detectorParams = cv2.aruco.DetectorParameters()

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)

        markerCorners, markerIds, rejectedCandidates = None, None, None
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(image)

        #print(markerCorners)
        #print(markerIds)
        
        out_image = np.copy(img)
        cv2.aruco.drawDetectedMarkers(out_image, markerCorners, markerIds)
        cv2.imshow('marker detection', out_image)
        cv2.waitKey(0)
        
        #marker corners go (tl, tr, br, bl)
        angles = []
        for i, mc in zip(markerIds, markerCorners):
            x = mc[0][0][0] - mc[0][3][0]  
            y = mc[0][0][1] - mc[0][3][1] 
            angle = np.arctan(x/y)
            angles.append(angle)
            if i[0]==23: left, top = map(int, mc[0][0])
            if i[0]==21: right, bottom = map(int, mc[0][2])

        angle = np.mean(angles)*(180/np.pi)
        return ((left, top, right, bottom), angle)
    
    (left, top, right, bottom), angle = get_markers_and_angle(img)

    if angle!=0:
        rotated_image = rotate_image(img, -angle)
        cv2.imshow('rotated', rotated_image)
        (left, top, right, bottom), angle = get_markers_and_angle(img)
        print(left, top, right, bottom)
        cropped_img = rotated_image[top:bottom, left:right]
        cv2.imshow('cropped', cropped_img)
        cv2.waitKey(0) 
    else:
        cropped_img = img[top:bottom, left:right]

    return cropped_img

grid_template = cv2.imread('templates\slide1.jpg')
grid_template = find_corners_and_crop(grid_template)

circle_template = cv2.imread('templates\slide2.jpg')
circle_template = find_corners_and_crop(circle_template)

test_image = cv2.imread('straight_on.jpg')
test_image = cv2.imread('PXL_20231213_184257873.MP.jpg')

test_image = find_corners_and_crop(test_image)
test_image = cv2.resize(test_image, (circle_template.shape[1], circle_template.shape[0]))

minDist = 1
param1 = 500 #500
param2 = 20 #200 #smaller value-> more false circles
minRadius = 5
maxRadius = 10 #10

gray_circles = cv2.cvtColor(circle_template, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray_circles, cv2.HOUGH_GRADIENT, 1, minDist, 
                        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

if False: # draw circles on template
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(circle_template, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Show result for testing:
    cv2.imshow('circles', circle_template)


if False: # Draw circles on test image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv2.circle(test_image, (i[0], i[1]), i[2], (0, 255, 0), 2)

    cv2.imshow('circles on test', test_image)
#cv2.waitKey(0)

# Next we need to detect the grid and translate that to a table
gray = cv2.cvtColor(grid_template, cv2.COLOR_BGR2GRAY)

kernel_size = 1
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size),5)
#cv2.imshow('blurred detection', blur_gray)
#cv2.waitKey(0)

low_threshold = 100
high_threshold = 200
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(test_image) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
vertical = []
horizontal = []
for line in lines:
    x1,y1,x2,y2 = line[0]
    if x1==x2: vertical.append(line[0].tolist())
    if y1==y2: horizontal.append(line[0].tolist())

vertical = sorted(vertical, key=lambda x: x[0])
vertical.insert(0, (0, 0, 0, line_image.shape[0]))
vertical.append((line_image.shape[1], 0, 
                 line_image.shape[1], line_image.shape[0]))

horizontal = sorted(horizontal, key=lambda x: x[1])
horizontal.insert(0, (0, 0, line_image.shape[1], 0))
horizontal.append((0, line_image.shape[0], 
                   line_image.shape[1], line_image.shape[0]))

columns = {i: (vertical[i][0], vertical[i+1][0]) for i in range(len(vertical)-1)}
rows = {i: (horizontal[i][1], horizontal[i+1][1]) for i in range(len(horizontal)-1)}

cells = np.empty((len(rows), len(columns)), dtype=object)

# Lines are doubling up
if False:
    for line in vertical:
        x1,y1,x2,y2 = line
        cv2.line(test_image,(x1,y1),(x2,y2),(255,0,0),5)
        cv2.imshow('marked detection', test_image)
        cv2.waitKey(0)

## Now we go through and mark cells with unfilled and filled circles
def get_cell_index(x, y, columns, rows):
    ri, ci = None, None
    for k, (l, r) in columns.items():
        if x<=r and x>=l:
            ci = k
            break
    for k, (t, b) in rows.items():
        if y>=t and y<=b:
            ri = k
            break
    return (ri, ci)

gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
(thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


circles = np.uint16(np.around(circles))

left_most = 999999
top_most = 99999


for c in circles[0]:
    # Check if filled
    c_x, c_y, c_r = c
    left_most = min(left_most, c_x-c_r)
    top_most = min(top_most, c_y-c_r)
    c_total = 0
    c_marked = 0

    for x in range(int(c_x-0.5*c_r), int(c_x+0.5*c_r)):
        for y in range(int(c_y-0.5*c_r), int(c_y+0.5*c_r)):
            if (c_x-x)**2 + (c_y-y)**2 <= c_r**2:
                c_total+=1
                if gray[y, x]<=150:
                    c_marked+=1
        
    ri, ci = get_cell_index(x,y, columns=columns, rows=rows)
    if c_marked/c_total>0.55:
        #print(ri, ci, x, y, c_marked/c_total)
        cells[ri][ci] = 1
        cv2.circle(gray, (c_x, c_y), c_r, (255, 0, 0), 2)
    else:
        cells[ri][ci] = 0

#print(cells.shape)
#print(cells[39])

#cv2.imshow('marked detection', gray)
#cv2.waitKey(0)

# For each circle, it's able to detect where the majority of the circle is
# filled in and mark which cell it belongs to.

# Next we need to find with row names belong to and column dates belong in

# Using pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# crop_image to left and top sections
names = gray[:, :left_most]
dates = gray[:top_most, :]

#cv2.imshow('names cropped', names)
#cv2.imshow('dates cropped', dates)
#cv2.waitKey(0)

name_data = pytesseract.image_to_data(names, output_type='dict')
date_data = pytesseract.image_to_data(dates, output_type='dict')

# Names
for i in range(len(name_data['level'])):
    text = str(re.sub('[^a-zA-Z\/\\ ]', '', name_data['text'][i]).strip())
    if text!='':
        x = name_data['left'][i] + name_data['width'][i]/2 # midpoint horizon of word
        y = name_data['top'][i] +name_data['height'][i]/2 # midpoint vert of word

        ri, ci = get_cell_index(x,y, columns=columns, rows=rows)
        #print(ri, ci, text)
        if cells[ri][ci] is None: cells[ri][ci]=''
        cells[ri][ci] = ' '.join((str(cells[ri][ci]), text))

        #Draw box   
        (x, y, w, h) = (name_data['left'][i], name_data['top'][i], name_data['width'][i], name_data['height'][i])
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Print Dates
for i in range(len(date_data['level'])):
    text = str(re.sub('[^a-zA-Z0-9\/\\ ]', '', date_data['text'][i]).strip())
    if text!='':
        #print(text)
        x = date_data['left'][i] + date_data['width'][i]/2 # midpoint horizon of word
        y = date_data['top'][i] +date_data['height'][i]/2 # midpoint vert of word

        ri, ci = get_cell_index(x,y, columns=columns, rows=rows)

        if ci is None or ri is None: continue
        if cells[ri][ci] is None: cells[ri][ci]=''
        cells[ri][ci] = ' '.join((cells[ri][ci], text))

        #Draw box   
        (x, y, w, h) = (date_data['left'][i], date_data['top'][i], date_data['width'][i], date_data['height'][i])
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
#cv2.imshow('labeled', gray)
#cv2.waitKey(0)

df = pd.DataFrame(cells)

df = df.dropna(how='all', axis=1)
df = df.dropna(how='all', axis=0)

df = df.reset_index(drop=True)
df.columns = [i for i in range(df.shape[1])]
#df = df.set_index(0)
columns = df.iloc[0, :]
columns[0] = 'Name'
df.columns = columns
df = df.iloc[1:, :]

df = df[df['Name']!='None']
df = df[df['Name'].notnull()]
print(df)
df = df.melt('Name', df.columns, 'Date', 'In Class')
df['In Class'].replace(0, None, inplace=True)
try:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].apply(lambda x: x.year)
    df['Month'] = df['Date'].apply(lambda x: x.month)
    df['Day'] = df['Date'].apply(lambda x: x.day)
except:
    df['Year'] = None
    df['Month'] = None
    df['Day'] = None

df = df[['Name','Date','Year','Month','Day','In Class']]
## Figure out how this should be exported
df.to_excel('Process.xlsx')

