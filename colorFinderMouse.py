import cv2
import numpy as np
# import sys
# sys.path.append("..")

# FUNCTIONS
# ---------------------------------------------------------------------
drawing = False  # true if mouse is pressed
mode = True  # if true, draw rectangle

refPt1 = np.array([0, 0])
refPt2 = np.array([0, 0])
refPt3 = np.array([0, 0])


def draw_rectangle(event, x, y, flags, param):

    global refPt1, refPt2, refPt3, rgb_ave, drawing, mode
    # if the left mouse button was clicked, record the starting
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        refPt1[0] = x
        refPt1[1] = y
        print('x = %d, y = %d press' % (x, y))

        # https://stackoverflow.com/questions/50234485/drawing-rectangle-in-opencv-python/50235566

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            refPt2[0] = x
            refPt2[1] = y
            roimouse = np.array(img1[refPt1[0]:refPt2[0], refPt1[1]:refPt2[1]])

            print("ROI mouse: ", roimouse)
            print("ROI size", roimouse.shape)

            # incorportate empty case protection!!
            red = roimouse[0:refPt2[0]-refPt1[0], 0:refPt2[1]-refPt1[1], 0].mean()
            blue = roimouse[0:refPt2[0]-refPt1[0], 0:refPt2[1]-refPt1[1], 1].mean()
            green = roimouse[0:refPt2[0]-refPt1[0], 0:refPt2[1]-refPt1[1], 2].mean()
            rgb_ave = np.array([red, blue, green])


            # incorporate reverse drag protection
            # if x1 > x2 then one way
            cv2.rectangle(img1, (refPt1[0], refPt1[1]), (x, y), (255, 255, 255), 3)
            if refPt2[0] != x | refPt2[1] != y:
                cv2.rectangle(img1, (refPt1[0], refPt1[1]), (refPt2[0], refPt2[1]), (rgb_ave), -1)
        # else:
            # cv2.circle(img1, (x, y), 5, (0, 0, 0), -1)

    if event == cv2.EVENT_LBUTTONUP:
        if mode is True:
            refPt3[0] = x
            refPt3[1] = y

            roimouse = np.array(img1[refPt1[0]:refPt2[0], refPt1[1]:refPt2[1]])

            print("ROI mouse: ", roimouse)
            print("ROI size", roimouse.shape)

            red = roimouse[0:refPt3[0]-refPt1[0], 0:refPt3[1]-refPt1[1], 0].mean()
            blue = roimouse[0:refPt3[0]-refPt1[0], 0:refPt3[1]-refPt1[1], 1].mean()
            green = roimouse[0:refPt3[0]-refPt1[0], 0:refPt3[1]-refPt1[1], 2].mean()
            rgb_ave = np.array([red, blue, green])

            print("Ave RGB: ", rgb_ave)
            cv2.rectangle(img1, (refPt1[0], refPt1[1]), (refPt3[0], refPt3[1]), (rgb_ave), -1)
        # else:
            # cv2.circle(img1, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Toulouse Brick', img1)
        drawing = False
        print('x = %d, y = %d release' % (x, y))

    # xstart = refPt1[0]
    # ystart = refPt1[1]
    #
    # xfinish = refPt2[0]
    # yfinish = refPt2[1]

# ---------------------------------------------------------------------

# MAIN
# ---------------------------------------------------------------------
# read image from file
# dims = 436 × 1026

img1 = cv2.imread(r"/Users/thomaslloyd/Desktop/toulousebrick.png",
                  cv2.IMREAD_COLOR)

# ^^ add specific file type for ideal analysis

img2 = np.array(cv2.imread(r"/Users/thomaslloyd/Desktop/toulousebrick.png",
                cv2.IMREAD_COLOR))


# parameterise image to ideal color space

rows, cols, channels = img2.shape
print("rows:", rows)
print("columns:", cols)
print("channels:", channels)
print(img1)

# create rectangle of color palette

# present image on screen with roi selected
cv2.namedWindow('Toulouse Brick', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('Toulouse Brick', draw_rectangle)
# cv2.moveWindow('Toulouse Brick', 300, -300)

# draw rectangle around pixels desired
# cv2.rectangle(img1, (x_start+2, y_start+2), (x_finish-2, y_finish-2), (rgb_ave), -1)
# draw rectangle around pixels desired
# cv2.rectangle(img1, (x_start, y_start), (x_finish, y_finish), (255, 0, 0), 2)

while(1):
    cv2.imshow('Toulouse Brick', img1)
    cv2.waitKey(0)
    break

cv2.destroyAllWindows()
