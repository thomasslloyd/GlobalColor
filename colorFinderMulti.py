# AUTHOR: THOMAS LLOYD
# FILE: colorFinderMulti

import cv2
import glob
import numpy as np
# import matplotlib.pyplot as plt
import time
import statistics
import exifread
from pathlib import Path

# INSTANTIATION
refPt1 = np.array([0, 0])
refPt2 = np.array([0, 0])
refPt3 = np.array([0, 0])
refpt4 = np.array([0, 0])

# redefine a new set of reference points for pre selected roimouse
refPtMS = np.array([10, 10])
refPtMF = np.array([20, 20])


# FUNCTIONS

# ---------------------------------------------------------------------


def red(s):
    return '\033[1;31m%s\033[m' % s
# ---------------------------------------------------------------------


def green(s):
    return '\033[1;32m%s\033[m' % s
# ---------------------------------------------------------------------


def log(*m):
    print(" ".join(map(str, m)))
# ---------------------------------------------------------------------


def check_images(img1, img2, img3, img1name, img2name, img3name):
    if check_array(img1, img2, img3, img1name, img2name, img3name):
        return
    log(green("PASS"), img1name, img2name, img3name, "")
# ---------------------------------------------------------------------


def check_array(img1, img2, img3, img1name, img2name, img3name):
    try:
        res1 = img1*2
        res2 = img2*2
        res3 = img1*2
    except NotImplementedError:
        log(red("FAIL"), img1name, img2name, img3name, ": not implemented")
        return True
    if not type(res1, res2, res3) == np.ndarray:
        log(red("FAIL"), img1name, img2name, img3name, ": does not return a numpy array, type: ", type(img1))
        return True

    # if not len(res) == len(exp_res):
        # log(red("FAIL"), ex_name, ": expected an array of shape ", exp_res.shape, " but got array of shape", res.shape)
        # return True
    # if not equals(res, exp_res):
        # log(red("FAIL"), ex_name, ": incorrect answer. Expected", exp_res, ", got: ", res)
        # return True
    return


# ---------------------------------------------------------------------
def calculateROI(images, numimages):
    # create a matrix of ROI (region of image) MATS
    rois = []
    print("Creating image region mats...")
    print('\n')
    for n in range(0, numimages):
        image = images[n]
        thisroi = image[refPtMS[0]: refPtMF[0], refPtMS[1]: refPtMF[1]]
        rois.append(thisroi)

    print("ROI 1 (exp): ", rois[numimages-1])
    print('\n')
    print("ROI size", rois[numimages-1].shape)
    print('\n')
    print("portait test ROI: ", rois[0])
    print('\n')

    # --------------------------------------------------------------------------
    # quick test segment
    roitest = images[1]
    roitest = roitest[0:300, 0:300]
    roitest = rois[1]

    # --------------------------------------------------------------------------

    # incorportate empty case protection!!
    blues = []
    reds = []
    greens = []

    print("Creating BGR arrays")
    print('\n')
    for n in range(0, numimages):
        # b, g, r average of each image
        blues.append(rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 0].mean()) # adds single average value to blues array
        greens.append(rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 1].mean())
        reds.append(rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 2].mean())

        # roiblue = rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 0]
        # roigreen = rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 1]
        # roired = rois[n][0:refPtMF[0]-refPtMS[0], 0:refPtMF[1]-refPtMS[1], 2]
        #
        # blues.append(statistics.mode(roiblue))  # adds single average value to blues array
        # greens.append(statistics.mode(roigreen))
        # reds.append(statistics.mode(roired))

    # MAYBE ROOM TO ADD ANOTHER SAMPLE TYPE OTHER THAN MEAN

    blues = np.array(blues)  # np conversion after using append for effeciency
    greens = np.array(greens)
    reds = np.array(reds)
    print("blue shape: ", blues.shape)
    print('\n')

    print('Creating imagewise bgr ave storage...')
    print('\n')
    # bgr avrerages of each image, stored in bgraves
    bgraves = np.empty([3, numimages])
    bgraves[0, :] = blues
    bgraves[1, :] = greens
    bgraves[2, :] = reds
    print("BGR aves sample: ", bgraves[:, 0])
    print('\n')

    print('Overlaying individual mean color rectangles on top of images...')
    print('\n')
    for n in range(0, numimages):
        cv2.rectangle(images[n], (refPtMS[0], refPtMS[1]), (refPtMF[0], refPtMF[1]), bgraves[:, n], -1)

    # MOVES FROM IMAGEWISE TO GLOBAL BGR ANALYSIS
    blueave = np.sum(blues)/len(blues)
    greenave = np.sum(greens)/len(greens)
    redave = np.sum(reds)/len(reds)

    print('Creating global average array...')
    print('\n')
    bgrave = np.array([blueave, greenave, redave])
    print("global bgr ave: ", bgrave)
    print('\n')
    print("bgr aves deets: ", (bgrave.shape))
    print('\n')

    # division to pre-empt the image resizing
    canvaswidth = int((images[0].shape[1])/4)
    canvasheight = int((images[0].shape[0])/4)

    # Create a black imagen (open CV MAT)
    print('Creating black ave canvas...')
    print('\n')
    meancanvas = np.zeros([canvasheight, canvaswidth, 3], np.uint8)
    print("Elements of mean canvas array before: ", meancanvas[:, :])
    print("\n")
    meancanvas[:, :] = bgrave
    print("Elements of mean canvas array after: ", meancanvas[:, :])
    print("\n")

    # now create a matrix to simulate an image1
    print('Creating entire image of the mean color...')
    print('\n')
    cv2.rectangle(meancanvas, (0, canvaswidth), (0, canvasheight), (bgrave), -1)
    print('Mean canvas shape: ', meancanvas.shape)
    print('\n')

    # --------------------------------------------------------------------------
    return (bgrave, bgraves, meancanvas, roitest)

def flickrImport():
    flickrimages = []
    return flickrimages

def importAndLabelImages(folder):
    # global images, dims, numimages, namearray
    print('\n')
    # MAC
    # path = "/Users/thomaslloyd/Desktop/colorFinderMultiImages/" + folder + "/*.jpg"
    # WSL
    path = "/mnt/c/Users/Thomas/Dropbox/Media/00New/*.jpg""
    images = np.array([cv2.imread(file) for file in glob.glob(path)])
    dims = images[0].shape
    # ^^ in the future make this more robust by doing an array search for the smallest image
    # for n in range(0, images.shape[0]):
    #     height = images[n].shape[0]
    #     width = images[n].shape[1]
    #     if (width/height < 1):
    #         images[n] = np.rot90(images[n], k=1, axes=(0, 1))

    print("dimension of imag set: ", dims)
    print("dimension of images storage array: ", images.shape)
    print("Import Done")
    print('\n')

    # image names
    print('generating image names...')
    print('\n')

    numimages = images.shape[0]
    # namearray = np.chararray((numimages, 4))
    # namearray = []*[numimages, 4]

    # namearray = np.empty([numimages, 4])

    # namearray = np.empty([numimages])
    # namearray = np.empty([numimages], dtype=str)
    namearray = []
    print(namearray)
    # p = Path("/Users/thomaslloyd/Desktop/colorFinderMultiImages/" + folder)
    # list(p.glob('**/*.jpg'))
    # ^^ for when labelling becomes important

    # place exif in name arrays
    print("name array: ")
    print('\n')
    for n in range(0, numimages):
        ## creates and extract from exif dictspyth
        # f = open(list[n], '-')
        # exif = exifread.process_file(f)
        # ^^ for when labelling becomes important

        namearray.append("img" + str(n))

        # namearray[n, 1] = exif['Image Make']
        # namearray[n, 2] = exif['Image Resolution']
        # namearray[n, 3] = exif['Image Datetime']
        # ^^ for when labelling becomes important

    print(namearray)
    print('\n')
    print("Naming Done")
    print('\n')

    return (images, dims, numimages, namearray)
# ---------------------------------------------------------------------


def resizeImages(dims, images, meancanvas, numimages):
    newwidth = int((dims[0]/4))
    newheight = int((dims[1]/4))
    print("Resizing Images...")
    print("\n")
    imagesResized = []
    for n in range(0, numimages):
        imagesResized.append(cv2.resize(images[n], None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA))
    meancanvas = cv2.resize(meancanvas, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)

    for n in range(0, images.shape[0]):
        height = images[n].shape[0]
        width = images[n].shape[1]
        if (width/height < 1):
            imagesResized[n] = np.rot90(imagesResized[n], k=1, axes=(0, 1))
    imagesResized = np.array(imagesResized)

    for n in range(0, images.shape[0]):
        print("Resized image dims: ", imagesResized[n].shape)
    print("Resized meancanvas dims: ", meancanvas.shape)
    print("\n")

    print("Displaying images...")
    print("\n")
    return (newwidth, newheight, imagesResized, meancanvas)
# ---------------------------------------------------------------------


def createTile(imagesResized, meancanvas):
    # tileaspectratio = 3/4  # just for reference at this stage
    border = 20  # amount of space left in between each image and anything around it
    numobjects = len(imagesResized) + 1  # is for the meancanvas

    # create np array of shapes
    objectdimslist = np.zeros([imagesResized.shape[0], 3], np.uint32)
    for n in range(0, imagesResized.shape[0]):
        objectdimslist[n] = imagesResized[n].shape

    print("Printing dims of objects to be tiled: ")
    print('\n')
    for n in range(0, objectdimslist.shape[0]):
        print("Dims: ", objectdimslist[n])
    print("Dims: ", meancanvas.shape)
    print("num_objects: ", numobjects)
    print('\n')

    print("Determining the required tiled canvas size...")
    print('\n')

    largest = np.amax(objectdimslist, axis=0)
    print("Largest image dims: ", largest)
    print('\n')

    # possibledimslist = [[3, 4], [6, 8], [9, 16], ]
    # can make more algorithmic

    # 4
    if(4 <= numobjects <= 12):  # look to replace this statement with something more versatile
        tilewidth = 4*(largest[1]) + 5*border
        topedge = 4
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 3*(largest[0] + 4*border)
        sideedge = 3
    # 8
    elif(12 < numobjects <= 48):
        tilewidth = 8*(largest[1]) + 9*border
        topedge = 8
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 6 * (largest[0]) + 7*border
        sideedge = 6
    # 32
    elif(48 < numobjects <= 192):
        tilewidth = 16*(largest[1]) + 17*border
        topedge = 16
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 12 * (largest[0] + 10*border)
        sideedge = 12
    # 64
    elif(192 < numobjects <= 768):
        tilewidth = 32*(largest[1]) + 33*border
        topedge = 32
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 24 * (largest[0] + 25*border)
        sideedge = 24

    print("topedge: ", topedge)
    print("sideedge: ", sideedge)

    print("Creating the blank, black (brg zeros) canvas...")
    print('\n')
    # tilecanvas = np.zeros([tileheight, tilewidth, 3], np.uint8)
    tilecanvas = np.full([tileheight, tilewidth, 3], 255, np.uint8)
    print("Tile canvas dims: ", tilecanvas.shape)
    print('\n')

    # initial image vertex points
    oldstartx = border
    oldstarty = border
    oldfinishx = border + imagesResized[0].shape[1]
    oldfinishy = border + imagesResized[0].shape[0]

# ------------------------------------------------------------------------------
    print("Entering loop that lays the images on the canvas...")
    print('\n')
    tilecount = 1
    rowprog = 0.0  # progress along the snaking row
    for n in range(0, numobjects):
        if(n > 0):
            changex = largest[1]+border  # next image shift amount
            changey = 0
        elif(n == 0):
            # for the first image to be pasted where, no change
            changex = 0
            changey = 0

        # when the count gets to 4 it switches down a line
        # MAKE THIS UNIVERSAL
        # if(tilecount == 5 or tilecount == 9):

        # technically this should be count +1, however we want to do the y
        # shift on the 5th not 4th objext
        if (n > 0):
            rowprog = float(n/topedge)

        elif(n == 0.0):
            rowprog = float(0.0)

        print("row progress: ", rowprog)
        print('\n')

        # if(rowprog == 1 or rowprog == 2 or rowprog == 3 or rowprog == 4 or rowprog == 5 or rowprog == 6):
        if(rowprog.is_integer()):
            changex = (-1*oldstartx) + border
            changey = largest[0] + border

        print("IMG ", n+1)
        print("Change x: ", changex)
        print("Change y: ", changey)

        thisimagestartx = oldstartx + changex
        thisimagestarty = oldstarty + changey

        thisimagefinishx = oldfinishx + changex
        thisimagefinishy = oldfinishy + changey

        print('thisimagestartx: ', thisimagestartx)
        print('thisimagestarty: ', thisimagestarty)
        print('thisimagefinishx: ', thisimagefinishx)
        print('thisimagefinishy: ', thisimagefinishy)
        print('\n')

        # print("Title canvas sample: ", tilecanvas[0:10, 0:8])
        # print('\n')

        if (tilecount < numobjects):
            tilecanvas[thisimagestarty:(thisimagestarty + imagesResized[n].shape[0]),
                       thisimagestartx:(thisimagestartx + imagesResized[n].shape[1])] = imagesResized[n]
            # will this achieve element wise appending?

        if (tilecount == numobjects):
            tilecanvas[thisimagestarty:(thisimagestarty + meancanvas.shape[0]),
                       thisimagestartx:(thisimagestartx + meancanvas.shape[1])] = meancanvas
            # for when place the mean canvas at the end

        # the new x,y start and finish points are now the old
        oldstartx = thisimagestartx
        oldstarty = thisimagestarty
        oldfinishx = thisimagefinishx
        oldfinishy = thisimagefinishy

        tilecount += 1
# ------------------------------------------------------------------------------
    return tilecanvas


def testImages(images, numimages):
    print("Importing test set from image 1...")
    print('\n')
    imgtest = images[0]
    print("test check: ", imgtest.shape)
    print('\n')
    print("test image1: ", imgtest)
    print('\n')
    print("test image2: ", images[numimages-9])
    print('\n')
# ---------------------------------------------------------------------


def displayImages(numimages, namearray, imagesResized, meancanvas, roitest,
                  tilecanvas, folder, start_time):
    while(1):

        for n in range(0, numimages):
            cv2.imshow(namearray[n], imagesResized[n])

        cv2.imshow('tot', meancanvas)
        cv2.imshow('roitest', roitest)
        # quick resize for screen
        tilecanvas = cv2.resize(tilecanvas, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        cv2.imshow('global tile', tilecanvas)
        finish_time = time.time() - start_time
        print(folder, "runtime: ", finish_time)
        cv2.waitKey(0)
        break

    cv2.destroyAllWindows()
    finish_time = time.time() - start_time
    print(finish_time)


# CITY RUNNERS
def toulouse():
    # Start time
    start_time = time.time()
    folder = "toulouse"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    print("Toulouse BGR ave: ", bgrave)
    displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)



def amsterdam():
    # Start time
    start_time = time.time()

    folder = "amsterdam"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    print("Amsterdam BGR ave: ", bgrave)
    displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)



def newyork():
    # Start time
    start_time = time.time()

    folder = "newyork"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("New York BGR ave: ", bgrave)



def randomimagefiles():
    # Start time
    start_time = time.time()

    folder = "randomimagefiles"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("randomimagefiles BGR ave: ", bgrave)


# MAIN
# ---------------------------------------------------------------------
# read image from file
# dims = 436 × 1026


print("---BEGINNING----")
randomimagefiles()
# amsterdam()
# newyork()

# images, dims, numimages, namearray = importAndLabelImages()
# bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
# newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas)
# tilecanvas = createTile(imagesResized, meancanvas)
# displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas

# while(1):
#
#     for n in range(0, numimages):
#         cv2.imshow(namearray[n], imagesResized[n])
#
#     cv2.imshow('tot', meancanvas)
#     cv2.imshow('roitest', roitest)
#     # quick resize for screen
#     tilecanvas = cv2.resize(tilecanvas, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)
#     cv2.imshow('global tile', tilecanvas)
#     finish_time = time.time() - start_time
#     print(finish_time)
#     cv2.waitKey(0)
#     break
#
# cv2.destroyAllWindows()
# finish_time = time.time() - start_time
# print(finish_time)
