# AUTHOR: THOMAS LLOYD
# FILE: colorFinderMulti
# data resources: National Technical University of Athens © 2008-2012

import cv2
import glob
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import sys
from scipy import stats
# import os
# import exifread
# from pathlib import Path

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


# -----------------------------------------------------------------------------
def calculateROI(images, numimages, ROI_BOOLEAN):
    # create a matrix of ROI (region of image) MATS
    rois = []
    print("Creating image region mats...")
    print('\n')
    for n in range(0, numimages):
        image = images[n]
        if (ROI_BOOLEAN is True):
            thisroi = image[refPtMS[0]: refPtMF[0], refPtMS[1]: refPtMF[1]]
            rois.append(thisroi)
        else:
            refPtMS[0] = 0
            refPtMS[1] = 0

            refPtMF[0] = image.shape[0]
            refPtMF[1] = image.shape[1]
            thisroi = image[refPtMS[0]: refPtMF[0], refPtMS[1]: refPtMF[1]]
            rois.append(thisroi)

    print("ROI 1 (example): ", rois[numimages-1])
    print('\n')
    print("ROI siz (example: )", refPtMS, ",", refPtMF)
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
    print("Number of images featured in BGR aves: ", bgraves.shape[0])
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


def import_and_label_images(folder):
    # global images, dims, numimages, namearray
    print('\n')
    # MAC --------
    # path = "/Users/thomaslloyd/Desktop/colorFinderMultiImages/" + folder + "/*.jpeg"

    # MAC HD --------
    # path = "/Volumes/2018_SSD_TL/GlobalColorImages/" + folder +"/"+ folder +"_flickr" + "/*.jpg"
    folder_path = "/Volumes/2018_SSD_TL/GlobalColorImages/" + folder +"/*"

    # creating a list of the folder paths for this city
    folder_list = glob.glob(folder_path)  # creates a list of folders available for this citywise
    print("folder_list: ", folder_list)
    print("\n")

    # use folder list to unpack contained images
    image_paths = []
    for folder in folder_list:
        image_paths = image_paths + glob.glob(folder + "/*.jpg")

    # WSL --------
    # path = "/mnt/f/" + folder + "/*.jpg"
    # images = np.array([cv2.imread(file) for file in glob.glob(path)])

    images = np.array([cv2.imread(file) for file in image_paths])

    dims = images[0].shape
    print("dimension of imag set: ", dims)
    print('\n')
    print("Import Done")
    print('\n')

    # image names
    print('generating image names...')
    print('\n')

    numimages = images.shape[0]
    namearray = []

    # p = Path("/Users/thomaslloyd/Desktop/colorFinderMultiImages/" + folder)
    # list(p.glob('**/*.jpg'))
    # ^^ for when labelling becomes important

    # place exif in name arrays
    print("name array: ")
    print('\n')
    for n in range(0, numimages):
        # creates and extract from exif dictspyth
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

    # for n in range(0, images.shape[0]):
    #     print("Resized image dims: ", imagesResized[n].shape)
    # print("Resized meancanvas dims: ", meancanvas.shape)
    # print("\n")

    print("Displaying images...")
    print("\n")
    return (newwidth, newheight, imagesResized, meancanvas)
# ------------------------------------------------------------------------------


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
    # 16
    elif(48 < numobjects <= 192):
        tilewidth = 16*(largest[1]) + 17*border
        topedge = 16
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 12 * (largest[0] + 13*border)
        sideedge = 12
    # 32
    elif(192 < numobjects <= 768):
        tilewidth = 32*(largest[1]) + 33*border
        topedge = 32
        # width of overall tile = width of all images and buffer thicknesses
        tileheight = 24 * (largest[0] + 25*border)
        sideedge = 24
    # 64
    elif(768 < numobjects <= 3072):
        topedge = 64
        sideedge = int((topedge/4)*3)  # (48)
        tilewidth = topedge * (largest[1] + (topedge+1)*border)
        tileheight = sideedge * (largest[0] + (sideedge+1)*border)

    print("topedge: ", type(topedge))

    print("sideedge: ", type(sideedge))

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
    return tilecanvas

# ------------------------------------------------------------------------------


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
# ------------------------------------------------------------------------------


def displayImages(numimages, namearray, imagesResized, meancanvas, roitest,
                  tilecanvas, folder, start_time):
    while(1):

        # for n in range(0, numimages):
            # displays individual images
            # cv2.namedWindow(namearray[n], cv2.WINDOW_NORMAL)
            # cv2.moveWindow(namearray[n], 300, 300)
            # cv2.imshow(namearray[n], imagesResized[n])

        cv2.imshow('tot', meancanvas)
        cv2.imshow('roitest', roitest)
        # quick resize for screen
        # tilecanvas = cv2.resize(tilecanvas, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        width, height = 1280, 800
        cv2.namedWindow('global tile', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('global tile', width, height)
        cv2.moveWindow('global tile', 20, 20)
        cv2.imshow('global tile', tilecanvas)
        finish_time = time.time() - start_time
        print(folder, "runtime: ", finish_time)
        cv2.waitKey(0)  # could add if statement here to check which city is
        # currently being run, then if its the last city, activate the waitKey
        # alternatively take the imshows out of the local loops
        break

    cv2.destroyAllWindows()
    finish_time = time.time() - start_time
    print(finish_time)
# ------------------------------------------------------------------------------


def display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest
                       , folder, start_time):

    print("BEGINNING MPL display procedure...")
    print("\n")

    print("setting width and height of plot")
    print("\n")

    print("number of images being plotted: ", numimages)
    subplotwidth = int(numimages**.5)
    print("\n")

    # protecting size of plot due to rounding
    roundingdiff = subplotwidth - (numimages**.5)
    if (roundingdiff < 0):
        subplotwidth = subplotwidth + 1
    subplotheight = subplotwidth

    print("subplotwidth and height: ", subplotwidth)
    print("\n")

    # example code:

# data = np.random.randn(2, 100)
# fig, axs = plt.subplots(2, 2, )
# axs[0, 0].hist(data[0])
# axs[1, 0].scatter(data[0], data[1])
# axs[0, 1].plot(data[0], data[1])
# axs[1, 1].hist2d(data[0], data[1])
# plt.show()

    # subplot setup
    fig, axs = plt.subplots(subplotwidth, subplotheight)

    columnPlot = 0
    rowPlot = 0
    for n in range(0, numimages):
        # axs[n, m]
        # n = columns
        # m = rows
        axs[columnPlot, rowPlot].imshow(cv2.cvtColor(imagesResized[n], cv2.COLOR_BGR2RGB))
        axs[columnPlot, rowPlot].axis('off')
        # axs[columnPlot, rowPlot].set_title(namearray[n])

        # plot figure column iterator
        # first check if interation is complete

        if (columnPlot == (subplotwidth-1)):
            columnPlot = 0
            rowPlot = rowPlot + 1
            print("column plot: ", columnPlot)
            print("row plot: ", rowPlot)
            print("\n")
        else:
            columnPlot = columnPlot + 1
            print("column plot: ", columnPlot)
            print("row plot: ", rowPlot)
            print("\n")
    print("mpl iterator complete")
    print("\n")
    fig.suptitle(folder, fontsize=16)
    plt.show()

    # toshow = plt.imshow(cv2.cvtColor(imagesResized[n], cv2.COLOR_BGR2RGB))
    # plt.show(toShow)
# ------------------------------------------------------------------------------


def display_canvas_set_MPL(meancanvasset, namearray, canvasnamearray, bgraves, citywise, folder):

    # To be used for either international canvas plot or city wise canvas plot.
    # Hence for city wise, every image will have a mean canvas plotted.
    # And hence for international, the overall city mean will be plotted.

    if (citywise is True):
        meancanvasset = bgraves
        print("mean canvas set from bgraves: ", "\n", meancanvasset)
        print("\n")
        numimages = meancanvasset.shape[1]
        print("number of canvas' being displayed: ", numimages)
        print("\n")

    else:
        # print("mean canvas set prior to np conversion: ", "\n", meancanvasset)
        # print('\n')
        meancanvasset = np.array(meancanvasset)
        # print("mean canvas set after to np conversion: ", "\n", meancanvasset)
        # print('\n')
        numimages = meancanvasset.shape[0]
        print("number of canvas' being displayed: ", numimages)
        print("\n")

    print("Setting up matplotlib display....")
    print("\n")

    print("setting width and height of plot")
    print("\n")

    print("number of images being plotted: ", numimages)
    print("\n")
    subplotwidth = int(numimages**.5)

    # protecting size of plot due to rounding
    roundingdiff = subplotwidth - (numimages**.5)
    if (roundingdiff < 0):
        subplotwidth = subplotwidth + 1
    subplotheight = subplotwidth

    print("subplotwidth and height: ", subplotwidth)
    print("\n")

    # subplot setup
    fig, axs = plt.subplots(subplotwidth, subplotheight)
    # returns a 2D array of subplots ^^

    columnPlot = 0
    rowPlot = 0
    for n in range(0, numimages):
        # axs[n, m]
        # n = columns
        # m = rows
        if (citywise is True):
            # for the case when plotting intra city mean mats
            # thisimage = np.full((200, 200, 3), bgraves[:, n])
            thisimage = np.float32(np.full((200, 200, 3), bgraves[:, n]/255))
            # print("intra city BGR canvas: ", thisimage)
            # print("intra city BGR canvas size: ", thisimage.shape)
            # print("intra city BGR canvas dtype: ", type(thisimage))
            # print("intra city BGR element dtype: ", type(thisimage[0, 0, 0]))
            # axs[columnPlot, rowPlot].imshow(cv2.cvtColor(thisimage, cv2.COLOR_BGR2RGB))
            axs[columnPlot, rowPlot].imshow(cv2.cvtColor(thisimage, cv2.COLOR_BGR2RGB))
            axs[columnPlot, rowPlot].axis('off')
            # axs[columnPlot, rowPlot].set_title(namearray[n])
        else:
            # for the case when plotting city total (international) mean mats
            axs[columnPlot, rowPlot].imshow(cv2.cvtColor(meancanvasset[n], cv2.COLOR_BGR2RGB))
            axs[columnPlot, rowPlot].axis('off')
            # axs[columnPlot, rowPlot].set_title(canvasnamearray[n])
            # plot figure column iterator
            # first check if interation is complete

        if (columnPlot == (subplotwidth-1)):
            columnPlot = 0
            rowPlot = rowPlot + 1
            print("column plot: ", columnPlot)
            print("row plot: ", rowPlot)
            print("\n")
        else:
            columnPlot = columnPlot + 1
            print("column plot: ", columnPlot)
            print("row plot: ", rowPlot)
            print("\n")

    print("mpl iterator complete")
    print("\n")

    if (citywise is True):
        title = 'Mean ' + folder + ' Color Tiles'
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Mean tiles of all cities considered', fontsize=16)
    plt.show()
# ------------------------------------------------------------------------------


def color_space_plot(meancanvasset, namearray, canvasnamearray, bgraves, citywise):

    print("3D color space plot beginning...")
    print('\n')
    print("bgraves: ", bgraves)
    print("bgraves size: ", bgraves.shape)
    print("bgraves type: ", type(bgraves))
    print('\n')

    plotx = bgraves[0, :]
    ploty = bgraves[1, :]
    plotz = bgraves[2, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(plotx, ploty, plotz, marker='o')

    ax.set_xlabel('B')
    ax.set_ylabel('G')
    ax.set_zlabel('R')

    fig.suptitle('All means plotted on R G B', fontsize=16)
    plt.show()
    print('3D color space plot complete')
    print('\n')
# ------------------------------------------------------------------------------


def calcMode(images, numimages):

    modes = np.zeros([numimages, 3])

    for i in range(0, numimages):
        # current image to calc the mode of
        print("calculating the mode of image ", i, "...")
        print("\n")
        image = images[i]

        # temportary lists to store the bgr values
        blues = []
        greens = []
        reds = []

        # n rows and m columns, shape will be (n,m)

        for m in range(0, image.shape[0]-1):
            for n in range(0, image.shape[1]-1):
                blues.append(int(image[m, n, 0]))
                greens.append(int(image[m, n, 1]))
                reds.append(int(image[m, n, 2]))

        print("number of blue pixels: ", len(blues))
        print("number of green pixels: ", len(greens))
        print("number of red pixels: ", len(reds))
        print("\n")

        # array containing the mode of each image
        bluemode = stats.mode(blues)[0]
        greenmode = stats.mode(greens)[0]
        redmode = stats.mode(reds)[0]

        print("Bluemode: ", bluemode)
        print("Greenmode: ", greenmode)
        print("Redmode: ", redmode)
        print("\n")

        modes[i, 0] = bluemode
        modes[i, 1] = greenmode
        modes[i, 2] = redmode

    return modes
# ------------------------------------------------------------------------------


def mean_canvas_stacker(meancanvas, meancanvasset, folder, canvasnamearray):
    meancanvasset.append(meancanvas)
    canvasnamearray.append(folder)
    return meancanvasset, canvasnamearray
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# CITY RUNNERS
def runAllCities():

    citiesList = ['newyork', 'amsterdam', 'london', 'moscow', 'singapore', 'auckland', 'barcelona', 'toulouse', 'taipei', 'tokyo']
    # Start time
    start_time = time.time()
    meancanvasset = []
    canvasnamearray = []
    citywise = False
    ROI_BOOLEAN = True
    bgravesfordisp = np.zeros([len(citiesList), 3])

    n = 0
    for city in citiesList:
        folder = city
        try:
            images, dims, numimages, namearray = import_and_label_images(folder)
            bgrave, bgraves, meancanvas, roitest = calculateROI(images,
                                                                numimages,
                                                                ROI_BOOLEAN)
            # mode = calcMode(images, numimages)
            newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
            # tilecanvas = createTile(imagesResized, meancanvas)

            # specifically to append the meancanvasset with city specific mat
            meancanvasset, canvasnamearray = mean_canvas_stacker(meancanvas, meancanvasset, folder, canvasnamearray)
            bgravesfordisp[n, :] = bgrave
            print(city, " BGR ave: ", bgrave)
            print("\n")
            # print(city, " BGR mode: ", mode)
            display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                               folder, start_time)
            # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
        except IndexError:
            print("Oops!", sys.exc_info()[0], "occured for:", folder,
                  '- image database is likely empty for this city.')
            print('\n')
            print("Analyzing the next city...")
            print('\n')
        n = n+1

    print('\n')
    print('All BGR city means: ', '\n', bgravesfordisp)
    print('\n')

    # displaying all mean canvas' using matplotlib
    try:
        display_canvas_set_MPL(meancanvasset, namearray, canvasnamearray, bgraves, citywise, folder)
    except IndexError:
        print("something went wrong while displaying the canvas set")

    # displaying all mean canvas' using matplotlib
    try:
        color_space_plot(meancanvasset, namearray, canvasnamearray, bgraves, citywise)
    except IndexError:
        print("something went wrong while running the color space plot")


def test():
    # Start time
    meancanvasset = []
    canvasnamearray = []
    citywise = True  # to denote the nature of the mean canvas plot (intracity here)
    ROI_BOOLEAN = True
    start_time = time.time()
    folder = "toulouse"
    images, dims, numimages, namearray = import_and_label_images(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages, ROI_BOOLEAN)
    # bgrmode = calcMode(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    # tilecanvas = createTile(imagesResized, meancanvas)
    print("Toulouse BGR ave: ", bgrave)
    print("\n")
    # print("Toulouse BGR ave: ", bgrmode)
    meancanvasset, canvasnamearray = mean_canvas_stacker(meancanvas, meancanvasset, folder, canvasnamearray)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    display_canvas_set_MPL(meancanvasset, namearray, canvasnamearray, bgraves, citywise, folder)
    color_space_plot(meancanvasset, namearray, canvasnamearray, bgraves, citywise)
# ------------------------------------------------------------------------------


def newyork():
    # Start time
    start_time = time.time()

    folder = "newyork"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    # mode = calcMode(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("New York BGR ave: ", bgrave)
    # print("New York BGR mode: ", mode)
# ------------------------------------------------------------------------------


def amsterdam():
    # Start time
    start_time = time.time()

    folder = "amsterdam"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("Amsterdam BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def london():
    # Start time
    start_time = time.time()
    folder = "london"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)

    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("London BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def moscow():
    # Start time
    start_time = time.time()
    folder = "moscow"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("Moscow BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def singapore():
    # Start time
    start_time = time.time()
    folder = "taipei"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("Taipei BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def auckland():
    # Start time
    start_time = time.time()
    folder = "auckland"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("Auckland BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def barcelona():
    # Start time
    start_time = time.time()
    folder = "barcelona"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("barcelona BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def toulouse():
    # Start time
    start_time = time.time()
    folder = "toulouse"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("Toulouse BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def taipei():
    # Start time
    start_time = time.time()
    folder = "toulouse"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    print("Taipei BGR ave: ", bgrave)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
# ------------------------------------------------------------------------------


def tokyo():
    # Start time
    start_time = time.time()
    folder = "tokyo"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    tilecanvas = createTile(imagesResized, meancanvas)
    display_Images_MPL(numimages, namearray, imagesResized, meancanvas, roitest,
                       tilecanvas, folder, start_time)
    print("Tokyo BGR ave: ", bgrave)
    # displayImages(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
# ------------------------------------------------------------------------------


def randomimagefiles():
    # Start time
    start_time = time.time()
    folder = "ec1m_landmark_images"
    images, dims, numimages, namearray = importAndLabelImages(folder)
    bgrave, bgraves, meancanvas, roitest = calculateROI(images, numimages)
    newwidth, newheight, imagesResized, meancanvas = resizeImages(dims, images, meancanvas, numimages)
    # tilecanvas = createTile(imagesResized, meancanvas)
    # displayImagesMPL(numimages, namearray, imagesResized, meancanvas, roitest, tilecanvas, folder, start_time)
    print("randomimagefiles BGR ave: ", bgrave)
# ------------------------------------------------------------------------------


def sizechecker(images):
    if images.size == 0:
        return False
    return True


# MAIN
# ---------------------------------------------------------------------
# read image from file
# dims = 436 × 1026
print('\n')
print("---BEGINNING---")

test()

# amsterdam()
# auckland()
# barcelona()
# london()
# newyork()
# toulouse()
# taipei()
# tokyo()
# amsterdam()
# newyork()
# runAllCities()

print("---COLOR FINDER MULTI COMPLTETE---")
print("\n")
print("\n")
