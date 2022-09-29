import os
import cv2 as cv2
import numpy as np
import argparse
from random import randrange
from threeD_object import *
import math

#-----------------------Generating ArUco markers in OpenCV-----------------------
def generateArucoMarkers(inputMedia):


    path = 'Markers'
    markers = os.listdir(path)
    print(markers)
    assignedIndex = 1 if len(markers) == 0 else int(markers[-1].split('.')[0]) + 1
    print("assignedIndex: ", assignedIndex)

    # Load the predefined dictionary
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    markerImage1 = np.zeros((200, 200), dtype=np.uint8)
    markerImage2 = np.zeros((200, 200), dtype=np.uint8)
    markerImage3 = np.zeros((200, 200), dtype=np.uint8)
    markerImage4 = np.zeros((200, 200), dtype=np.uint8)

    markerImage1 = cv2.aruco.drawMarker(dictionary, assignedIndex, 200, markerImage1, 1)
    markerImage2 = cv2.aruco.drawMarker(dictionary, assignedIndex+1, 200, markerImage2, 1)
    markerImage3 = cv2.aruco.drawMarker(dictionary, assignedIndex+2, 200, markerImage3, 1)
    markerImage4 = cv2.aruco.drawMarker(dictionary, assignedIndex+3, 200, markerImage4, 1)

    # Save Aruco Markers
    cv2.imwrite(f"Markers/{assignedIndex}.jpg", markerImage1)
    cv2.imwrite(f"Markers/{assignedIndex+1}.jpg", markerImage2)
    cv2.imwrite(f"Markers/{assignedIndex+2}.jpg", markerImage3)
    cv2.imwrite(f"Markers/{assignedIndex+3}.jpg", markerImage4)

    # Save input media
    extension = inputMedia.split('.')[1]

    print('extension: ', extension)
    if (extension == 'mp4'):
        vidCapture = cv2.VideoCapture(inputMedia)
        success, vid = vidCapture.read()
        width = int(vidCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vidCapture.get(cv2.CAP_PROP_FPS)
        print("width: ", width)
        print("height: ", height)
        print("fps: ", fps)
        frameCounter = 0
        videoWriter = cv2.VideoWriter(f"Gallery/{assignedIndex}.{extension}", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
        print("vidCapture.get(cv2.CAP_PROP_FRAME_COUNT): ", vidCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        while (vidCapture.isOpened() and frameCounter <= vidCapture.get(cv2.CAP_PROP_FRAME_COUNT)):

            ret, frame = vidCapture.read()
            videoWriter.write(frame)
            frameCounter += 1
            # print("frameCounter: ", frameCounter)

        vidCapture.release()
        videoWriter.release()


    elif (extension == 'jpg' or extension == 'png'):
        mediaFile = cv2.imread(inputMedia)
        cv2.imwrite(f"Gallery/{assignedIndex}.{extension}", mediaFile)



    return assignedIndex

#---------------------------------------------------------------------------------

#-----------------------------Detecting Aruco markers-----------------------------

def loadOverlayImages(path):
    myList = os.listdir(path)
    print("myList: ", myList)
    noOfMarkers = len(myList)
    print("Total number of markers detected: ", noOfMarkers)
    augDict = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f"{path}/{imgPath}")
        augDict[key] = imgAug
    return augDict

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(cv2.aruco, f"DICT_{markerSize}X{markerSize}_{totalMarkers}")
    arucoDict = cv2.aruco.Dictionary_get(key)
    arucoParam = cv2.aruco.DetectorParameters_create()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(markerIds)
    if draw:
        cv2.aruco.drawDetectedMarkers(img, markerCorners)
    return [markerCorners, markerIds]


def augmentAruco(bbox, id, img, overlayImg, drawId=True):
    tl = int(bbox[0][0][0]), int(bbox[0][0][1])
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]

    ht, wt, c = overlayImg.shape
    pt1 = np.array([tl, tr, br, bl])
    pt2 = np.float32([[0,0], [wt,0], [wt,ht], [0, ht]])
    # Find homography
    matrix, _ = cv2.findHomography(pt2, pt1)
    # Warp perspective
    imgOut = cv2.warpPerspective(overlayImg, matrix, (img.shape[1], img.shape[0]))
    print('imgOut: ', imgOut)
    cv2.fillConvexPoly(img, pt1.astype(int), (0, 0, 0))
    imgOut = img + imgOut

    # Draw id
    print("id: ", str(id))
    print("tl: ", tl[0])
    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 255), thickness=2)

    return imgOut

def augmentImage(primaryKey):
    assignedID = int(primaryKey)
    print("assignedID: ", assignedID)
    cap = cv2.VideoCapture(0)
    # Load overlay images from Markers folder
    augDict = loadOverlayImages("Markers")
    augImg = cv2.imread(f'Gallery/{assignedID}.jpg')


    while True:

        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        # print("foundArucos[0]: ", foundArucos[0])
        # print("foundArucos[1]: ", foundArucos[1])

        if len(foundArucos[0]) == 4:

            ArucosId = foundArucos[1].flatten()
            # print("ArucosId: ", ArucosId)
            bbox1_index = np.where(ArucosId == assignedID)[0][0]
            bbox2_index = np.where(ArucosId == assignedID+1)[0][0]
            bbox3_index = np.where(ArucosId == assignedID+2)[0][0]
            bbox4_index = np.where(ArucosId == assignedID+3)[0][0]

            bbox1 = foundArucos[0][bbox1_index]
            bbox2 = foundArucos[0][bbox2_index]
            bbox3 = foundArucos[0][bbox3_index]
            bbox4 = foundArucos[0][bbox4_index]

            tl = bbox1[0][0][0], bbox1[0][0][1]
            tr = bbox2[0][1][0], bbox2[0][1][1]
            br = bbox4[0][2][0], bbox4[0][2][1]
            bl = bbox3[0][3][0], bbox3[0][3][1]

            ht, wt, c = augImg.shape

            pt1 = np.array([tl, tr, br, bl])
            pt2 = np.float32([[0, 0], [wt, 0], [wt, ht], [0, ht]])
            # Find homography
            matrix, _ = cv2.findHomography(pt2, pt1)
            # Warp perspective
            imgOut = cv2.warpPerspective(augImg, matrix, (img.shape[1], img.shape[0]))
            cv2.fillConvexPoly(img, pt1.astype(int), (0, 0, 0))
            img = img + imgOut

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break

def augmentVideo(primaryKey):
    assignedIndex = int(primaryKey)
    cap = cv2.VideoCapture(0)
    # Load overlay images from Markers folder

    myVid = cv2.VideoCapture(f'Gallery/{assignedIndex}.mp4')
    success, augVid = myVid.read()
    print("augVid.shape: ", augVid.shape)
    detection = False
    frameCounter = 0

    while True:

        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        # print("foundArucos[0]: ", foundArucos[0])
        # print("foundArucos[1]: ", foundArucos[1])

        if detection == False:
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        else:
            if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):
                myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frameCounter = 0
            success, augVid = myVid.read()

        if len(foundArucos[0]) == 4:
            detection = True
            ArucosId = foundArucos[1].flatten()
            # print("ArucosId: ", ArucosId)
            bbox1_index = np.where(ArucosId == assignedIndex)[0][0]
            bbox2_index = np.where(ArucosId == assignedIndex+1)[0][0]
            bbox3_index = np.where(ArucosId == assignedIndex+2)[0][0]
            bbox4_index = np.where(ArucosId == assignedIndex+3)[0][0]

            bbox1 = foundArucos[0][bbox1_index]
            bbox2 = foundArucos[0][bbox2_index]
            bbox3 = foundArucos[0][bbox3_index]
            bbox4 = foundArucos[0][bbox4_index]

            tl = bbox1[0][0][0], bbox1[0][0][1]
            tr = bbox2[0][1][0], bbox2[0][1][1]
            br = bbox4[0][2][0], bbox4[0][2][1]
            bl = bbox3[0][3][0], bbox3[0][3][1]

            ht_vid, wt_vid, c_vid = augVid.shape

            pt1 = np.array([tl, tr, br, bl])
            pt2 = np.float32([[0, 0], [wt_vid, 0], [wt_vid, ht_vid], [0, ht_vid]])
            # Find homography
            matrix, _ = cv2.findHomography(pt2, pt1)
            # Warp perspective
            imgOut = cv2.warpPerspective(augVid, matrix, (img.shape[1], img.shape[0]))
            cv2.fillConvexPoly(img, pt1.astype(int), (0, 0, 0))
            img = img + imgOut

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break

        frameCounter += 1

def augmentArucoSquares():
    cap = cv2.VideoCapture(0)
    # Load overlay images from Markers folder
    augDict = loadOverlayImages("Gallery")

    while True:
        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        if len(foundArucos[0]) != 0:
            for bbox, id in zip(foundArucos[0], foundArucos[1]):
                # Check if the overlay image is available for specific aruco id
                if int(id) in augDict:
                    img = augmentAruco(bbox, id, img, augDict[int(id)])

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', type=str,
                        help='Option: Image, Video, Visualize, GenerateMarkers')

    parser.add_argument('mediaFile', type=str,
                        help='Option: img.jpg, vid.mp4, None')

    parser.add_argument('primaryKey', type=str,
                        help='Option: [0, 249], None')


    args = parser.parse_args()
    if args.operation == 'Image':
        if (args.primaryKey == 'none' or args.primaryKey == 'None' or args.primaryKey == ''):
            print('Please provide aruco ID!')
        else:
            augmentImage(args.primaryKey)

    elif args.operation == 'Video':
        if (args.primaryKey == 'none' or args.primaryKey == 'None' or args.primaryKey == ''):
            print('Please provide aruco ID!')
        else:
            augmentVideo(args.primaryKey)

    elif args.operation == 'Visualize':
        visualize()

    elif args.operation == 'GenerateMarkers':
        if (args.mediaFile != 'None'):
            primaryKey = generateArucoMarkers(args.mediaFile)
            print("Primary key: ", primaryKey)

def get_extended_RT(A, H):
    # finds r3 and appends
    # A is the intrinsic mat, and H is the homography estimated
    H = np.float64(H)  # for better precision
    A = np.float64(A)
    R_12_T = np.linalg.inv(A).dot(H)

    r1 = np.float64(R_12_T[:, 0])  # col1
    r2 = np.float64(R_12_T[:, 1])  # col2
    T = R_12_T[:, 2]  # translation

    # ideally |r1| and |r2| should be same
    # since there is always some error we take square_root(|r1||r2|) as the normalization factor
    norm = np.float64(math.sqrt(np.float64(np.linalg.norm(r1)) * np.float64(np.linalg.norm(r2))))

    r3 = np.cross(r1, r2) / (norm)
    R_T = np.zeros((3, 4))
    R_T[:, 0] = r1
    R_T[:, 1] = r2
    R_T[:, 2] = r3
    R_T[:, 3] = T
    return R_T

def visualize():
    obj = three_d_object('dog/low-poly-dog-by-pixelmannen.obj', 'dog/texture.png')
    print("obj: ", obj)
    vertices = obj.vertices
    print("vertices: ", vertices)
    print("len(vertices): ", len(vertices))
    print("len(obj.faces): ", len(obj.faces))

    A = [[1019.37187, 0, 618.709848], [0, 1024.2138, 327.280578], [0, 0, 1]]
    A = np.array(A)


    cap = cv2.VideoCapture(0)


    while True:
        success, img = cap.read()
        foundArucos = findArucoMarkers(img)
        # Loop through all markers and augment each one
        if len(foundArucos[0]) != 0:
            for bbox, id in zip(foundArucos[0], foundArucos[1]):
                # Check if the overlay image is available for specific aruco id

                tl = int(bbox[0][0][0]), int(bbox[0][0][1])
                tr = bbox[0][1][0], bbox[0][1][1]
                br = bbox[0][2][0], bbox[0][2][1]
                bl = bbox[0][3][0], bbox[0][3][1]

                ht, wt, c = img.shape
                pt1 = np.array([tl, tr, br, bl])
                pt2 = np.float32([[0, 0], [wt, 0], [wt, ht], [0, ht]])
                # Find homography
                matrix, _ = cv2.findHomography(pt2, pt1)
        
                R_T = get_extended_RT(A, matrix)
                transformation = A.dot(R_T)

                w1 = tr[0] - tl[0]
                w2 = br[0] - bl[0]
                h1 = bl[1] - tl[1]
                h2 = br[1] - tr[1]

                w = w1 if w1 < w2 else w2
                h = h1 if h1 < h2 else h2
                color = False
                # projecting the faces to pixel coords and then drawing
                for face in obj.faces:
                    # a face is a list [face_vertices, face_tex_coords, face_col]
                    face_vertices = face[0]
                    points = np.array(
                        [vertices[vertex - 1] for vertex in face_vertices])  # -1 because of the shifted numbering
                    points = 4 * points
                    points = np.array([[p[2] + w / 2, p[0] + h / 2, p[1]] for p in points])  # shifted to centre
                    dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), transformation)  # transforming to pixel coords
                    imgpts = np.int32(dst)
                    if color is False:
                        cv2.fillConvexPoly(img, imgpts, (50, 50, 50))
                    else:
                        cv2.fillConvexPoly(img, imgpts, face[-1])

        cv2.imshow("Frame", img)
        if cv2.waitKey(1) == ord('q'):
            break




main()

# visualize()

#---------------------------------------------------------------------------------


