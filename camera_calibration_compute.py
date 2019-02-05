#!/usr/bin/env python3
"""
Compute camera calibration

Use the images in calibration/, taken by running camera_calibration_record.py on
the RPi Zero and copying back the images, to compute the camera parameters.

Taken from:
https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
"""
import os
import glob
import cv2 as cv
import numpy as np

def find_images(input_dir, expr="*.jpg"):
    """ Find all images of a certain extension in a particular directory """
    return glob.glob(os.path.join(input_dir, expr))

def find_grid(images, horiz=8, vert=6, show_images=False):
    """
    Find the grid in all of the images (or try, ignore images where we can't)
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((horiz*vert,3), np.float32)
    objp[:,:2] = np.mgrid[0:vert,0:horiz].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (vert,horiz), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (vert,horiz), corners2, ret)

            if show_images:
                cv.imshow('img', img)
                cv.waitKey(500)

    cv.destroyAllWindows()

    return objpoints, imgpoints, shape

def calibrate(objpoints, imgpoints, shape):
    """ Calculate calibration """
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)

    return mtx, dist, rvecs, tvecs

def undistort(images, mtx, dist, show_images=False):
    """ Undistort the images using the calibration data """
    for filename in images:
        img = cv.imread(filename)
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        assert len(dst) > 0, "Something went wrong"

        if show_images:
            cv.imshow("img", img)
            cv.imshow("corrected", dst)
            cv.waitKey(2000)

        #cv.imwrite(images[0].replace("jpg", "png"), dst)

def calculate_error(objpoints, mtx, dist, rvecs, tvecs):
    mean_error = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    return mean_error/len(objpoints)

def save_camera_calibration(filename, mtx, dist):
    np.save(filename, {
        "mtx": mtx,
        "dist": dist,
    })

def load_camera_calibration(filename):
    d = np.load(filename).item()
    return d["mtx"], d["dist"]

if __name__ == "__main__":
    # Find our images
    images = find_images("calibration/")

    # Find grid in each image
    objpoints, imgpoints, shape = find_grid(images, show_images=False)

    # Generate calibration
    mtx, dist, rvecs, tvecs = calibrate(objpoints, imgpoints, shape)
    save_camera_calibration("camera_calibration.npy", mtx, dist)

    # Test undistorting
    undistort(images, mtx, dist, show_images=False)

    # Calculate error
    print("Total error:", calculate_error(objpoints, mtx, dist, rvecs, tvecs))
