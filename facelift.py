# !/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyUnresolvedReferences,PyPackageRequirements
import os

import cv2


def iter_images_in(folder_path):
    files = os.listdir(folder_path)
    image_paths = [f for f in files if f.endswith('.png') or f.endswith('.jpg')]
    images = (load_image(folder_path + path) for path in image_paths)
    return (image for image in images if image is not None)


def load_image(image_path):
    """
    Load image from image file path. Currently serves as a wrapper for
    OpenCV library.
    :param image_path: Valid path to image.
    :return: Image object.
    """
    assert image_path
    return cv2.imread(image_path)


def get_faces_in(image):
    """
    Detect all faces in the image.
    :param image: Image with faces to detect.
    :return: Tuple of face region rectangles, empty if none detected.
    """
    classifier_name = 'resources/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(classifier_name)
    if face_cascade.empty():
        if __name__ == '__main__':
            print 'Failed to load Classifier "%s"' % classifier_name
        return tuple()

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return face_cascade.detectMultiScale(gray, 1.3, 5)


if __name__ == '__main__':
    img = load_image('test_resources/test_face.jpg')
    fcs = get_faces_in(img)
    for (x, y, w, h) in fcs:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
