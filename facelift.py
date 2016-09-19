# !/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyUnresolvedReferences,PyPackageRequirements
import os

import cv2
import numpy


def iter_images_in(folder_path):
    """
    Return an iterator to all images in the folder (.png and .jpg).
    :param folder_path: Folder where to load the images from.
    :return: Generator/Iterator of loaded images.
    """
    try:
        if folder_path[-1] != '/':
            folder_path += '/'
        files = os.listdir(folder_path)
        image_paths = [f for f in files
                       if f.endswith('.png') or f.endswith('.jpg')]
        images = (load_image(folder_path + path) for path in image_paths)
        return (image for image in images if image is not None)
    except OSError as e:
        if e.errno == 2:
            return None
        else:
            raise
    except IndexError:
        return None


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
        return numpy.empty((0, 4))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if isinstance(faces, tuple):
        return numpy.empty((0, 4))
    else:
        return faces


def calc_final_position_for_all(faces):
    """
    Calculate the point where all faces should be moved to.
    This is the centroid of all face rectangles to minimise movement.
    :param faces: Face rectangles to calculate the centroid from.
    :return: Centroid point or None if no rectangles.
    """
    if faces.size == 0:
        return None

    centres = map(calc_centre_of, faces)
    return numpy.mean(centres, axis=0)


def calc_centre_of(face):
    x, y, w, h = face
    assert w >= 0 and h >= 0
    return numpy.array([x + (w / 2.), y + (h / 2.)])


def calc_best_face_width_for_all(faces):
    assert all([f > 0 for f in faces[:, 2]])
    return numpy.average(faces[:, 2])


# noinspection PyTypeChecker
def calc_rectangle_for(centroid, width, height):
    assert isinstance(centroid, numpy.ndarray)
    size = numpy.array([width, height])
    return numpy.append((centroid - (size / 2)), size)


if __name__ == '__main__':
    def main():
        images = list(iter_images_in('test_resources/test_photos/'))
        faces = numpy.empty((0, 4))
        for i, img in enumerate(images):
            fcs = get_faces_in(img)
            faces = numpy.concatenate([faces, fcs], axis=0)
            for (x, y, w, h) in fcs:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.imshow('img%d' % i, img)

        final = numpy.zeros(images[0].shape, images[0].dtype)
        centroid = calc_final_position_for_all(faces)
        width = calc_best_face_width_for_all(faces)
        x, y, w, h = calc_rectangle_for(centroid, width, 20).astype(int)
        cv2.rectangle(final, (x, y), (x + w, y + h), (255, 0, 0), 5)

        cv2.imshow('final', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    main()
