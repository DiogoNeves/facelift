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


def calc_best_face_width_for_all(faces):
    """
    Calculate the best width for the final face size.
    :param faces: All faces we want to resize.
    :return: The final width.
    """
    assert all([f > 0 for f in faces[:, 2]])
    return numpy.average(faces[:, 2])


# noinspection PyTypeChecker
def calc_rectangle_for(centre, width, height):
    """
    Calculate a full rectangle with centre at centre and
    width and height.
    :param centre: Central point of the rectangle.
    :param width: Width of the rectangle.
    :param height: Height of the rectangle.
    :return: ndarray with the (x, y, w, h) of the rectangle.
    """
    assert isinstance(centre, numpy.ndarray)
    size = numpy.array([width, height])
    return numpy.append((centre - (size / 2)), size)


def draw_image_with_face(image, face, buffer, target_face_width):
    """
    Draw an image into the buffer after applying the needed
    transformations so that its face fills the final face rect.
    This changes buffer's contents.
    :param image: Image to draw.
    :param face: Valid face rectangle of that image.
    :param buffer: Buffer where to draw the image into.
    :param target_face_width: The face width ```face``` should have
        after drawing to buffer.
    """
    # Just a test
    # TODO: Calculate final position and draw
    s = 100
    buffer[:s, :s] = image[:s, :s]


if __name__ == '__main__':
    def main():
        images = list(iter_images_in('test_resources/test_photos/'))

        images_with_faces = []
        image_faces = []
        for img in images:
            faces = get_faces_in(img)
            if len(faces) > 0:
                images_with_faces.append(img)
                image_faces.append(faces[0])

        final = numpy.zeros(images[0].shape, images[0].dtype)
        faces = numpy.array(image_faces)
        width = calc_best_face_width_for_all(faces)

        # Testing, probably remove after
        centre = numpy.array(final.shape) / 2
        final_face_rect = calc_rectangle_for(centre, width, 20).astype(int)
        x, y, w, h = final_face_rect
        cv2.rectangle(final, (x, y), (x + w, y + h), (255, 0, 0), 5)

        for i in xrange(len(images_with_faces)):
            image = images_with_faces[i]
            x, y, w, h = face = image_faces[i]
            draw_image_with_face(image, face, final, width)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
            cv2.imshow('img%d' % i, image)

        cv2.imshow('final', final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    main()
