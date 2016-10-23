# !/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyUnresolvedReferences,PyPackageRequirements
import os

import cv2
import numpy
from matplotlib import animation
from matplotlib import pyplot

_TARGET_WIDTH = 640


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
    This is not normalised by face size because we'll want to use the
    extra image size to fill empty spaces after transforming images to
    fit the final face rect.
    :param faces: All faces we want to resize.
    :return: The final width.
    """
    assert all([f > 0 for f in faces[:, 2]])
    return numpy.average(faces[:, 2])


def transformed_face(image, face, face_centre, target_face_width, frame_size):
    """
    Draw an image into the buffer after applying the needed
    transformations so that its face fills the final face rect.
    This changes buffer's contents.
    :param image: Image to draw.
    :param face: Valid face rectangle of that image.
    :param face_centre: Centre point of the transformed face.
    :param target_face_width: The face width `face` should have
        after drawing to buffer.
    :param frame_size: Size of the final frame.
    """
    _, _, face_width, _ = face
    size_ratio = target_face_width / float(face_width)

    image = cv2.resize(image, None, fx=size_ratio, fy=size_ratio,
                       interpolation=cv2.INTER_CUBIC)

    x, y, _, h = numpy.array(face) * size_ratio
    final_x, final_y, _, _ = calc_rectangle_for(face_centre, target_face_width, h)
    x, y = final_x - x, final_y - y

    translation = numpy.float32([[1, 0, x], [0, 1, y]])
    image = cv2.warpAffine(image, translation, frame_size)

    return image


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


def save_animation(images, filename):
    figure = pyplot.figure()
    axis = figure.add_subplot(111)
    axis.set_axis_off()

    images = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              for image in images)
    # TODO: Make it render from an iterator
    image_figures = [(axis.imshow(image), axis.set_title(''))
                     for image in images]
    image_animation = animation.ArtistAnimation(figure, image_figures,
                                                interval=200, repeat_delay=0,
                                                blit=False)

    image_animation.save(filename, writer='imagemagick')
    pyplot.show()


if __name__ == '__main__':
    def main():
        images = iter_images_in('photos/')
        images_with_faces = ((image, get_faces_in(image)) for image in images)
        images_with_faces = ((image, faces[0])
                             for image, faces in images_with_faces
                             if len(faces) > 0)

        width = _TARGET_WIDTH / 6
        centre = numpy.array((_TARGET_WIDTH / 2, _TARGET_WIDTH / 5))
        frame_size = (_TARGET_WIDTH, _TARGET_WIDTH)
        transformed = (transformed_face(image, face, centre, width, frame_size)
                       for image, face in images_with_faces)

        save_animation(transformed, 'output.gif')

    main()
