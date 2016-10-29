# !/usr/bin/env python
# -*- coding: utf-8 -*-

# noinspection PyUnresolvedReferences,PyPackageRequirements
import argparse
import os

import cv2
import numpy
from PIL import Image


def iter_images_in(folder_path, frame_size):
    """
    Return an iterator to all images in the folder (.png and .jpg).
    :param folder_path: Folder where to load the images from.
    :param frame_size: Target frame size for all images.
    :return: Generator/Iterator of loaded images.
    """
    try:
        if folder_path[-1] != '/':
            folder_path += '/'
        files = os.listdir(folder_path)
        image_paths = (f for f in files
                       if f.lower()[-4:] in ['.png', '.jpg'])
        images = (load_image(folder_path + path) for path in image_paths)
        return (resize_image(image, frame_size)
                for image in images if image is not None)
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


def resize_image(image, frame_size):
    x, y, __ = image.shape
    if abs(x - frame_size[0]) >= abs(y - frame_size[1]):
        ratio = (frame_size[0] * 2) / float(x)
    else:
        ratio = (frame_size[1] * 2) / float(y)
    return cv2.resize(image, None, fx=ratio, fy=ratio,
                      interpolation=cv2.INTER_CUBIC)


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
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=4, minSize=(24, 24),
                                          flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
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
    __, __, face_width, __ = face
    size_ratio = target_face_width / float(face_width)

    image = cv2.resize(image, None, fx=size_ratio, fy=size_ratio,
                       interpolation=cv2.INTER_CUBIC)

    x, y, __, h = numpy.array(face) * size_ratio
    final_x, final_y, __, __ = calc_rectangle_for(face_centre, face_width, h)
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
    """
    Create and save animation using images provided.
    :param images: Images to animate.
    :param filename: Filename of the output file.
    :return: Number of images saved.
    """
    images = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images)
    images = [Image.fromarray(image) for image in images]
    if len(images) > 1:
        frame = images[0]
        with open(filename, 'wb') as out:
            frame.save(out, save_all=True, append_images=images[1:],
                       duration=450)
    return len(images)


if __name__ == '__main__':
    def parse_args():
        description = 'Create gif animations of people with their faces centred'
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument('--frame-size', nargs=2, type=int,
                            help='Output image size (width height) in pixels.',
                            default=[640, 640])
        parser.add_argument('--dir', nargs=1, type=unicode,
                            help='Directory where to load the images from.',
                            default=['photos/'])
        parser.add_argument('--face-width', nargs=1, type=float,
                            help='Face width in the output file as a %% of '
                                 '--frame-size.',
                            default=[0.25])
        parser.add_argument('--centre', nargs=2, type=float,
                            help='Face centre position in the output file as '
                                 'a %% of the --frame-size',
                            default=[0.5, 0.5])

        args = parser.parse_args()

        directory = args.dir[0]
        frame_size = tuple(args.frame_size)
        width = frame_size[0]
        face_width = width * args.face_width[0]
        centre = numpy.array((width * args.centre[0], width * args.centre[1]))

        return directory, face_width, centre, frame_size


    def main(directory, target_face_width, centre, frame_size):
        images = iter_images_in(directory, frame_size)
        images_with_faces = ((image, get_faces_in(image)) for image in images)
        images_with_faces = ((image, faces[0])
                             for image, faces in images_with_faces
                             if len(faces) > 0)

        width = target_face_width
        transformed = (transformed_face(image, face, centre, width, frame_size)
                       for image, face in images_with_faces)

        saved_count = save_animation(transformed, 'output.gif')
        print 'Saved {} images from "{}"'.format(saved_count, directory)


    main(*parse_args())
