# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pytest

from facelift import load_image, get_faces_in, iter_images_in, \
    calc_best_face_width_for_all, calc_rectangle_for

TEST_FOLDER = 'test_resources/'
TEST_FACE_PATH = TEST_FOLDER + 'test_face.jpg'
TEST_NO_FACE_PATH = TEST_FOLDER + 'test_no_face.jpg'
TEST_PHOTO_FOLDER = TEST_FOLDER + 'test_photos/'


def test_it_loads_image():
    image = load_image(TEST_FACE_PATH)
    assert image is not None
    assert image.shape == (904, 904, 3)


def test_load_image_with_invalid_path_returns_none():
    image = load_image('invalid path')
    assert image is None


def test_it_detects_a_face():
    image = load_image(TEST_FACE_PATH)
    faces = get_faces_in(image)
    assert len(faces) == 1


def test_it_doesnt_detect_face_where_it_doesnt_exist():
    image = load_image(TEST_NO_FACE_PATH)
    faces = get_faces_in(image)
    assert len(faces) == 0
    assert isinstance(faces, numpy.ndarray)
    assert faces.shape == (0, 4)


def test_loading_folder_returns_iterator():
    image_folder = iter_images_in(TEST_PHOTO_FOLDER, (100, 100))
    assert image_folder


def test_loading_folder_returns_all_images():
    real_images = [
        'test_photo1.jpg',
        'test_photo2.jpg',
        'test_photo3.png',
    ]
    image_folder = list(iter_images_in(TEST_PHOTO_FOLDER, (100, 100)))
    assert len(image_folder) == len(real_images)


def test_loading_folder_resize_images():
    size = (10, 10)
    image_folder = list(iter_images_in(TEST_PHOTO_FOLDER, size))
    assert all([image.shape[:2] <= size for image in image_folder])


def test_loading_folder_appends_slash_to_path():
    real_images = [
        'test_photo1.jpg',
        'test_photo2.jpg',
        'test_photo3.png',
    ]
    path = TEST_FOLDER + 'test_photos'
    image_folder = list(iter_images_in(path, (100, 100)))
    assert len(image_folder) == len(real_images)


def test_loading_invalid_folder_returns_none():
    assert iter_images_in('invalid folder', (100, 100)) is None


def test_loading_folder_with_empty_path_returns_none():
    assert iter_images_in('', (100, 100)) is None


def test_calc_best_face_width_for_all_single_face_returns_its_width():
    faces = numpy.array([(1, 3, 4, 5)])
    assert calc_best_face_width_for_all(faces) == 4.


def test_calc_best_face_width_for_all_two_faces_returns_average_width():
    faces = numpy.array([
        (1, 2, 3, 4),
        (0, 0, 6, 6)
    ])
    assert calc_best_face_width_for_all(faces) == 4.5


def test_calc_best_face_width_for_all_invalid_faces_asserts():
    faces = numpy.array([
        (0, 0, 2, 2),  # valid
        (0, 0, 0, 2)   # invalid
    ])
    with pytest.raises(AssertionError):
        calc_best_face_width_for_all(faces)


def test_calc_rectangle_for():
    centroid = numpy.array([1., 1.])
    rectangle = numpy.array([0, 0, 2, 2])
    assert (calc_rectangle_for(centroid, 2., 2.) == rectangle).all()


def test_calc_rectangle_for_asserts_invalid_type():
    with pytest.raises(AssertionError):
        calc_rectangle_for(1, 2, 2)
