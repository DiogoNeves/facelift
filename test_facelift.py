# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy
import pytest

from facelift import load_image, get_faces_in, iter_images_in, \
    calc_centre_of, calc_centre_of_all

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


def test_loading_folder_returns_iterator():
    image_folder = iter_images_in(TEST_PHOTO_FOLDER)
    assert image_folder


def test_loading_folder_returns_all_images():
    real_images = [
        'test_photo1.jpg',
        'test_photo2.jpg',
        'test_photo3.png',
    ]
    image_folder = list(iter_images_in(TEST_PHOTO_FOLDER))
    assert len(image_folder) == len(real_images)
    assert all([(img == load_image(TEST_PHOTO_FOLDER + path)).all()
                for img, path in zip(image_folder, real_images)])


def test_loading_invalid_folder_returns_none():
    assert iter_images_in('invalid folder') is None


def test_calc_centre_of_rectangle():
    assert (calc_centre_of((0, 0, 2, 2)) == numpy.array([1, 1])).all()
    assert (calc_centre_of((0, 0, 3, 3)) == numpy.array([1.5, 1.5])).all()
    assert (calc_centre_of((3, 3, 2, 2)) == numpy.array([4., 4.])).all()
    assert (calc_centre_of((3, 3, 3, 2)) == numpy.array([4.5, 4.])).all()
    assert (calc_centre_of((-3, 3, 2, 2)) == numpy.array([-2, 4])).all()


def test_calc_centre_of_invalid_rectangle_returns_asserts():
    with pytest.raises(AssertionError):
        assert calc_centre_of((3, 3, 2, -2)) is None
    with pytest.raises(AssertionError):
        assert calc_centre_of((3, 3, -2, 2)) is None


def test_calc_centre_of_all_single_rectangle_returns_its_centre():
    rectangle = (1, 1, 2, 2)
    assert (calc_centre_of_all([rectangle]) == calc_centre_of(rectangle)).all()


def test_calc_centre_of_all_two_rectangles_returns_middle():
    rectangle1 = (0, 0, 2, 2)
    rectangle2 = (2, 2, 2, 2)
    centre = numpy.array([2., 2.])
    assert (calc_centre_of_all([rectangle1, rectangle2]) == centre).all()


def test_calc_centre_of_all_four_rectangles_returns_right_point():
    rectangles = [
        (0, 0, 2, 2),
        (2, 2, 2, 2),
        (2, 0, 2, 2),
        (0, 2, 2, 2)
    ]
    centre = numpy.array([2., 2.])
    assert (calc_centre_of_all(rectangles) == centre).all()


def test_calc_centre_of_all_empty_returns_none():
    assert calc_centre_of_all([]) is None
