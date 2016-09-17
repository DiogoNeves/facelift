# !/usr/bin/env python
# -*- coding: utf-8 -*-

from facelift import load_image, get_faces_in, iter_images_in

TEST_FOLDER = 'test_resources/'
TEST_IMAGE_PATH = TEST_FOLDER + 'test_face.jpg'
TEST_PHOTO_FOLDER = TEST_FOLDER + 'test_photos/'


def test_it_loads_image():
    image = load_image(TEST_IMAGE_PATH)
    assert image is not None
    assert image.shape == (904, 904, 3)


def test_load_image_with_invalid_path_returns_none():
    image = load_image('invalid path')
    assert image is None


def test_it_detects_a_face():
    image = load_image(TEST_IMAGE_PATH)
    faces = get_faces_in(image)
    assert len(faces) == 1


def test_loading_folder_returns_iterator():
    image_folder = iter_images_in(TEST_PHOTO_FOLDER)
    assert image_folder


def test_loading_folder_returns_all_images():
    real_images = ['test_photo1.jpg']
    image_folder = list(iter_images_in(TEST_PHOTO_FOLDER))
    assert len(image_folder) == len(real_images)
    assert all([(img == load_image(TEST_PHOTO_FOLDER + path)).all()
                for img, path in zip(image_folder, real_images)])


def test_loading_invalid_folder_returns_none():
    assert iter_images_in('invalid folder') is None


def test_get_centre_of_rectangle():
    pass


def test_get_centre_of_invalid_rectangle_returns_none():
    pass
