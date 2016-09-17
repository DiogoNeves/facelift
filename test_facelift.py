# !/usr/bin/env python
# -*- coding: utf-8 -*-
from facelift import load_image, get_faces_in

TEST_IMAGE_PATH = 'test_resources/test_face.jpg'


def test_it_detects_a_face():
    image = load_image(TEST_IMAGE_PATH)
    faces = get_faces_in(image)
    assert len(faces) == 1
