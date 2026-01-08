# FILE : temperature.py
# WRITER : Elal Gilboa , elal.gilboa , 323083188
# EXERCISE : intro2cs ex2 2025
# DESCRIPTION: This program checks if voldmir is safe depending on temperature mesures of previous 3 days and minimal
#              necessary temperature.
# STUDENTS I DISCUSSED THE EXERCISE WITH: None
# WEB PAGES I USED: https://www.w3schools.com/python/ref_string_split.asp
# NOTES: None

from image_editor import *


def test_combine_channels():
    """This function test combine_channels()"""
    assert combine_channels([[[1]], [[2]]]) == [[[1, 2]]]
    image_lst = [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                 [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]]
    image = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    assert combine_channels(image_lst) == image


def test_separate_channels():
    """This function test combine_channels()"""
    assert separate_channels([[[1, 2]]]) == [[[1]], [[2]]]
    image = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
             [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    image_lst = [[[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
                 [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]]
    assert separate_channels(image) == image_lst


def test_RGB2grayscale():
    """This function test RGB2grayscale"""
    image1 = [[[100, 180, 240]]]
    image2 = [[[200, 0, 14], [15, 6, 50]]]

    assert RGB2grayscale(image1) == [[163]]
    assert RGB2grayscale(image2) == [[61, 14]]


def test_cell_on_edge():
    """this function test check_on_edge() with different sizes of kernels"""
    matrix = [[1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4, 5]]
    kernel_size = 5
    assert check_on_edge(matrix, kernel_size, 0, 1) == True
    assert check_on_edge(matrix, kernel_size, 4, 1) == True
    assert check_on_edge(matrix, kernel_size, 4, 4) == True
    assert check_on_edge(matrix, kernel_size, 3, 3) == True
    assert check_on_edge(matrix, kernel_size, 2, 2) == False
    kernel_size = 3
    assert check_on_edge(matrix, kernel_size, 0, 1) == True
    assert check_on_edge(matrix, kernel_size, 4, 1) == True
    assert check_on_edge(matrix, kernel_size, 4, 4) == True
    assert check_on_edge(matrix, kernel_size, 3, 3) == False
    assert check_on_edge(matrix, kernel_size, 2, 2) == False
    assert check_on_edge(matrix, kernel_size, 1, 2) == False


def test_calc_kernel():
    """This function test calc_kernel() and blur_kernel()"""
    blur_3 = [[1 / (3 ** 2) for i in range(3)] for j in range(3)]
    blur_5 = [[1 / (5 ** 2) for i in range(5)] for j in range(5)]
    assert blur_kernel(3) == blur_3
    assert blur_kernel(5) == blur_5
    matrix1 = [[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]]
    matrix2 = [[3, 4, 5],
               [3, 4, 5],
               [3, 4, 5]]
    matrix3 = [[2, 3, 4],
               [2, 3, 4],
               [2, 3, 4]]
    matrix4 = [[0, 0, -1],
               [-1, 0, 0],
               [-1, 0, 0]]
    matrix5 = [[500, 500, 500],
               [500, 500, 500],
               [500, 500, 500]]
    assert calc_kernel(matrix1, blur_3) == 2
    assert calc_kernel(matrix2, blur_3) == 4
    assert calc_kernel(matrix3, blur_3) == 3
    assert calc_kernel(matrix4, blur_3) == 0
    assert calc_kernel(matrix5, blur_3) == 255


def test_const_center_matrix():
    """This function test const_center_matrix()"""
    image = [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]]
    size = 3
    assert construct_cntr_matrix(image, size, 1, 1) == [[1, 2, 3], [1, 2, 3],
                                                        [1, 2, 3]]
    assert construct_cntr_matrix(image, size, 3, 3) == [[3, 4, 5], [3, 4, 5],
                                                        [3, 4, 5]]


def test_construct_edge_matrix():
    """This function test const_edge_matrix()"""
    image = [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]]
    size = 3
    assert construct_edge_matrix(image, size, 0, 0) == [[1, 1, 1],
                                                        [1, 1, 2],
                                                        [1, 1, 2]]
    assert construct_edge_matrix(image, size, 4, 4) == [[4, 5, 5],
                                                        [4, 5, 5],
                                                        [5, 5, 5]]
    assert construct_edge_matrix(image, size, 4, 2) == [[2, 3, 4],
                                                        [2, 3, 4],
                                                        [3, 3, 3]]
    size = 5
    assert construct_edge_matrix(image, size, 0, 0) == [[1, 1, 1, 1, 1],
                                                        [1, 1, 1, 1, 1],
                                                        [1, 1, 1, 2, 3],
                                                        [1, 1, 1, 2, 3],
                                                        [1, 1, 1, 2, 3]]

    assert construct_edge_matrix(image, size, 4, 4) == [[3, 4, 5, 5, 5],
                                                        [3, 4, 5, 5, 5],
                                                        [3, 4, 5, 5, 5],
                                                        [5, 5, 5, 5, 5],
                                                        [5, 5, 5, 5, 5]]

    assert construct_edge_matrix(image, size, 4, 2) == [[1, 2, 3, 4, 5],
                                                        [1, 2, 3, 4, 5],
                                                        [1, 2, 3, 4, 5],
                                                        [3, 3, 3, 3, 3],
                                                        [3, 3, 3, 3, 3]]


def test_apply_kernel():
    """This function test apply kernel()"""
    image = [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]]
    size = 3
    kernel = blur_kernel(size)
    assert apply_kernel(image, kernel) == [[1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5],
                                           [1, 2, 3, 4, 5]]
    assert apply_kernel([[0, 128, 255]], blur_kernel(3)) == [[14, 128, 241]]
    mat_after_kernel = [[12, 20, 26, 34, 44],
                        [11, 17, 22, 27, 34],
                        [10, 16, 20, 24, 29],
                        [7, 11, 16, 18, 21]]
    assert apply_kernel([[10, 20, 30, 40, 50],
                         [8, 16, 24, 32, 40],
                         [6, 12, 18, 24, 30],
                         [4, 8, 12, 16, 20]],
                        blur_kernel(5)) == mat_after_kernel


def test_calc_distances():
    pass


def test_bilinear_interpolation() -> None:
    """This function test bilinear_interpolation()"""
    assert bilinear_interpolation([[0, 64], [128, 255]], 0, 0) == 0
    assert bilinear_interpolation([[0, 64], [128, 255]], 1, 1) == 255
    assert bilinear_interpolation([[0, 64], [128, 255]], 0.5, 0.5) == 112
    assert bilinear_interpolation([[0, 64], [128, 255]], 0.5, 1) == 160
    assert bilinear_interpolation([[15, 30, 45, 60, 75],
                                   [90, 105, 120, 135, 150],
                                   [165, 180, 195, 210, 225]], 4 / 5,
                                  8 / 3) == 115


def test_calc_relative_loc():
    image = [[0 for j in range(5)] for i in range(3)]
    new_image = [[0 for j in range(7)] for i in range(6)]
    assert calc_relative_loc(image, new_image, 2, 4) == (4 / 5, 8 / 3)


def test_locate_corners():
    image = [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]]
    new_image = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
    locate_corners(image, new_image)
    assert new_image == [[1, 0, 0, 5],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 5]]
    new_image2 = [[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]]
    locate_corners(image, new_image2)
    assert new_image2 == [[1, 0, 0, 0, 0, 5],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 0, 5]]


def test_resize():
    assert resize([[0, 64, 128],
                   [0, 128, 255],
                   [100, 100, 100]], 2, 2) == [[0, 128], [100, 100]]
    assert resize([[0, 13, 26, 39],
                   [52, 65, 78, 91],
                   [104, 117, 130, 143],
                   [156, 169, 182, 195],
                   [208, 221, 234, 247]], 2, 3) == [[0, 20, 39],
                                                    [208, 228, 247]]


def test_rotate_90():
    assert rotate_90([[1, 2, 3], [4, 5, 6]], 'R') == [[4, 1], [5, 2], [6, 3]]
    assert rotate_90([[1, 2, 3], [4, 5, 6]], 'L') == [[3, 6], [2, 5], [1, 4]]
    assert rotate_90([[[1, 2, 3], [4, 5, 6]], [[0, 5, 9], [255, 200, 7]]],
                     'L') == [[[4, 5, 6], [255, 200, 7]],
                              [[1, 2, 3], [0, 5, 9]]]


def test_get_edges():
    assert get_edges([[200, 50, 200]], 3, 3, 10) == [[255, 0, 255]]


def test_calc_avg():
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert calc_average(mat) == 5


def test_calc_threshold():
    mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert calc_threshold(mat, 3, 0) == [[17 / 9, 3, 31 / 9],
                                         [39 / 9, 5, 51 / 9],
                                         [59 / 9, 7, 73 / 9]]


def test_quantize_image():
    assert quantize([[0, 50, 100], [150, 200, 250]], 8) == [[0, 36, 109],
                                                            [146, 219, 255]]
