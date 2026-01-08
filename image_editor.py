#################################################################
# FILE : image_editor.py
# WRITER : Elal Gilboa , elal.gilboa , 323083188
# EXERCISE : intro2cs ex6 2025
# DESCRIPTION: A simple program that...
# STUDENTS I DISCUSSED THE EXERCISE WITH: None
# WEB PAGES I USED: https://www.w3schools.com/python/ref_func_round.asp
#                   https://datacarpentry.github.io/image-processing/06-blurring.html
#                   https://stackoverflow.com/questions/41290350/inplace-rotation-of-a-matrix
#                   https://pillow.readthedocs.io/en/stable/reference/Image.html
# NOTES: ...
#################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from ex6_helper import *
from typing import List, Dict, Union, Tuple
from copy import deepcopy
import math
import sys
import os.path
import PIL

##############################################################################
#                                   Typing                                   #
##############################################################################
SingleChannelImage = List[List[int]]
ColoredImage = List[List[List[int]]]
Image = Union[ColoredImage, SingleChannelImage]
Kernel = List[List[float]]


##############################################################################
#                                  Functions                                 #
##############################################################################

def separate_channels(image: ColoredImage) -> List[SingleChannelImage]:
    """This function receive an image, check how many color channel it has.
    And separate the image to the matching color channels. Func return a list
    of images in Single Color channel"""
    height: int = len(image)
    width: int = len(image[0])
    num_of_channels = len(image[0][0])
    separated_image = [[[0 for k in range(width)] for m in range(height)] for n
                       in range(num_of_channels)]
    for i in range(height):
        for j in range(width):
            pixel: List[int] = image[i][j]
            for channel in range(len(pixel)):
                separated_image[channel][i][j] = pixel[channel]
    return separated_image


def image_constructor(height: int, width: int,
                      num_channels: int) -> ColoredImage:
    """This function is an image constructor - receive image sizes and return
     and black RGB formated List[List[List[int]]] with all values equal to 0"""
    new_image: list[list[List[int]]] = []
    for i in range(height):
        new_image.append([])
        for j in range(width):
            new_image[-1].append([0 for k in range(num_channels)])
    return new_image


def combine_channels(channels: List[SingleChannelImage]) -> ColoredImage:
    """This function receive a list of separated color channels and return
    one combined list of all channels"""
    height = len(channels[0])
    width = len(channels[0][0])
    num_channels = len(channels)
    combined_image = image_constructor(height, width, num_channels)
    for i in range(height):
        for j in range(width):
            pixel: List[int] = []
            for channel in channels:
                pixel.append(channel[i][j])
            combined_image[i][j] = pixel
    return combined_image


def RGB2grayscale(colored_image: ColoredImage) -> SingleChannelImage:
    """This function receive an RGB image and return grayscale image -
    List[List[int]]. The function does not change original image"""
    RED_CONSTANT: float = 0.299
    GREEN_CONSTANT: float = 0.587
    BLUE_CONSTANT: float = 0.114
    grayscale_image: List[List[int]] = []
    height: int = len(colored_image)
    width: int = len(colored_image[0])
    for i in range(height):
        grayscale_image.append([])
        for j in range(width):
            red_value, green_value, blue_value = colored_image[i][j]
            grayscale_value = (red_value * RED_CONSTANT +
                               green_value * GREEN_CONSTANT +
                               blue_value * BLUE_CONSTANT)
            grayscale_image[-1].append(round(grayscale_value))
    return grayscale_image


def blur_kernel(size: int) -> Kernel:
    """This function receive a size for a blur kernel and return a list[list[int]]
    with sizes: size X size. Each cell has the same value - 1/(size^2)"""
    CELL_VALUE: float = 1 / (math.pow(size, 2))
    kernel: List[List[float]] = []
    for i in range(size):
        kernel.append([])
        for j in range(size):
            kernel[-1].append(CELL_VALUE)
    return kernel


def check_on_edge(image: List[List[float]], kernel_size: int, i: int,
                  j: int) -> bool:
    """This function receive a matrix and (i,j) index and return True if it is
    on the edge of the matrix - meaning (i,j) is inside the buffer frame and
    False otherwise"""
    BUFFER_EDGE: int = kernel_size // 2  # size is an odd number
    height: int = len(image)
    width: int = len(image[0])
    if i == 0 or j == 0:
        # 0 is the first line / column in a matrix
        return True
    elif i == height - 1 or j == width - 1:
        # this is the last line / column
        return True
    elif not (((BUFFER_EDGE + i) < height) and (i - BUFFER_EDGE >= 0)):
        # in case kernel is not 3*3 - buffer is bigger than 1
        return True
    elif not (((BUFFER_EDGE + j) < width) and (j - BUFFER_EDGE >= 0)):
        return True
    else:
        return False


def calc_kernel(matrix: List[List[int]], kernel: List[List[float]]) -> int:
    """This function receive a matrix, a kernel This function calc kernel by
    multiplying each matrix cell with the matching kernel cell and sum the result.
     Function return an int - calculation rounded result. If calc < 0 func
    return 0, if calc > 255 - func return 255 """
    MAX_RGB_VALUE: int = 255
    MIN_RGB_VALUE: int = 0
    height: int = len(matrix)
    width: int = len(matrix[0])
    sum_kernel: int = 0
    for i in range(height):
        for j in range(width):
            sum_kernel += kernel[i][j] * matrix[i][j]
    if sum_kernel < MIN_RGB_VALUE:
        return MIN_RGB_VALUE
    elif sum_kernel > MAX_RGB_VALUE:
        return MAX_RGB_VALUE
    else:
        return round(sum_kernel)


def construct_cntr_matrix(image: List[List[int]], size: int, i: int, j: int) -> \
        List[List[int]]:
    """This function receive an image, a requested size for new_matrix and an
    index (i,j) for the center of the new_matrix. Function assume (i,j) is not
    on edge of the image."""
    START_I: int = max(i - 1, 0)
    START_J: int = max(j - 1, 0)
    new_matrix_i: int = 0
    new_matrix_j: int = 0
    new_matrix = [[0 for i in range(size)] for j in range(size)]
    for line in range(START_I, min(START_I + size, len(image))):
        for col in range(START_J, min(START_J + size, len(image[0]))):
            new_matrix[new_matrix_i][new_matrix_j]: int = image[line][col]
            new_matrix_j += 1
        new_matrix_i += 1
        new_matrix_j = 0
        # both new_matrix and required size of the image are same size
        # but different indexes - that's why there are 2 sets of indexes
    return new_matrix


def construct_edge_matrix(image: List[List[int]], size: int, i: int, j: int) -> \
        List[List[int]]:
    """This function receive an image, a requested size for new_matrix and an
    index (i,j) for the center of the new_matrix. Function assume (i,j) is on
    the edge of the image and complete the missing cells to the value of (i,j)
    """
    BUFFER_SIZE = size // 2
    CENTER_CELL = image[i][j]
    new_matrix = [[CENTER_CELL for i in range(size)] for j in range(size)]
    for line in range(size):
        for column in range(size):
            original_i = i + line - BUFFER_SIZE
            original_j = j + column - BUFFER_SIZE
            if 0 <= original_i < len(image) and 0 <= original_j < len(
                    image[0]):
                #  this condition checks if the original indexes are within
                #  range, otherwise pass (center value already in cells)
                new_matrix[line][column] = image[original_i][original_j]
    return new_matrix


def apply_kernel(image: SingleChannelImage,
                 kernel: Kernel) -> SingleChannelImage:
    """This function receive an image and apply kernel on image, function
    return new_image. Function is using check_on_edge() to check if (i,j) point
    is in the buffer area of the kernel or not. In both cases a matching
    matrix is created and sent to calc_kernel() for the actual calc. The matrix
    is different between edge point and center point."""
    height: int = len(image)
    width: int = len(image[0])
    new_image: List[List[int]] = [[0 for j in range(width)] for i in
                                  range(height)]
    for i in range(height):
        for j in range(width):
            if check_on_edge(image, len(kernel), i, j):
                edge_matrix = construct_edge_matrix(image, len(kernel), i, j)
                new_image[i][j] = calc_kernel(edge_matrix, kernel)
            else:
                matrix = construct_cntr_matrix(image, len(kernel), i, j)
                new_image[i][j] = calc_kernel(matrix, kernel)
    return new_image


def sort_min_centers(min_centers: List[Tuple[int, int]]) -> List[
    Tuple[int, int]]:
    """This function receive 4 min_centers are return them sorted to a,b,c,d
    correct order. Function return a List of tuples sorted a,b,c,d"""
    min_centers_copy = deepcopy(min_centers)
    a_cell: Tuple[int, int] = min(min_centers_copy)  # min_i, min_j
    min_centers_copy.remove(a_cell)
    c_cell: Tuple[int, int] = min(min_centers_copy)  # min_i, j>min_j
    min_centers_copy.remove(c_cell)
    b_cell: Tuple[int, int] = min(min_centers_copy)  # i>min_i, min_j
    min_centers_copy.remove(b_cell)
    d_cell: Tuple[int, int] = min(min_centers_copy)  # i>min_i, j>min_j
    return [a_cell, b_cell, c_cell, d_cell]


def calc_distances(cell_centers: List[Tuple[int, int]], y: float, x: float) -> \
        Dict[Tuple[int, int], float]:
    """This function receive a list of points (int, int) and return
    a Dict of Tuples(y1,x1): distance between each point to the point (y,x) """
    # distance_formula = square_root((x1-x2)^2 + (y1-y2)^2))
    relative_dist_dict: Dict[Tuple[int, int], float] = {}
    # Dict[(y1,x1):distance]
    for y1, x1 in cell_centers:
        if not (abs(x - x1) > 1 or abs(y - y1) > 1):
            distance: float = math.sqrt((y1 - y) ** 2 + (x1 - x) ** 2)
            relative_dist_dict[(y1, x1)] = distance
    return relative_dist_dict


def find_min_dist_to_centers(cell_centers: List[Tuple[int, int]], y: float,
                             x: float) -> List[Tuple[int, int]]:
    """This function receive a list of tuples[int,int] of the center points
    of the original image, calc the distance to point (y,x) from each point
    and return the 4 closest cells to (x,y)"""
    DST_IN_TUP = 1
    RQRD_MIN_CNTR = 4
    distances: Dict[Tuple[int, int], float] = calc_distances(cell_centers, y,
                                                             x)
    min_distances = sorted(distances.items(),
                           key=lambda item: item[DST_IN_TUP])[:RQRD_MIN_CNTR]
    # Extract only the points (keys)
    min_cntr_point = [center for center, tup in min_distances]
    a_cell, b_cell, c_cell, d_cell = sort_min_centers(min_cntr_point)
    return [a_cell, b_cell, c_cell, d_cell]


def find_closest_centers(image: List[List[int]], y: float, x: float) -> Tuple[
    int, int, int, int, float, float]:
    """This function receive an image, [y,x] location in original image and
    return the value of a,b,c,d,delta_x,delta_y cell values in original image.
    a,b,c,d are the nearest centers to [y,x] """
    X_LOC = 1
    Y_LOC = 0
    height: int = len(image)
    width: int = len(image[0])
    cell_cntrs: List[Tuple[int, int]] = [(i, j) for i in range(height) for j in
                                         range(width) if not (
                abs(y - i) > 1 or abs(x - j) > 1)]
    a_cell, b_cell, c_cell, d_cell = find_min_dist_to_centers(cell_cntrs, y, x)
    a_cell_value: int = image[a_cell[Y_LOC]][a_cell[X_LOC]]
    b_cell_value: int = image[b_cell[Y_LOC]][b_cell[X_LOC]]
    c_cell_value: int = image[c_cell[Y_LOC]][c_cell[X_LOC]]
    d_cell_value: int = image[d_cell[Y_LOC]][d_cell[X_LOC]]
    delta_y: float = y - a_cell[Y_LOC]
    delta_x: float = x - a_cell[X_LOC]
    return a_cell_value, b_cell_value, c_cell_value, d_cell_value, delta_x, delta_y


def bilinear_interpolation(image: SingleChannelImage, y: float,
                           x: float) -> int:
    """This function receive (y,x) location of destination pixel in the original
     image. The function use bilinear interpolation formula and return round
     value of pixel for destination image"""
    a, b, c, d, delta_x, delta_y = find_closest_centers(image, y, x)
    pix_val: float = (a * (1 - delta_x) * (1 - delta_y)) + (
            b * delta_y * (1 - delta_x)) + (
                             c * delta_x * (1 - delta_y)) + (
                             d * delta_x * delta_y)
    return round(pix_val)


def locate_corners(image: List[List[int]], new_image: List[List[int]]) -> None:
    """This function receive original image and a new image and place cell values
    of all 4 corners of original image in the new one"""
    R_U = (0, 0)  # right_up corner
    R_D = (len(image) - 1, 0)  # right_down_corner
    L_U = (0, len(image[0]) - 1)  # left_up_corner
    L_D = (len(image) - 1, len(image[0]) - 1)  # left_down_corner
    R_U_NEW = (0, 0)  # right_up corner
    R_D_NEW = (len(new_image) - 1, 0)  # right_down_corner
    L_U_NEW = (0, len(new_image[0]) - 1)  # left_up_corner
    L_D_NEW = (len(new_image) - 1, len(new_image[0]) - 1)  # left_down_corner
    i_orig, j_orig = R_U
    i_new, j_new = R_U_NEW
    new_image[i_new][j_new] = image[i_orig][j_orig]
    # locate up_right corner
    i_orig, j_orig = R_D
    i_new, j_new = R_D_NEW
    new_image[i_new][j_new] = image[i_orig][j_orig]
    # locate right_down corner
    i_orig, j_orig = L_U
    i_new, j_new = L_U_NEW
    new_image[i_new][j_new] = image[i_orig][j_orig]
    # locate left_up corner
    i_orig, j_orig = L_D
    i_new, j_new = L_D_NEW
    new_image[i_new][j_new] = image[i_orig][j_orig]
    # locate left_down corner


def calc_relative_loc(image, new_image, y, x) -> Tuple[float, float]:
    """This function receive original image, new image and (y,x) in the new
    image. Func calculate the relative location of (y,x) in the original image.
    Func return a Tuple(orig_y, orig_x)"""
    NEW_HEIGHT = len(new_image) - 1
    NEW_WIDTH = len(new_image[0]) - 1
    ORIG_HEIGHT = len(image) - 1
    ORIG_WIDTH = len(image[0]) - 1
    relative_new_y = y / NEW_HEIGHT
    relative_new_x = x / NEW_WIDTH
    orig_y = relative_new_y * ORIG_HEIGHT
    orig_x = relative_new_x * ORIG_WIDTH
    return orig_y, orig_x


def resize(image: SingleChannelImage, new_height: int,
           new_width: int) -> SingleChannelImage:
    """This function receive an original image and sizes for new_image. Func
    locate 4 points of original image to the 4 point of the new one. For every
    other cell func use: bilinear_interpolation() to calc the new image relative
    color cell value. Func return new image resized"""
    new_image: List[List[int]] = [[0 for j in range(new_width)] for i in
                                  range(new_height)]
    # locate 4 corners of original image in the new one's corners
    locate_corners(image, new_image)
    for y in range(new_height):
        for x in range(new_width):
            relative_y, relative_x = calc_relative_loc(image, new_image, y, x)
            new_cell_val: int = bilinear_interpolation(image, relative_y,
                                                       relative_x)
            new_image[y][x]: int = new_cell_val
    return new_image


def rotate_90(image: Image, direction: str) -> Image:
    """This function receive an image and a string with direction to rotate.
    In order to rotate an image we insert (i,j) to location (j,i) than reverse
    every line in the matrix"""
    RIGHT_ROTATION: str = "R"
    LEFT_ROTATION: str = "L"
    height: int = len(image)
    width: int = len(image[0])
    max_i_index = height - 1
    max_j_index = width - 1
    rotated_image: List[List[Union[int, List[int]]]] = [
        [0 for k in range(height)] for m in range(width)]
    if direction == RIGHT_ROTATION:
        for i in range(height):
            for j in range(width):
                rotated_image[j][max_i_index - i] = image[i][j]
    elif direction == LEFT_ROTATION:
        for i in range(height):
            for j in range(width):
                rotated_image[max_j_index - j][i] = image[i][j]
    return rotated_image


def calc_average(matrix: List[List[int]]) -> float:
    """This func receive a matrix and calc the average of its cells.
    Func return a float value for avg"""
    avg: float = 0
    var_counter: int = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            avg += matrix[i][j]
            var_counter += 1
    return avg / var_counter


def calc_threshold(blured_image: List[List[int]], block_size: int, c: float) -> \
        List[List[float]]:
    """This func receive a blured image, block size and constant c. It calc the
    average around each cell and place it in a threshold. Func return threshold.
    Func is using apply_kernel() func to calc average"""
    height: int = len(blured_image)
    width: int = len(blured_image[0])
    # r: int = block_size // 2
    threshold: List[List[float]] = [[0 for j in range(width)] for i in
                                    range(height)]
    for i in range(height):
        for j in range(width):
            if check_on_edge(blured_image, block_size, i, j):
                edge_matrix = construct_edge_matrix(blured_image, block_size,
                                                    i, j)
                threshold[i][j] = calc_average(edge_matrix) - c
            else:
                matrix = construct_cntr_matrix(blured_image, block_size, i, j)
                threshold[i][j] = calc_average(matrix) - c
    return threshold


def get_edges(image: SingleChannelImage, blur_size: int, block_size: int,
              c: float) -> SingleChannelImage:
    """This func receive an image, a blur_size, block_size and constant c. Func
    calc the threshold (minimal limit for each cell) and compare the threshold
    value to a blured image value (to reduce noice). In case blured_cell is
    smaller than threshold cell - it's an edge (black) otherwise it's not an edge
    (white). Func return new image."""
    BLACK_CELL: int = 0
    WHITE_CELL: int = 255
    height: int = len(image)
    width: int = len(image[0])
    kernel: List[List[float]] = blur_kernel(blur_size)
    blured_image: List[List[int]] = apply_kernel(image, kernel)
    threshold = calc_threshold(blured_image, block_size, c)
    new_image: List[List[int]] = [[0 for k in range(width)] for m in
                                  range(height)]
    for i in range(height):
        for j in range(width):
            if blured_image[i][j] < threshold[i][j]:
                cell_val = BLACK_CELL
            else:
                cell_val = WHITE_CELL
            new_image[i][j] = cell_val
    return new_image


def quantize(image: SingleChannelImage, N: int) -> SingleChannelImage:
    """This function receive an image and N constant for N colors to keep.
    and return quantized image with N colors"""
    height: int = len(image)
    width: int = len(image[0])
    COLOR_RANGE = 256
    LOWER_C_RANGE = 255
    q_image: List[List[int]] = [[0 for k in range(width)] for m in
                                range(height)]
    for i in range(height):
        for j in range(width):
            q_image[i][j] = round(
                math.floor(image[i][j] * N / COLOR_RANGE) * LOWER_C_RANGE / (
                        N - 1))
    return q_image


def get_image_path() -> Union[str, None]:
    """This func get image path from sys.argv, if path does not exist - throw
    exception"""
    PATH_LOC = 1
    image_path: Union[str, None] = None
    try:
        args: List[str] = sys.argv[:]
        image_path = args[PATH_LOC]
    except:
        print("ARGUMENTS INSERTED ARE INVALID."
              " Insert: image_editor.py <image_path>")
    finally:
        return image_path


def check_valid_path(path) -> bool:
    """This func test image path, if it's valid return True, False otherwise"""
    try:
        with open(path, 'r') as f:
            f.close()
        return True
    except FileNotFoundError:
        print("Image path is not valid. Try again")
        return False


def check_grayscale(image: ColoredImage) -> bool:
    """This func Checks if an RGB image is grayscale by verifying R,G,B  values
    are equal for all pixels. Func return True if it's grayscale and False
    otherwise"""
    try:
        for line in image:
            for pix in line:
                r, g, b = pix
                if r != g or r != b:
                    return False
                    # un-equal RGB values - mean it's not grayscale
    except:
        return True


def check_kernel_size(kernel_size: str) -> bool:
    """This function check if kernel size string is an integer and odd.
    Return True if it is and False otherwise"""
    if kernel_size.isdigit():
        if int(kernel_size) % 2 == 1:
            return True
        else:
            return False
    else:
        return False


def check_sizes(height: str, width: str):
    """This function receive to sizes for resize and return True if they
    are both integer and False otherwise"""
    return height.isdigit() and width.isdigit()


def check_edges_parms(blur_size, block_size, c) -> bool:
    """This func receive sizes for get_edges(). It checks if blur_size and
    block_size are odd integers and c is positive float."""
    if check_kernel_size(blur_size) and check_kernel_size(block_size):
        if c.isdecimal() and float(c) > 0:
            return True
        else:
            return False
    else:
        return False


def check_n_colors(n: str) -> bool:
    """This func receive a -  N string and return True if its integer bigger than
    1 and False otherwise"""
    if n.isdigit():
        return int(n) > 1
    else:
        return False


def apply_quantize(image: Image) -> Image:
    """This function check if image is RGB or grayscale and apply to quantize
    accordingly. If image is grayscale - quantize applied only on one color.
    If it's RGB quantize applied on every channel separately """
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    RED_CH_LOC = 0
    GREEN_CH_LOC = 1
    BLUE_CH_LOC = 2
    if check_grayscale(image):
        mode = GRAYSCALE_MODE
    else:
        mode = RGB_MODE
    n_colors: str = input("Insert N colors to quantize")
    if check_n_colors(n_colors):
        if mode == GRAYSCALE_MODE:
            new_image = quantize(image, int(n_colors))
            return new_image
        else:
            separated_image = separate_channels(image)
            red_channel = separated_image[RED_CH_LOC]
            green_channel = separated_image[GREEN_CH_LOC]
            blue_channel = separated_image[BLUE_CH_LOC]
            new_red_im = quantize(red_channel, int(n_colors))
            new_green_im = quantize(green_channel, int(n_colors))
            new_blue_im = quantize(blue_channel, int(n_colors))
            new_image = combine_channels(
                [new_red_im, new_green_im, new_blue_im])
            return new_image
    else:
        print("N colors number is invalid")


def apply_blur(image: Image) -> Image:
    """This function receive an image and apply blur on image. If its RGB every
    channel is applied separately and then combined"""
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    if check_grayscale(image):
        mode = GRAYSCALE_MODE
    else:
        mode = RGB_MODE
    kernel_size: str = input("Insert kernel size for blur")
    if check_kernel_size(kernel_size):  # check odd integer
        kernel: List[List[float]] = blur_kernel(int(kernel_size))
        if mode == RGB_MODE:
            separated_image = separate_channels(image)
            channels_after_action = []
            for channel in separated_image:
                channels_after_action.append(apply_kernel(channel, kernel))
            new_image: ColoredImage = combine_channels(channels_after_action)
            return new_image
        else:
            new_image: SingleChannelImage = apply_kernel(image, kernel)
            return new_image
    else:
        print("Kernel size is invalid. Try again")


def apply_rotate(image: Image) -> Image:
    """This function receive an image and apply rotate_90() on image. If its
    RGB apply func on each channel separately. Func return new image"""
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    if check_grayscale(image):
        mode = GRAYSCALE_MODE
    else:
        mode = RGB_MODE
    direction: str = input("Insert rotation direction L / R")
    if direction == "L" or direction == "R":
        if mode == RGB_MODE:
            separated_image = separate_channels(image)
            channels_after_action = []
            for channel in separated_image:
                channels_after_action.append(rotate_90(channel, direction))
            new_image: ColoredImage = combine_channels(channels_after_action)
            return new_image
        else:
            new_image: SingleChannelImage = rotate_90(image, direction)
            return new_image
    else:
        print("Direction inserted is invalid")


def apply_get_edges(image) -> Image:
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    parms: str = input(
        "Insert parameters for edges <blur_size>,<block_size>,<c>")
    if check_grayscale(image):
        mode = GRAYSCALE_MODE
    else:
        mode = RGB_MODE
    try:
        blur_size, block_size, c = parms.split(",")
        if check_edges_parms(blur_size, block_size, c):
            if mode == GRAYSCALE_MODE:
                new_image: SingleChannelImage = get_edges(image,
                                                          int(blur_size),
                                                          int(block_size),
                                                          float(c))
                return new_image
            else:
                # if image is RGB it needs to be converted to grayscale first
                grayscale_im: SingleChannelImage = RGB2grayscale(image)
                new_image: SingleChannelImage = get_edges(grayscale_im,
                                                          int(blur_size),
                                                          int(block_size),
                                                          float(c))
            return new_image
        else:
            print("parameters values are invalid. Try again")
    except:
        print(
            "Invalid input format. Please enter three values separated by commas.")


def apply_resize(image: Image) -> Image:
    """This function receive an image and apply resize on image, func return
    a new resized image"""
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    sizes: str = input("Insert resize <height,width>")
    try:
        height, width = sizes.split(",")
        if check_sizes(height, width):
            if check_grayscale(image):
                mode = GRAYSCALE_MODE
            else:
                mode = RGB_MODE
            if mode == RGB_MODE:
                separated_image = separate_channels(image)
                channels_after_action = []
                for channel in separated_image:
                    channels_after_action.append(
                        resize(channel, int(height), int(width)))
                new_image: ColoredImage = combine_channels(
                    channels_after_action)
                return new_image
            else:
                new_image: SingleChannelImage = resize(image, int(height),
                                                       int(width))
                return new_image
        else:
            print("Sizes for resize are invalid. Try again")
    except:
        print("Sizes for resize are invalid. Try again")


def check_valid_action(user_action: str) -> bool:
    """This function receive user_action and return True if it's 1-8 or False
    otherwise"""
    VALID_ACTIONS = ['1', '2', '3', '4', '5', '6', '7', '8']
    if user_action in VALID_ACTIONS:
        return True
    else:
        return False


def print_menu() -> None:
    """This func print user action menu"""
    print("IMAGE EDITOR ACTION MENU:")
    print("RGB2GRAY = 1")
    print("BLUR = 2")
    print("RESIZE = 3")
    print("ROTATE = 4")
    print("EDGES = 5")
    print("QUANTIZE = 6")
    print("SHOW IMAGE = 7")
    print("EXIT = 8")


def commit_actions(image: Image, mode: str) -> None:
    """This function commit user's requested actions on image until user
    request to exit. Function return result image after all user actions"""
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    # COMMAND NUMBERS:
    BLUR = "2"
    RESIZE = "3"
    ROTATE = "4"
    EXIT = "8"
    SHOW_IM = "7"
    QUANTIZE = "6"
    EDGES = "5"
    RGB2GRAY = "1"
    new_image: Image = deepcopy(image)
    user_action: str = "0"
    print_menu()
    print(f"Image is loaded, mode: {mode}")
    while user_action != EXIT:
        user_action = input("Please choose action to commit on image from 1-8")
        if check_valid_action(user_action):
            print(f"Editor is commiting action: {user_action}")
            if user_action == SHOW_IM:
                show_image(new_image)
            elif user_action == RGB2GRAY:
                if mode == GRAYSCALE_MODE:
                    #  no need to apply grayscale on grayscale image
                    new_image = new_image
                else:
                    new_image = RGB2grayscale(new_image)
            elif user_action == EDGES:
                edged_im = apply_get_edges(new_image)
                if edged_im is not None:
                    new_image = edged_im
            elif user_action == QUANTIZE:
                new_image = apply_quantize(new_image)
            elif user_action == BLUR:
                blured_image = apply_blur(new_image)
                if blured_image is not None:
                    new_image = blured_image
            elif user_action == RESIZE:
                resized_image = apply_resize(new_image)
                if resized_image is not None:
                    new_image = resized_image
            elif user_action == ROTATE:
                rotated = apply_rotate(new_image)
                if rotated is not None:
                    new_image = rotated
        else:
            print("This command is invalid. Try again")
    save_path: str = input("Enter path to save result image")
    try:
        save_image(new_image, save_path)
    except ValueError:
        print("Save path is invalid")


def main():
    RGB_MODE = "RGB"
    GRAYSCALE_MODE = "L"
    image_path = get_image_path()
    if image_path is not None:
        if check_valid_path(image_path):
            try:
                image = load_image(image_path, RGB_MODE)
                if check_grayscale(image):
                    grayscale_im = load_image(image_path, GRAYSCALE_MODE)
                    commit_actions(grayscale_im, GRAYSCALE_MODE)
                else:
                    color_im = image
                    commit_actions(color_im, RGB_MODE)
            except PIL.UnidentifiedImageError:
                print("Image type is invalid")
        else:
            print("Image path is invalid")


if __name__ == '__main__':
    main()
