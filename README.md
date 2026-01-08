# Python Algorithmic Image Editor

A command-line image processing tool built from scratch in Python.
This project implements core image manipulation algorithms manually (using matrix manipulation) rather than relying on high-level processing libraries. It demonstrates a deep understanding of digital image representation, convolution kernels, and interpolation.

## 🚀 Features

* **Pure Python Implementation:** Image processing logic is implemented directly on 2D and 3D arrays (Lists), providing a look "under the hood" of how images are manipulated.
* **CLI Interface:** Interactive shell-based menu for easy operation.
* **Format Support:** Supports loading and saving standard image formats (JPEG, PNG) via a lightweight PIL wrapper for I/O.
* **Robust Input Handling:** Validates file paths and user inputs to prevent crashes.

## 🧠 Algorithms & Implementation Details

The core of this project lies in the mathematical manipulation of pixel data. The image is treated as a matrix (grayscale) or a 3-dimensional tensor (RGB).

### 1. Gaussian Blur (Convolution)
Implemented a smoothing filter using **Convolution** with a kernel.
* **Logic:** The algorithm iterates over every pixel and computes the weighted average of its neighbors based on a specific kernel size.
* **Edge Handling:** Handles image boundaries strictly to ensure the kernel does not read out of bounds.

### 2. Image Resizing (Bilinear Interpolation)
Custom implementation of image scaling.
* **Logic:** Instead of simple nearest-neighbor sampling, the algorithm maps the coordinates of the *target* image back to the *source* image.
* **Handling:** Calculates the new pixel value based on the relative distance to the nearest original pixels, resulting in smoother resizing without blocky artifacts.

### 3. Edge Detection
Implements an edge detection filter (using a derivative-based kernel).
* **Logic:** Uses a specific kernel to calculate the gradient magnitude of the image, highlighting areas of high intensity change (edges).

### 4. Rotation
* **Matrix Rotation:** Implements 90-degree rotations by re-mapping the matrix indices `(i, j) -> (j, width-1-i)`.

### 5. Color Quantization
* Reduces the number of distinct colors in the image, useful for compression simulation or artistic effects.

### 6. Grayscale Conversion
* Converts RGB images to single-channel Grayscale using the luminance formula:
    $$Y = 0.299R + 0.587G + 0.114B$$

## 💻 Usage

To run the editor, execute the main script from your terminal:

```bash
python image_editor.py ```

1. Load Image
Upon running, the script will prompt you to enter the full path to an image file.

```Plaintext

Enter image path: /path/to/your/image.jpg
Input Validation: The program checks if the file exists and is a valid image format.```

2. Select Operations
Once loaded, you can perform multiple operations sequentially. Enter the number corresponding to the desired action:

```Plaintext

Select an operation:
1. Grayscale
2. Blur
3. Resize
4. Rotate Right
5. Rotate Left
6. Show Image
7. Edge Detection
8. Quantize
3. Save Result
After editing, the program allows you to save the modified image to a new path.```

##📂 Project Structure
image_editor.py: Contains the main application logic, menu loop, and algorithm implementations.

ex6_helper.py: A utility module handling the interface between raw Python lists and the filesystem (using Pillow for I/O only).

image_editor_test.py: Unit tests ensuring the correctness of matrix operations (e.g., verify rotation logic and channel combining).

##🛠️ Requirements
Python 3.x

Pillow (PIL) (Only used for loading/saving files, not for processing)

To install the dependencies:

Bash

pip install Pillow

##✍️ Author
Developed by Elal Gilboa as part of the Image Processing course at the Hebrew University.
