# Python-Canny-Edge-Detector
NumPy optimized edge detection algorithm for image processing, applied for detecting cell nuclei. Made from ground up with NumPy, with custom padding and convolve functions. Convolve uses either frequency domain multiplication or spatial domain convolve based on which would be faster. Optimized matrix multiplication by using linear dependance in kernel matrix to decompose to a pair of 1D vector multiplications, which saves amount of computations required. 

![cell_nuclei_fig](https://user-images.githubusercontent.com/103672622/165499004-da6fb8d9-c2e0-4677-b00c-96ff610c0136.png)
