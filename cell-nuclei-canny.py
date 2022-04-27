import numpy as np
import matplotlib.pyplot as plt
import imageio
import math
import time
# Fast canny's algorithm for edge detection of cell nuclei

# a)

start = time.time()

def pad(f, M, N, R, P):
    r2 = R//2
    p2 = P//2

    if M%2 != 0 and N%2 != 0:
        f_pad = np.zeros((M+r2*2+1, N+p2*2+1), dtype="float")
    else:
        f_pad = np.zeros((M+r2*2, N+p2*2), dtype="float")

    # apenbart
    f_pad[:r2, :p2] = f[0,0]
    f_pad[:r2, p2:N+p2] = f[0]
    f_pad[:r2, N+p2:] = f[0, N-1]

    f_pad[r2:M+r2, :p2] = f[:, :1]
    f_pad[r2:M+r2, p2:N+p2] = f[:,]
    f_pad[r2:M+r2, N+p2:] = f[:, N-1:N]

    f_pad[M+r2:, :p2] = f[M-1, 0]
    f_pad[M+r2:, p2:N+p2] = f[M-1]
    f_pad[M+r2:, N+p2:] = f[M-1, N-1]

    return f_pad

def konv(kernel, f):
    M, N = f.shape
    R, P = kernel.shape

    # output
    g = np.zeros((M,N), dtype="float")

    # sjekker om fft hadde vaert fortere
    if ((R+P)/2)**2 > np.log2((M+N)/2):
        kernel_pad = pad(kernel, R, P, M-R, N-P)

        f_freq = np.fft.fft2(f)
        kernel_freq = np.fft.fft2(kernel_pad)
        g_freq = f_freq * kernel_freq
        g = np.fft.ifftshift(np.real(np.fft.ifft2(g_freq)))

        return g

    else:
        # padding
        f_pad = pad(f, M, N, R, P)
        # sjekker om kernel er separerbar
        if np.linalg.matrix_rank(kernel) == 1:
            v = (kernel[:, :1]/kernel[0,0])[::-1]
            w = np.array([(kernel[0])[::-1]]).T

            r2 = R//2
            p2 = P//2
            # utregning for separerbar
            for i in range(M):
                for j in range(N):
                    m = f_pad[i:i+R, j:j+P]
                    g[i, j] = (m@w).T@v
            return g

        else:
            # utregning for useparerbar kernel
            for i in range(M):
                for j in range(N):
                    g[i,j] = np.sum(f_pad[i:i+R, j:j+P]*kernel)
            return g

# b)

# hjelpemetode som oppretter gaussfilter
def gauss(sigma):
    n = 1+8*math.ceil(sigma)
    v = np.ones((n, 1), dtype="float")

    for i in range(n):
        v[i] *= math.exp(-((i-n//2)**2)/(2*sigma**2))
    v = v/sum(v)

    return v@v.T

# hjelpemetode som finner naermeste vinkelkategori
def closestDir(a, d):
    if a < 0:
        a += np.pi
    dist = abs(d[0]-a)
    closest = d[0]
    for i in d:
        if abs(i-a) < dist:
            dist = abs(i-a)
            closest = i
    return closest

# hjelpemetode hysteresis threshold
def threshold(i, j, gnl, g_out, M, N):
    for r in range(-1,2):
        for p in range(-1,2):
            if i+r < M and j+p < N:
                if gnl[i+r, j+p] != 0 and g_out[i+r, j+p] != 255:
                    g_out[i+r, j+p] = 255
                    g_out = threshold(i, j, gnl, g_out, M, N)

    return g_out

def canny(g, sigma, lth, hth):
    # allerede smoothed med gauss

    # 1D operatorer
    v = np.array([[1, 0, -1]])
    w = np.matrix.transpose(np.array([[1, 0, -1]]))
    gx = konv(w, g)
    gy = konv(v, g)

    # magnitud og vinkel matrix
    mag = abs(gx)+abs(gy)
    alfa = np.arctan2(gx, gy)

    M, N = mag.shape
    gn = np.zeros((M,N), dtype="float")
    mag = pad(mag, M, N, 3, 3)

    # mulige vinkelkategorier
    d = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

    # maxima supression
    for i in range(M):
        for j in range(N):
            alfa_xy = closestDir(alfa[i,j], d)
            if mag[i+round(math.sin(alfa_xy))+1, j+round(math.cos(alfa_xy))+1] <= mag[i+1,j+1] and mag[i-round(math.sin(alfa_xy))+1, j-round(math.cos(alfa_xy))+1] <= mag[i+1,j+1]:
                gn[i,j] = mag[i+1,j+1]

    # hysteresis threshold
    tl = max(map(max, gn))*lth
    th = max(map(max, gn))*hth

    gnh = (gn >= th).astype("int")*255
    gnl = (gn >= tl).astype("int")*255

    g_out = np.zeros((M, N), dtype="float")
    g_out[:,] = gnh[:,]

    for i in range(1, M-1):
        for j in range(1, N-1):
            if int(gnh[i,j]) != 0:
                g_out = threshold(i, j, gnl, g_out, M, N)

    return g_out

f = imageio.imread("cellekjerner.png", as_gray=True)

gaussian = gauss(sigma=6)
gaussian_blur = konv(gaussian, f)
g_out = canny(gaussian_blur, sigma=6, lth=0.12, hth=0.27)

plt.subplot(121)
plt.imshow(f, cmap="gray")
plt.title("original image")
plt.subplot(122)
plt.imshow(g_out, cmap="gray", vmax=100)
plt.title("detected edges of cell nuclei")
print(time.time()-start)

plt.show()
