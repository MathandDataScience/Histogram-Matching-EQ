import numpy as np
import cv2
import matplotlib.pyplot as plt


# ordering based on k value

def get_avgs(img):
    kernel1 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]) * 1 / 5

    kernel2 = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) * 1 / 9

    kernel3 = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0]]) * 1 / 13

    kernel4 = np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0]]) * 1 / 21

    kernel5 = np.array([[1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]]) * 1 / 25

    avg1 = cv2.filter2D(img, -1, kernel1)
    avg2 = cv2.filter2D(img, -1, kernel2)
    avg3 = cv2.filter2D(img, -1, kernel3)
    avg4 = cv2.filter2D(img, -1, kernel4)
    avg5 = cv2.filter2D(img, -1, kernel5)

    b = [avg1, avg2, avg3, avg4, avg5]
    b = np.asarray(b)

    return b


def col_order_val(img1):
    w = img1.shape[0]
    h = img1.shape[1]
    c = np.zeros((w * h,))
    for i in range(len(img1)):
        for k in range(len(img1[1])):
            c[h * i + k] = img1[i, k]

    return c


def Q_matrix(img, k):
    ordering = [col_order_val(img)]
    if k > 0:
        for f in range(k):
            ordering.append(col_order_val(get_avgs(img)[f]))
        return np.asarray(ordering)
    if k == 0:
        return np.asarray([ordering])


def lex_map(img4, k):
    Q = Q_matrix(img4, k)
    if k == 0:
        lexor = np.lexsort((Q[0]))
        return lexor
    if k == 1:
        lexor = np.lexsort((Q[1], Q[0]))
        """lexor = np.flipud(lexor)
        orQ = Q_2[lexor]
        return np.flipud(np.rot90(orQ))"""
        return lexor
    if k == 2:
        lexor = np.lexsort((Q[2], Q[1], Q[0]))
        """#lexor = np.flipud(lexor)
        orQ = Q_2[lexor]
        return np.flipud(np.rot90(orQ))"""
        return lexor
    if k == 3:
        lexor = np.lexsort((Q[2], Q[1], Q[0]))
        """#lexor = np.flipud(lexor)
        orQ = Q_2[lexor]
        return np.flipud(np.rot90(orQ))"""
        return lexor
    if k == 4:
        lexor = np.lexsort((Q[3], Q[2], Q[1], Q[0]))
        """#lexor = np.flipud(lexor)
        orQ = Q_2[lexor]
        return np.flipud(np.rot90(orQ))"""
        return lexor
    if k == 5:
        lexor = np.lexsort((Q[4], Q[3], Q[2], Q[1], Q[0]))
        return lexor


def true_HIS_EQ(img, k):
    flatsize = img.shape[0] * img.shape[1]
    index = lex_map(img, k)
    inten = np.uint8(np.linspace(0, 256, flatsize, endpoint=False))# histogram
    I_flat = col_order_val(img)  # copy of flat
    I_flat[index] = inten
    W = img.shape[0]
    H = img.shape[1]
    U_flat = np.uint8(I_flat.reshape((W, H)))
    return U_flat



def HIS_EQ_match(img, ref, k):
    if img.shape[0] == ref.shape[0]:
        if img.shape[1] == ref.shape[1]:
            index = lex_map(img, k)
            index2 = lex_map(ref, k)
            I_flat = col_order_val(img)  # copy of flat
            I_flat2 = col_order_val(ref)
            I_flat[index] = I_flat2[index2]
            W = img.shape[0]
            H = img.shape[1]
            U_flat = np.uint8(I_flat.reshape((W, H)))
            return U_flat

    if img.shape[0] != ref.shape[0]:
        print("ref img wrong size")
    if img.shape[1] != ref.shape[1]:
        print("ref img wrong size")


img = cv2.imread("peppers_color.tif", cv2.IMREAD_GRAYSCALE)
img_2 = cv2.imread("cameraman.tif", cv2.IMREAD_GRAYSCALE) # ref image

plt.hist(img.ravel(), 256, (0, 256)) # original hist
plt.show()
plt.hist(img_2.ravel(), 256, (0, 256)) # ref hist
plt.show()


# image have to be the same size
img2 = HIS_EQ_match(img,img_2, 0)
cv2.imshow('de', img2)


plt.hist(img2.ravel(), 256, (0, 256)) # new hist
plt.show()


img = cv2.imread("peppers_color.tif")
k = 0

"""HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)"""

out_img = np.zeros((512, 512, 3), dtype=np.uint8)
out_img[:, :, 0] = true_HIS_EQ(img[:, :, 0],k)
out_img[:, :, 1] = true_HIS_EQ(img[:, :, 1],k)
out_img[:, :, 2] = true_HIS_EQ(img[:, :, 2],k)
cv2.imshow("EQ_BGR", out_img)

"""HSL_img = np.zeros((512, 512, 3), dtype=np.uint8)
HSL_img[:, :, 0] = true_HIS_EQ(HLS[:, :, 0],k)
HSL_img[:, :, 1] = true_HIS_EQ(HLS[:, :, 1],k)
HSL_img[:, :, 2] = true_HIS_EQ(HLS[:, :, 2],k)
HSL_img = cv2.cvtColor(HSL_img, cv2.COLOR_HLS2BGR)
cv2.imshow("EQ_HSL", HSL_img)

HSV_img = np.zeros((512, 512, 3), dtype=np.uint8)
HSV_img[:, :, 0] = true_HIS_EQ(HSV[:, :, 0],k)
HSV_img[:, :, 1] = true_HIS_EQ(HSV[:, :, 1],k)
HSV_img[:, :, 2] = true_HIS_EQ(HSV[:, :, 2],k)
HSV_img= cv2.cvtColor(HSV_img, cv2.COLOR_HSV2BGR)
cv2.imshow("EQ_HSV", HSV_img)
"""

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.hist(out_img.ravel(), 256, (0, 256))
plt.show()

"""flatsize = img.shape[0]*img.shape[1]
index = lex_map(img,0)
print(index)
inten = np.uint8(np.linspace(0 ,256, flatsize ,  endpoint=False))# histogram

I_flat = col_order_val(img) # copy of flat
I_flat[index] = inten
print(I_flat)
W=img.shape[0]
H=img.shape[1]
I_flat.shape = W , H
print(I_flat)
U_flat = np.uint8(I_flat.reshape((W,H)))
print(U_flat)"""
