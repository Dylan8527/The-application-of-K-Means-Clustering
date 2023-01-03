import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_path = "data/dsm.jpg"
# save_path = "data/dsm_kmeans.jpg"
img_path = "data/syj.jpg"
save_path = "data/syj_kmeans.jpg"


max_iter = 10
kmeans_num = 2

if __name__ == "__main__":
    ori_img = cv2.imread(img_path)
    print(ori_img.shape)
    # show img
    # plt.imshow(ori_img)
    # plt.show()

    img = ori_img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + \
        cv2.TERM_CRITERIA_MAX_ITER, max_iter, 1.0)

    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(img, kmeans_num, None, criteria, max_iter, flags)

    # We can change centers into colors~
    for i in range(len(centers)):
        # randomly generate a color
        centers[i] = np.random.rand(3) * 255
    print(centers)
    # centers = np.array([[0, 0, 255], [0, 255, 0]]) # for dhz
    centers = centers.astype(np.uint8)

    result = centers[labels.flatten()].reshape(ori_img.shape)


    # plt.imshow(result)
    # plt.show()

    cv2.imwrite(save_path, result)