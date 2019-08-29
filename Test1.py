import cv2
import pytesseract
from PIL import Image
from PIL import Image
image="ssss"
def change_size(read_file):
    image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)  # 改为单通道

    x = binary_image.shape[0]
    print("高度x=", x)
    y = binary_image.shape[1]
    print("宽度y=", y)
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)  # 左边界
    right = max(edges_x)  # 右边界
    width = right - left  # 宽度
    bottom = min(edges_y)  # 底部
    top = max(edges_y)  # 顶部
    height = top - bottom  # 高度

    pre1_picture = image[left:left + width, bottom:bottom + height]  # 图片截取
    return pre1_picture  # 返回图片数据


# 透视矫正
def perspective_transformation(img):
    # 读取图像，做灰度化、高斯模糊、膨胀、Canny边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    dilate = cv2.dilate(blurred, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    # edged = cv2.Canny(dilate, 75, 200)
    edged = cv2.Canny(dilate, 30, 120, 3)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # // cnts = cnts[0] if imutils.is_cv2() else cnts[1]  # 判断是OpenCV2还是OpenCV3
    docCnt = None

    # 确保至少找到一个轮廓
    if len(cnts) > 0:
        # 按轮廓大小降序排列
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # 近似轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果我们的近似轮廓有四个点，则确定找到了纸
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt
text = pytesseract.image_to_string(Image.open(image), lang='chi_sim')
# text = pytesseract.image_to_string(Image.open(image), lang='eng')
print(text)
# 对原始图像应用四点透视变换，以获得纸张的俯视图
#     paper = four_point_transform(img, docCnt.reshape(4, 2))
# img=change_size("./eng.jpg")
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = Image.open("./eng.jpg")
# print(img.size)
# cropped = img.crop((0, 0, 512, 128))  # (left, upper, right, lower)
# cropped.save("./pil_cut_thor.jpg")
#
#
# # Sobel边缘检测算子
# img = cv2.imread('./image2.jpg', 0)
# # gray = cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
# #
# # gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# # gradY = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
#
# # # subtract the y-gradient from the x-gradient
# # gradient = cv2.subtract(gradX, gradY)
# # gradient = cv2.convertScaleAbs(gradient)
# # blurred = cv2.blur(gradient, (9, 9))
# # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
# # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# # closed = cv2.erode(closed, None, iterations=4)
# # closed = cv2.dilate(closed, None, iterations=4)
# # cv2.imshow('img', img)
# x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
# y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
#
# gradient = cv2.subtract(x, y)
# gradient = cv2.convertScaleAbs(gradient)
#
# blurred = cv2.blur(gradient, (9, 9))
# (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
#
# # cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])
# # 可选参数alpha是伸缩系数，beta是加到结果上的一个值，结果返回uint类型的图像
# # Scale_absX = cv2.convertScaleAbs(x)  # convert 转换  scale 缩放
# # Scale_absY = cv2.convertScaleAbs(y)
# # result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
#
# cv2.imshow('img', img)
# # cv2.imshow('Scale_absX', Scale_absX)
# # cv2.imshow('Scale_absY', Scale_absY)
# # cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# text = pytesseract.image_to_string(Image.open('./WechatIMG1.jpeg'), lang='eng')  # 读取片及使用的文字识别类型
# print(text);
#
# def rotate_bound(image, angle):
#     # grab the dimensions of the image and then determine the
#     # center
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # grab the rotation matrix (applying the negative of the
#     # angle to rotate clockwise), then grab the sine and cosine
#     # (i.e., the rotation components of the matrix)
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # compute the new bounding dimensions of the image
#     nW = int((h * sin) + (w * cos))
#     nH = int((h * cos) + (w * sin))
#
#     # adjust the rotation matrix to take into account translation
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     # perform the actual rotation and return the image
#     return cv2.warpAffine(image, M, (nW, nH))
#
#
# ## 图片旋转
# def rotate_bound(image, angle):
#     # 获取宽高
#     (h, w) = image.shape[:2]
#     (cX, cY) = (w // 2, h // 2)
#
#     # 提取旋转矩阵 sin cos
#     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
#     cos = np.abs(M[0, 0])
#     sin = np.abs(M[0, 1])
#
#     # 计算图像的新边界尺寸
#     nW = int((h * sin) + (w * cos))
#     #     nH = int((h * cos) + (w * sin))
#     nH = h
#
#     # 调整旋转矩阵
#     M[0, 2] += (nW / 2) - cX
#     M[1, 2] += (nH / 2) - cY
#
#     return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#
# ## 获取图片旋转角度
# def get_minAreaRect(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.bitwise_not(gray)
#     thresh = cv2.threshold(gray, 0, 255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     coords = np.column_stack(np.where(thresh > 0))
#     return cv2.minAreaRect(coords)
#
#
# image_path = "eng.jpg"
# image = cv2.imread(image_path)
# angle = get_minAreaRect(image)[-1]
# rotated = rotate_bound(image, angle)
#
# cv2.putText(rotated, "angle: {:.2f} ".format(angle),
#             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
# # show the output image
# print("[INFO] angle: {:.3f}".format(angle))
# cv2.imshow("iuput", image)
# # cv2.imshow("output", rotated)
# # cv2.waitKey(0)
#
#
#
# # print("sss")
# # image=cv2.imread('eng.jpg')
# # angle=90

# imag=rotate_bound(rotated,90)
# cv2.imshow('ww',imag)
# cv2.waitKey()


# image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('origin', image)
#
# h, w = image.shape  # 获取图像的高度和宽度
#
# # Sobel 滤波器 进行边的检测
# sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # 水平方向
# sobel_vetical = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # 垂直方向
# cv2.imshow('sobel_H', sobel_horizontal)  # 水平方向
# cv2.imshow('sobel_V', sobel_vetical)  # 垂直方向
#
# # 拉普拉斯算子 进行边的检测    64F代表每一个像素点元素占64位浮点数
# laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
# cv2.imshow('laplacian', laplacian)
#
# # # Canny边检测器
# canny = cv2.Canny(image, 50, 240)
# cv2.imshow('Canny', canny)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
