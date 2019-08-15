from mtcnn.mtcnn import MTCNN
import cv2
import os
import time

if __name__ == '__main__':
    
    detector = MTCNN()

    # 待识别的图片路径
    path = r'F:\faceImages'
    names = os.listdir(path)
    for name in names:
        # 建立文件夹以保存提取结果
        result_path = 'F:/faceImageGray/' + name
        folder = os.path.exists(result_path)
        if not folder:
            os.makedirs(result_path)

        img_names = os.listdir(os.path.join(path, name))
        for i, img_name in enumerate(img_names):
            img = cv2.imread(os.path.join(path, name, img_name))
            # run detector
            results = detector.detect_faces(img)

            if results is not None:
                for result in results:
                    x1, y1, w, h = result['box']
                    new_img = img[y1:y1+h,x1:x1+w]
                    img_gray = cv2.cvtColor(new_img,cv2.COLOR_RGB2GRAY)
                    cv2.imwrite(result_path + '/' + str(i)  + '.jpg', img_gray)