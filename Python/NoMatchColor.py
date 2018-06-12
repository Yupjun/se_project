import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

################################
################################
# Class명    : GatherColorInformation
# 작성자    : 이현지
# 설명      : user가 입력한 이미지를 분석하여 색 띠 반환
# 참고한 코드 출처 : https://buzzrobot.com/dominant-colors-in-an-image-using-k-means-clustering-3c7af4622036
################################
################################
class GatherColorInformation:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    #kmeans clustering 알고리즘을 이용하여 각 픽셀 처리
    #cluster값은 5로 제시한다.
    def dominantColors(self):
        img = cv2.imread(self.IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1]), 3)
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        return kmeans

    # clusters=5에 따라 5개의 색상(r,g,b)을 return 하게 된다.
    def getRGB(self, kmeans):
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

        return self.COLORS.astype(int)

    #히스토그램으로 표현하기
    def intoHistogram(self, kmeans):
        numLabels = np.arange(0, self.CLUSTERS+1)
        (hist, _)=np.histogram(self.LABELS, bins=numLabels)

        hist=hist.astype("float")
        hist /= hist.sum()

        return hist

    #색상이 차지하는 비율 순으로 정렬한 후, 이를 바탕으로 색 띠를 생성하여 리턴한다
    def plot_colors(self, hist, centroids):
        colors=self.COLORS
        colors=colors[(-hist).argsort()]
        hist=hist[(-hist).argsort()]

        bar=np.zeros((50,500,3), dtype="uint8")
        startX=0

        for i in range(0, self.CLUSTERS-1):
            endX= startX+hist[i]*500

            r=int(colors[i][0])
            g=int(colors[i][1])
            b=int(colors[i][2])

            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), (r,g,b), -1)
            startX=endX

        return bar

################################
################################
# Class명    : Compare
# 작성자    : 이현지
# 설명      : 두 이미지를 분석한 색상 결과를 비교
################################
################################
class Compare:

    colorA=None
    ColorB=None

    def __init__(self, colors):
        self.colorA=colors[0]
        self.colorB=colors[1]

    #두 이미지를 분석한 결과로 나온 dominant colors들을 비교
    #색상이 일치하지 않으면서, 이미지 내에서 차지하는 비율이 큰 색상을 no match color로 선정한다.
    def getNoMatchColor(self):
        noMatches = []
        for color in self.colorA:
            if color not in self.colorB:
                noMatches.append(color)

        noMatchColor = noMatches[0]

        return noMatchColor


################################
################################
# Class명    : ImageProducing
# 작성자    : 이현지
# 설명      : no match color 영역을 이미지에서 찾아 표시한다
# 참고한 코드 출처 : https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
################################
################################
class ImageProducing:

    IMAGE = None

    def __init__(self, image):
        self.IMAGE = cv2.imread(image)

    def markNoMatchColor(self,color):

        hsv = cv2.cvtColor(self.IMAGE, cv2.COLOR_BGR2HSV)

        #rgb순으로 표현되어있는 color를 bgr형태로 변환
        [b,g,r]=[color[2],color[1],color[0]]
        bgrColor=np.uint8([[[b,g,r]]])

        #색상 추출을 위해 bgr을 hsv로 변환
        hsv_noMatch=cv2.cvtColor(bgrColor, cv2.COLOR_BGR2HSV)

        #색상의 min hsv, max hsv 범위 추출
        min_h=hsv_noMatch[0][0][0]-10
        min_noMatch=np.array([min_h, 100, 100])
        max_h=hsv_noMatch[0][0][0]+10
        max_noMatch=np.array([max_h, 255, 255])
        kernel = np.ones((5, 5), "uint8")

        noMatch=cv2.inRange(hsv, min_noMatch, max_noMatch)
        noMatch=cv2.dilate(noMatch, kernel)

        #no match color 영역 표시하기
        (_, contours, hierarchy) = cv2.findContours(noMatch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 150):
                ellipse=cv2.fitEllipse(contour)
                cv2.ellipse(self.IMAGE, ellipse, (0,0,255),1, cv2.LINE_AA)

        cv2.imshow('marked image', self.IMAGE)


user='../images/NoMatch_testImage/room2.jpg'
ideal='../images/NoMatch_testImage/room6.jpg'
testimage='testimage.jpg'
images=[user, ideal]
colors=[]
bars=[]
clusters=5

for i in range(len(images)):
    colorInfo=GatherColorInformation(images[i], clusters)
    kmeans=colorInfo.dominantColors()
    color=colorInfo.getRGB(kmeans)
    colors.append(color)
    hist=colorInfo.intoHistogram(kmeans)
    bar=colorInfo.plot_colors(hist, kmeans.cluster_centers_)
    bars.append(bar)


cmp=Compare(colors)
noMatch=cmp.getNoMatchColor()

#이미지 display
userRoom=cv2.imread('../images/NoMatch_testImage/room2.jpg')
idealRoom=cv2.imread('../images/NoMatch_testImage/room6.jpg')

cv2.imshow('user room', userRoom)
if(len(images)==2):
    cv2.imshow('ideal room', idealRoom)

for i in range(len(images)):
    plt.figure()
    plt.axis("off")
    plt.imshow(bars[i])
plt.show()

mark=ImageProducing(images[0])
mark.markNoMatchColor(noMatch)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()



