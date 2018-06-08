import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class GatherColorInformation:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):
        img = cv2.imread(self.IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1]), 3)
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        return kmeans

    def getRGB(self, kmeans):
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

        return self.COLORS.astype(int)

    def intoHistogram(self, kmeans):
        numLabels = np.arange(0, self.CLUSTERS+1)
        (hist, _)=np.histogram(self.LABELS, bins=numLabels)

        #normalize histogram
        hist=hist.astype("float")
        hist /= hist.sum()

        return hist

    def plot_colors(self, hist, centroids):
        colors=self.COLORS
        print("colors:", colors)
        print("hist:", hist)
        colors=colors[(-hist).argsort()]
        hist=hist[(-hist).argsort()]
        print("sort: ", colors)

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

class Compare:

    colorA=None
    ColorB=None

    def __init__(self, colors):
        self.colorA=colors[0]
        self.colorB=colors[1]

    def getNoMatchColor(self):
        noMatches = []
        for color in self.colorA:
            if color not in self.colorB:
                noMatches.append(color)

        noMatchColor = noMatches[0]

        return noMatchColor


#Class명: ImageProducing
#input: noMatchColor, user 방 이미지
#output: user 방 이미지에 차이나는 색상만 강조되어 보여지는 이미지
class ImageProducing:

    IMAGE = None

    def __init__(self, image):
        self.IMAGE = cv2.imread(image)

    def markNoMatchColor(self,color):

        hsv = cv2.cvtColor(self.IMAGE, cv2.COLOR_BGR2HSV)

        #rgb순으로 표현되어있는 color를 bgr형태로 변환
        [b,g,r]=[color[2],color[1],color[0]]
        bgrColor=np.uint8([[[b,g,r]]])
        print(bgrColor)

        #색상 추출을 위해 bgr을 hsv로 변환
        hsv_noMatch=cv2.cvtColor(bgrColor, cv2.COLOR_BGR2HSV)
        print("hsv:", hsv_noMatch)
        '''
        [r,g,b]=[color[0], color[1], color[2]]

        [h,s,v]=[0,0,0]
        v = max(r, g, b)
        if(v!=0):
            s=(v-min(r,g,b))
        else:
            s=0

        if(v==r):
            h=((g-b)*60)/s
        elif(v==g):
            h=120+((b-r)*60)/s
        else:
            h=240+((r-g)*60)/s

        if(h<0):
            h=h+360

        hsv_noMatch = np.uint8([[[h,s,v]]])

        '''
        '''
        alist = []
        aa = float(r / 255)
        ab = float(g / 255)
        ac = float(b / 255)
        alist.append(aa)
        alist.append(ab)
        alist.append(ac)
        alist.sort()
        aMax = alist[0]
        aMin = alist[2]
        aaaa = aMax - aMin
        h = int(60 * (((alist[1] - alist[0]) / aaaa) + 2))
        s = int(aaaa / aMax)
        v = int(aMax)
        if (g > r > b):
            h = int(60 * (((ac - aa) / aaaa) + 2))
            s = int(aaaa / aMax)
            v = int(aMax)
        '''

        print("noMatch:", hsv_noMatch)
        #색상의 min hsv, max hsv 범위 추출
        min_h=hsv_noMatch[0][0][0]-10
        min_noMatch=np.array([min_h, 100, 100])
        max_h=hsv_noMatch[0][0][0]+10
        max_noMatch=np.array([max_h, 255, 255])
        kernel = np.ones((5, 5), "uint8")

        noMatch=cv2.inRange(hsv, min_noMatch, max_noMatch)
        noMatch=cv2.dilate(noMatch, kernel)

        (_, contours, hierarchy) = cv2.findContours(noMatch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if (area > 150):
                #image = cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
                #x, y, w, h = cv2.boundingRect(contour)
                #image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                ellipse=cv2.fitEllipse(contour)
                cv2.ellipse(self.IMAGE, ellipse, (0,0,255),1, cv2.LINE_AA)

        cv2.imshow('marked image', self.IMAGE)


user='room2.jpg'
ideal='room6.jpg'
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

print(colors)
cmp=Compare(colors)
noMatch=cmp.getNoMatchColor()
print(noMatch)

#이미지 display
userRoom=cv2.imread('room2.jpg')
idealRoom=cv2.imread('room6.jpg')

cv2.imshow('user room', userRoom)
if(len(images)==2):
    cv2.imshow('ideal room', idealRoom)

for i in range(len(images)):
    plt.figure()
    plt.axis("off")
    plt.imshow(bars[i])
plt.show()


#noMatch 이미지에 표시하기
print("noMatch:", noMatch)
mark=ImageProducing(images[0])
mark.markNoMatchColor(noMatch)

key = cv2.waitKey(0)
if key == 27:
    cv2.destroyAllWindows()



