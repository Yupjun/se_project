from bs4 import BeautifulSoup
import urllib.request
from selenium import webdriver
import Image
from tesseract import image_to_string

######  가구 정보 가져올 함수 == >
#####   javaScript에서 동적 crawling 할 수 있게 변형해야 함


def get_inform(html,driver):

    soup = BeautifulSoup(html,"html.parser")

    for item in soup.find_all("p",class_="prd_img label_link"):
        print(item)

        ## p 태그 안에 있는 img 태그 찾기
        sub_tag = item.find("img")
        
        ## img 태그에서 src 추출 / http: 붙여서 완전한 주소 만듦
        img = sub_tag
        img_src = img.get("src")
        img_url ="http:"+img_src
        img_name = img_src.replace("/","")
        print(img_url)

        ## 가구 이미지 다운로드

        urllib.request.urlretrieve(img_url,"./img/" + img_name)

        print("이미지 src:", img_src)
        print("이미지 url:", img_url)
        print("이미지 명:", img_name)
        print("\n")


        ## 상세 정보
        detail = item.find("a")
        a = detail
        a_href = a.get("href")
        


        print("크롤링 종료")


def get_detail(URL):

    


        ## tag 타고 들어가서 나오는 상세 정보 창에서
        ## 상세 정보 이미지 다운로드


        ## 상세 정보 이미지 열어서 글자로 변환

   



def get_text(URL,tag, class_name , att):
    html = urllib.request.urlopen(URL)
    source = html.read()

    ## text return
    text= ''

    ## 몇개인지 세보려고 넣음
    count = 0

    ## 코드 html로 parsing
    soup = BeautifulSoup(source, "html.parser")


    ## 찾고 있는 class name 태그의 하위 태그에서 원하는 속성 값 추출 
    for item in soup.find_all(tag,class_=class_name):



        for item2 in item.find_all("li"):

            item3 = item2.find("a")
            item3_att = item3.get(att)

            text = text + item3_att + " "
            count += 1


    print(count)

    return text


def main():
        
    url_list = []
    url_list.append("http://mall.hanssem.com/main.html?gclid=Cj0KCQjwxtPYBRD6ARIsAKs1XJ6qyxMBunLaBs8LMtKN2usRiDY43TJJft_oFGwacixSRw1ddcFtSSAaAuJ9EALw_wcB")

##    url_text = get_text(url_list[0],"point", "href")

    url_text = get_text(url_list[0],"ul","sitemap_list", "href")


    ## 주소값 return 받은것 공백으로 다시 나눠서 list에 집어넣음
    menu_list = url_text.split(" ")


    ##selenium lib로 cmd 창 열여서 chrome 실행
    driver = webdriver.Chrome(executable_path=r"C:\Users\aaa\Desktop\prgram\chromedriver_win32\chromedriver.exe")
    

    for i in menu_list:
        print( "menu " )
        print()
        
        ## menu_list에 있는 url 에 chrome으로 접속-앞에서 chrome열어놓음
        driver.get(i)

        ## driver에서 page source를 얻어서 매개변수로 넘겨줌
        get_inform(driver.page_source,driver)

    driver.close()
    
main()
