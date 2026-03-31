import requests
from bs4 import BeautifulSoup

url = "https://jiji.co.tz/lighting?query=neon+light"

r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')

boxes = soup.find_all("div", class_ = "masonry-item")[4]
#print(boxes)

name = boxes.find("div", class_ = "b-advert-title-inner qa-advert-title b-advert-title-inner--div").text
print(name)

desc = boxes.find("div", class_ = "b-list-advert-base__description-text").text
print(desc)

price = boxes.find("div", class_ = "qa-advert-price").text
print(price) #Exracting any info either name,price or description from a single item
