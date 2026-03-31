import requests
from bs4 import BeautifulSoup
import re


url = "https://www.flashscore.com/news/"

r = requests.get(url)
# print(r.status_code)
# print(r.text)

soup = BeautifulSoup(r.text, 'lxml')
#print(soup.prettify())
# Now we look upon the html tags to find the data we want
# soup.find_all('div') . Example of finding all div tags
#tag = soup.find('div') 
# print(tag.attrs) # Example of accessing attributes of a tag
# tag = soup.find('h1')
# print(tag.string) # Example of accessing the string inside a tag
news = soup.find_all("div",class_= "section__mainTitle")

romano = soup.find(string= re.compile("Hugo Ekitike"))
print(romano)