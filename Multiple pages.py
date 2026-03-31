import requests
from bs4 import BeautifulSoup
#for i in range [1:11]:
url = "https://www.kikuu.co.tz/search/result?belongCategory=999577&kw=Phones"#+str(i)
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
#print(soup.prettify())

while True:
    nextp = soup.find("svg", class_ = "icon___39ZDk").get('href')
    comp_nextp = "https://www.kikuu.co.tz/search/result?belongCategory=999577&kw=Phones" + nextp
    print(comp_nextp)
# Obtaining url of the new page

url = comp_nextp
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
#Obtaining the html of the new page