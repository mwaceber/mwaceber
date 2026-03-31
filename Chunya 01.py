import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

url = "https://www.flashscore.com/news/football/"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
#print(r.status_code)
#print(soup.prettify())

news = soup.find_all('div', class_ = 'wcl-news-heading-07_917dY wcl-headline_Q8-Zc')
headlines = []
for i in news:
    new = i.text
    headlines.append(new)

df = pd.DataFrame({"Headlines":headlines})
#print(df)

df.to_csv("Flashscore football news.csv")