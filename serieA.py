import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.livescore.com/en/football/england/premier-league/standings/"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
table = soup.find('table')
print(table) 