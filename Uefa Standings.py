import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.bbc.com/sport/football/champions-league/table"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
table = soup.find('table', class_ = 'ssrcss-1faa8h0-Table e13j9mpy3')
#print(table)

headers = table.find_all('th')
headers_list = []
for i in headers:
    header = i.text
    headers_list.append(header)
    
df = pd.DataFrame(columns= headers_list)
#print(df)

rows = table.find_all('tr')[1:]
for i in rows:
    row = i.find_all('td')
    standing = [tr.text for tr in row]
    l = len(df)
    df.loc[l] = standing

#print(df)

df.to_csv("Uefa Standings.csv")