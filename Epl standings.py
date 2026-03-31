import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://www.skysports.com/serie-a-table"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
table = soup.find('table', class_ = 'sdc-site-table')
#print(table)

headers = table.find_all('th')
headers_list = []
for i in headers:
    header = i.text.strip()
    headers_list.append(header)
#print(headers_list)

df = pd.DataFrame(columns=headers_list)

rows = table.find_all('tr')[1:]
for i in rows:
    first_team = i.find_all('td')[0].find('span').text.strip()
    teams = i.find_all('td')[1:]
    more_data = [tr.text for tr in teams]
print(more_data)