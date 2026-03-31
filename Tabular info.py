import requests
import pandas as pd
from bs4 import BeautifulSoup

url = "https://ligikuu.co.tz/nbc-premier-league/season-2025-2026/"
r = requests.get(url)
#print(r)

soup = BeautifulSoup(r.text, 'lxml')
#print(soup)

table = soup.find('table', class_ = 'sp-league-table sp-data-table sp-sortable-table sp-responsive-table table_69629884dea5c sp-scrollable-table')
#print(table.prettify())

headers = table.find_all('th')
header_list = []
for i in headers:
    header = i.text
    header_list.append(header)

df = pd.DataFrame(columns = header_list)
#print(df)

rows = table.find_all('tr')[1:] #Skip the header row
for i  in rows:
    row = i.find_all('td')
    standing = [tr.text for tr in row]
    l= len(df)
    df.loc[l]= standing
print(df)

df.to_csv("NBC Prem Standings.csv")
#print(standing)
#print(header_list)    