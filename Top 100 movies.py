from bs4 import BeautifulSoup
import requests
import pandas as pd

url = "https://www.boxofficemojo.com/year/world/?ref_=bo_nb_hm_tab"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
tables = soup.find_all('table')

headers = tables[0].find_all('th')
header_list = []
for i in headers:
    header = i.text
    header_list.append(header)
df = pd.DataFrame(columns = header_list)

rows = tables[0].find_all('tr')[1:]
for i in rows:
    row = i.find_all('td')
    movie_list = [tr.text for tr in row]
    l = len(df)
    if len(movie_list) == len(header_list):
        df.loc[l] = movie_list

df.to_csv("Top 160 Movies.csv")