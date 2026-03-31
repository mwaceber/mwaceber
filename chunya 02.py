from bs4 import BeautifulSoup
import requests
import pandas as pd

for i in range (1,10):
    url  = f"https://oceantogames.com/page/{i}"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')

game = soup.find_all('h2', class_ = 'title')
games_list = []
for j in game:
    more = j.text.strip()
    games_list.append(more)

gen = soup.find_all('div',class_ = 'post-info')
genre_list = []
for k in gen:
    gen_more = k.text.strip()
    genre_list.append(gen_more)

ocean_of_games = pd.DataFrame({'Game Title': games_list, 'Genre': genre_list})
ocean_of_games.to_csv('Ocean of Games.csv') 