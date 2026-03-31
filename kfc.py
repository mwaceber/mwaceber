import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.kfc.co.tz/en/menu#section_id=&section_name=Browse%20Categories"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')

menu = soup.find_all('h3', class_ = 'MuiTypography-root MenuItem__StyledItemTitle-sc-pcesxc-6 fuBzVf MuiTypography-h5')
menu_items = []
for i in menu:
    food = i.text
    menu_items.append(food)


price = soup.find_all('span')
price = [p for p in price if not p.get('class') or 'MuiTab-wrapper' not in p.get('class', [])]
menu_prices = [p.text for p in price if p.text.strip()][1:]

kfc_menu = pd.DataFrame({'Food Item': menu_items, 'Price (TZS)': menu_prices})
kfc_menu.to_csv('KFC Tanzania Menu.csv')