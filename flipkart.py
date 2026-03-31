import requests
from bs4 import BeautifulSoup
import pandas as pd

Names = []
Prices = []
Desc = []
Discounts = []

for i in range(1,4): 
    url = "https://www.flipkart.com/search?q=mobile+5g+under+10000&as=on&as-show=on&otracker=AS_Query_OrganicAutoSuggest_4_7_na_na_na&otracker1=AS_Query_OrganicAutoSuggest_4_7_na_na_na&as-pos=4&as-type=RECENT&suggestionId=mobile+5g+under+10000&requestId=3b59dfad-79b9-4be7-859b-78c95eac2507&as-searchtext=mobilr+&page="+str(i)
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')
    box = soup.find("div", class_ = "QSCKDh dLgFEE")
    name = box.find_all('div', class_ = 'RG5Slk')
    for i in name:
        n = i.text
        Names.append(n)
    price = box.find_all('div', class_ = 'hZ3P6w DeU9vF')
    for i in price:
        p = i.text
        Prices.append(p)
    disc = box.find_all('div', class_ = 'HQe8jr')
    for i in disc:
        di = i.text
        Discounts.append(di)
    data = box.find_all('ul', class_ = 'HwRTzP')
    for j in data:
        md = j.text
        Desc.append(md)

df = pd.DataFrame({"Product name":Names, "Product price":Prices, "Discount given":Discounts, "Descriptions":Desc})
print(df)    

#next_page = soup.find('a', class_ = 'jgg0SZ').get('href')
#print(next_page) #If it didn't provide the second page, then
#comp_next_page = "https://www.flipkart.com/"+next_page
#print(comp_next_page) 

#url = comp_next_page
#r = requests.get(url)
#soup = BeautifulSoup(r.text, 'lxml')