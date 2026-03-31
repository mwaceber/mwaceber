import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

url = "https://jiji.co.tz/lighting?query=neon+light"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'lxml')
#print(r.status_code)
#print(soup.prettify())

prices = soup.find_all("div", {"class": "qa-advert-price"})
#print(len(prices)) # Example of getting the number of items found
#print(prices[3]) # Example of accessing a specific item in the list
#
for price in prices:
    print(price.text) # Example of iterating through the list 
# and printing the text of each item

#for g in descriptions : 
    #print(g.text) #Example of iterating through the list

#inneed = soup.find_all(string= re.compile("neon"))
#print(inneed) # Example of searching for a specific string pattern

names = soup.find_all("div", {"class":"b-advert-title-inner qa-advert-title b-advert-title-inner--div"})
product_name = []

#for i in names:
#    name = i.text
#    product_name.append(name)

#print(product_name)    

product_price = []
for p  in prices:
    price = p.text
    product_price.append(price)

print(product_price) 

data = soup.find_all("div", {"class":"b-list-advert-base__description-text"})
more_data = []

#for d in data:
# #  info = d.text
# more_data.append(info)
#print(more_data)  

df = pd.DataFrame({"Product Name":product_name, "Product Price": product_price, "Description": more_data }) 
#print(df)    

#df.to_csv("jiji_neon_lights.csv") #Exporting a data frame to a csv file

