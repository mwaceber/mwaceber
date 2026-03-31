from bs4 import BeautifulSoup

with open ('sample1.html', 'r') as html_file:
    content = html_file.read()
    
    soup = BeautifulSoup(content, 'lxml')
    print(soup.prettify())
    tags = soup.find_all('h5')
    print(tags)
#After the above we are determined to create a for loop so that 
# we can know the text inside each h5 tag 