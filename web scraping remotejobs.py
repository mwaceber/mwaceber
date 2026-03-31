from bs4 import BeautifulSoup
import requests

html_doc = requests.get('https://weworkremotely.com/remote-jobs/search?term=AI').text
soup = BeautifulSoup(html_doc, 'lxml')
jobs = soup.find_all('li', class_ = 'new-listing-container-feature')
for job in jobs :
    published_date = job.find_all('p', class_='new-listing__header__icons__date').text.replace(' ', '')
    company_name = job.find_all('span', class_= 'new-listing__company-name').text
    job_title = job.find_all('h3', class_ = 'new-listing__header__title').text
    job_location = job.find_all('p', class_ = 'new-listing__categories__category').text
   