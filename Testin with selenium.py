from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
import time

s = Service("C:/Program Files/Google/chromedriver-win64/chromedriver.exe")
driver = webdriver.Chrome(service= s)
driver.get("https://www.wscubetech.com/")
#driver.get("https://www.tutorialsfreak.com/")


#driver.find_element("""/html/body/div/div[2]/div[2]/section[1]/div/div[1]/div/div/div/div[2]/button""").click()
#time.sleep(2)
#driver.find_element(By.XPATH, """/html/body/section[1]/div[2]/div/div[1]/div[2]/div/button""").click()

#Now we look user input on how to automatically enter any website
driver.get("https://www.google.com/")
time.sleep(1)
search = driver.find_element("""/html/body/div[2]/div[4]/form/div[1]/div[1]/div[1]/div[1]/div[3]/textarea""")
time.sleep(2)
search.send_keys('wscubetech')
search.send_keys(Keys.ENTER)

#How to save the screenshot
#driver.save_screenshot("C:/Users/mwace/Python Projects/full_page.png")
driver.find_element(""""/html/body/section[2]/div/div/div/div[2]/div/div/div/div/div/div/div/div/div""").screenshot_as_png("C:/Users/mwace/Python Projects/nigga.png")

#To determine the height of infinite scrolling
height = driver.execute_script('return document.body.scrollHeight')
print(height)
driver.execute_script("window.ScrollTo(0, document.body.scrollHeight)")

