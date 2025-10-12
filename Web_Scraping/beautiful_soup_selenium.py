###################################
# Beautiful Soup
# Getting Started
###################################

# pip install beautifulsoup4

from bs4 import BeautifulSoup

html = """
        <!DOCTYPE html><html><head><title>Example HTML</title></head><body><h1>Hello, World!</h1><p>A simple HTML page for testing web scraping with BeautifulSoup.</p>
                <a class='link' href='www.miuul.com' target='blank' aria-label='Miuul (Opens Miuul Page)'>Click</a>
                <li>Outsider</li>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </body>
            </html>

"""

soup = BeautifulSoup(html, "html.parser")

type(soup)

title = soup.title
type(title)

title.text
title.string

print(soup.prettify())

soup.ul
soup.li

ul = soup.ul
type(ul)
ul.li

###################################
# Beautiful Soup
# Navigating and Searching HTML
###################################

from bs4 import BeautifulSoup

html = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Example HTML</title>
            </head>
            <body>
                <h1>Hello, World!</h1>
                <p id="paragraph" >A simple HTML page for testing web scraping with BeautifulSoup.</p>
                <a class='link' href='www.miuul.com' target='blank' aria-label='Miuul (Opens Miuul Page)'>Click</a>
                <li>Outsider</li>
                <ul>
                    <li class="list-item">Item 1</li>
                    <li class="list-item">Item 2</li>
                </ul>
                <li>Outsider 2</li>
            </body>
            </html>
"""

soup = BeautifulSoup(html, "html.parser")

soup.find("a", attrs={"class": "link", "target": "blank"})

soup.find("li")
soup.find_all("li")

li_elements = soup.find_all("li", attrs={"class": "list-item"})
li_elements
li_elements[-1]


###################################
# Beautiful Soup
# Extracting Data from HTML Elements
###################################

from bs4 import BeautifulSoup

html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Animal Table</title>
  <style>
    table {
      width: 80%;
      border-collapse: collapse;
      margin: 20px;
    }

    th, td {
      border: 1px solid #dddddd;
      text-align: left;
      padding: 8px;
    }

    th {
      background-color: #f2f2f2;
    }

    img {
object-fit: cover;
      max-width: 50px;
      max-height: 50px;
    }
  </style>
</head>
<body>

  <h2>Animal Table</h2>

  <table>
    <thead><tr>
      <th>Image</th>
      <th>Animal</th>
      <th>Description</th>
      <th>Nickname</th>
    </tr></thead>
    <tbody>
    <tr>
      <td><img src="https://images.unsplash.com/photo-1534188753412-3e26d0d618d6?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Lion"></td>
      <td><a href="https://en.wikipedia.org/wiki/Lion" target="_blank">Lion</a></td>
      <td>The lion is a large carnivorous mammal. It is known for its majestic appearance and is often referred to as the "king of the jungle."</td>
      <td> Majestic<br>King  </td>
    </tr>
    <tr>
      <td><img src="https://images.unsplash.com/photo-1551316679-9c6ae9dec224?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Elephant"></td>
      <td><a href="https://en.wikipedia.org/wiki/Elephant" target="_blank">Elephant</a></td>
      <td>Elephants are the largest land animals. They are known for their long trunks and large ears.</td>
      <td> Trunked<br>  Giant</td>
    </tr>
    <tr>
      <td><img src="https://images.unsplash.com/photo-1570481662006-a3a1374699e8?q=80&w=1965&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Dolphin"></td>
      <td><a href="https://en.wikipedia.org/wiki/Dolphin" target="_blank">Dolphin</a></td>
      <td>Dolphins are highly intelligent marine mammals known for their playful behavior and communication skills.</td>
      <td> Playful<br>Communicator</td>
    </tr>
    <tr>
      <td><img src="https://images.unsplash.com/photo-1599631438215-75bc2640feb8?q=80&w=2127&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Butterfly"></td>
      <td><a href="https://en.wikipedia.org/wiki/Butterfly" target="_blank">Butterfly</a></td>
      <td>Butterflies are beautiful insects with colorful wings. They undergo a process called metamorphosis from caterpillar to butterfly.</td>
      <td> Colorful<br>Metamorphosis</td>
    </tr>
    <tr>
      <td><img src="https://images.unsplash.com/photo-1552633832-4f5a1b110980?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D" alt="Penguin"></td>
      <td><a href="https://en.wikipedia.org/wiki/Penguin" target="_blank">Penguin</a></td>
      <td>Penguins are flightless birds that are well-adapted to life in the water. They are known for their tuxedo-like black and white plumage.</td>
      <td> Tuxedoed     <br>Adaptation  </td>
    </tr>
  </tbody>
  </table>
</body>
</html>
"""

soup = BeautifulSoup(html, "html.parser")

tbody_tag = soup.find("tbody")
tr_tag_list = tbody_tag.find_all("tr")
print(tr_tag_list)

tr_tag = tr_tag_list[0]
tr_tag

img_tag = tr_tag.find("img")
a_tag = tr_tag.find("a")

nicname_td = tr_tag.find_all("td")[-1]
desc_td = tr_tag.find_all("td")[-2]

nicname_td.text
nicname_td.get_text(separator=" ", strip=True)

img_tag

alt_attribute = img_tag["alt"]
alt_attribute

scr_attribure = img_tag["src"]
scr_attribure

a_tag

a_tag["href"]
a_tag["target"]

###################################
# Beautiful Soup
# Scraping a Web Page
###################################

#pip install requests

import requests
from bs4 import BeautifulSoup

result = requests.get("https://www.example.com")
result.status_code
result.content
html = result.content
soup = BeautifulSoup(html, "html.parser")

soup.find("h1").text





###################################
# Selenium
# Getting Started
###################################

#pip install selenium

from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument("--headless")  #arkaplanda çalıştırma

driver = webdriver.Chrome(options)
driver.get("http://www.example.com")
driver.title
driver.current_url
driver.quit()

driver = webdriver.Chrome(options)
driver.get("http://www.miuul.com")
driver.title
driver.current_url
driver.quit()

###################################
# Selenium
# Finding Elements and Extracting Data
###################################

from selenium import webdriver
from selenium.webdriver.common.by import By

driver = webdriver.Chrome()
driver.get("http://www.example.com")
element = driver.find_element(By.XPATH, "//a")

type(element)
element.text
element.get_attribute("innerText")
element.get_attribute("href")
element.get_attribute("innerHTML")

###################################
# Selenium
# Finding Elements and Extracting Data "Better Way"
###################################

from selenium import webdriver
from selenium.webdriver.common.by import By

import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("https://www.example.com")
time.sleep(2)
h1_elem = driver.find_element(By.XPATH, "//h1")
a_elem = driver.find_element(By.XPATH, "//a")
p_elem = driver.find_element(By.XPATH, "//p")
#....
#....
#....

# alternative of time.sleep()
#selector = (By.XPATH, "//p")
#wait = WebDriverWait(driver, 10)
#p_element = wait.until(EC.visibility_of_element_located(selector))

# all elements
p_elements = driver.find_elements(By.XPATH, "//p")
p_elements

elem = None
if p_elements:
    elem = p_elements[0]
else:
    print("Elements not found")
print(elem)


###################################
# Selenium
# Interacting with Elements
###################################

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome()
driver.get("https://www.miuul.com")
time.sleep(2)

btn_elements = driver.find_elements(By.XPATH, "//a[@id='login']")
btn = btn_elements[0]
btn.click()

inputs = driver.find_elements(By.XPATH, "//input[@name='arama']")
input = inputs[0]
input
input.send_keys("Data Science", Keys.ENTER)

###################################
# Selenium
# Scrolling and Scrolling Inside Dropdown
###################################
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options)
driver.get("https://www.miuul.com/katalog/egitimler")
time.sleep(2)

a_element = driver.find_elements(By.XPATH, "//a[contains(@href, 'product-road-map')]")[1]
driver.execute_script("arguments[0].scrollIntoView();", a_element)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # down to the page
time.sleep(2)
a_element.click() #### this doesnt work??????????


dropdown_button = driver.find_elements(By.XPATH, "//a[@data-bs-toggle='dropdown']")[1]
dropdown_button.click()
time.sleep(0.5)
ul_element = driver.find_elements(By.XPATH, "//ul[@aria-labelledby='navbarDropdown']")[1]
driver.execute_script("arguments[0].setAttribute('style', arguments[1]);", ul_element, "overflow: scroll; height:80px;")

driver.execute_script("argument[0].focus()", ul_element)

from selenium.webdriver.common.action_chains import ActionChains

actions = ActionChains(driver)
actions.send_keys(Keys.ARROW_DOWN).perform()
time.sleep(0.25)
actions.send_keys(Keys.ARROW_DOWN).perform()
time.sleep(0.25)
actions.send_keys(Keys.ARROW_DOWN).perform()
time.sleep(0.25)
actions.send_keys(Keys.ARROW_DOWN).perform()


###################################
# Selenium
# Pagination
###################################

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Initialize Driver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options)
driver.get("https://learning.miuul.com/enrollments")

course_titles = []
for i in range(1, 999):
    driver.get(f"https://learning.miuul.com/enrollments?page={i}")
    time.sleep(3)
    # Get Course Titles Per Page
    course_elements = driver.find_elements(By.XPATH, "//ul//h3")
    if not course_elements: # len(course_elements) <= 0
        break
    for course in course_elements:
        title = course.get_attribute("innerText")
        course_titles.append(title)

print(course_titles)
print(len(course_titles))


###################################
# Selenium
# Scraping a Web Page
###################################

import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import pandas as pd
# pip install pandas

# Initialize Driver
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options)
driver.get("https://miuul.com/katalog")
time.sleep(2)

input = driver.find_elements(By.XPATH, "//label[contains(text(),'İleri')]/preceding-sibling::input")
input[0].click() if input else None

course_blocks = driver.find_elements(By.XPATH, "//div[contains(@class,'card catalog') and (contains(@class,'block'))]")

data = []
for block in course_blocks:
    course_title = block.find_elements(By.XPATH, ".//h6")
    course_desc = block.find_elements(By.XPATH, ".//p")

    course_title = course_title[0].get_attribute("innerText") if course_title else None
    course_desc = course_desc[0].get_attribute("innerText") if course_desc else None

    print(course_title)
    print(course_desc)



###################################
# Headers & Proxy
###################################

###################################
# Using Proxy with Beautiful Soup
###################################

import requests
from bs4 import BeautifulSoup

url = 'https://www.ipaddress.my/'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7",
    "Accept-Language": "en-US,en;q=0.5"
}
proxies = {
    #username:password@ip_address:port
    "http": 'http://JX2kX2Gf:eWKNTTTF@216.19.205.244:6565',
    "https": 'http://JX2kX2Gf:eWKNTTTF@216.19.205.244:6565',
}

response = requests.get(url,  headers=headers, proxies=proxies)
response.status_code

soup = BeautifulSoup(response.content, 'html.parser')
ip_address_element = soup.find("div", attrs={"class": "panel-body"}).find("li").find("span")
flag_element = soup.find("div", attrs={"class": "panel-body"}).find_all("li")[-1].find("img")
ip_address = ip_address_element.text
flag = flag_element["alt"]

print(ip_address)
print(flag)


###################################
# Using Proxy with Selenium
###################################

import zipfile


def proxies(username, password, endpoint, port):
    manifest_json = """
    {
        "version": "1.0.0",
        "manifest_version": 2,
        "name": "Proxies",
        "permissions": [
            "proxy",
            "tabs",
            "unlimitedStorage",
            "storage",
            "<all_urls>",
            "webRequest",
            "webRequestBlocking"
        ],
        "background": {
            "scripts": ["background.js"]
        },
        "minimum_chrome_version":"22.0.0"
    }
    """

    background_js = """
    var config = {
            mode: "fixed_servers",
            rules: {
              singleProxy: {
                scheme: "http",
                host: "%s",
                port: parseInt(%s)
              },
              bypassList: ["localhost"]
            }
          };

    chrome.proxy.settings.set({value: config, scope: "regular"}, function() {});

    function callbackFn(details) {
        return {
            authCredentials: {
                username: "%s",
                password: "%s"
            }
        };
    }

    chrome.webRequest.onAuthRequired.addListener(
                callbackFn,
                {urls: ["<all_urls>"]},
                ['blocking']
    );
    """ % (endpoint, port, username, password)

    extension = 'proxies_extension.zip'

    with zipfile.ZipFile(extension, 'w') as zp:
        zp.writestr("manifest.json", manifest_json)
        zp.writestr("background.js", background_js)

    return extension


import requests
from selenium import webdriver
from extension import proxies

url = 'https://www.ipaddress.my/'

proxy_username = 'JX2kX2Gf'
proxy_password = 'eWKNTTTF'
proxy_ip_address = '104.239.108.144'
proxy_port = '6379'

# username:password@ip_address:port
options = webdriver.ChromeOptions()
# options.add_argument(f'--proxy-server={proxy_ip_address}:{proxy_port}')

proxies_extension_path = proxies(proxy_username, proxy_password, proxy_ip_address, proxy_port)
options.add_extension(proxies_extension_path)
# selenium-wire

driver = webdriver.Chrome(options=options)
driver.get(url)

###################################
# Using undetected-chromedriver to Pass Bot Tests
###################################

#pip install undetected_chromedriver
import undetected_chromedriver as uc

from selenium import webdriver

url = "https://bot.sannysoft.com"

driver = uc.Chrome()
with driver:
    driver.get(url)
















