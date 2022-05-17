url = "https://en.wikipedia.org/wiki/Characters_of_the_Marvel_Cinematic_Universe"

# importing the libraries
from bs4 import BeautifulSoup
import requests

# Make a GET request to fetch the raw HTML content
html_content = requests.get(url).text
# Parse the html content
soup = BeautifulSoup(html_content, "html.parser")
#print(soup.prettify()) # print the parsed data of html
character_list = soup.findAll("span",{'class':'mw-headline'})
for item in character_list:
    print(item.text)
pass