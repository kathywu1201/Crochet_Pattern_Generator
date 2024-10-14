from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os
import requests

# chromedriver
chrome_driver_path = "/Users/ciciwxp/Downloads/chromedriver/chromedriver"

# pdf folder
download_dir = "/Users/ciciwxp/Desktop/AC215_pdf"
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# initialize chromedriver
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service)

# download function
def download_pdf(pdf_url, file_name):
    print(f"Attempting to download {file_name} from {pdf_url}...")
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Check for HTTP errors
        pdf_path = os.path.join(download_dir, file_name)
        with open(pdf_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        print(f"Downloaded: {file_name}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {file_name}: {e}")

# rugs info:
#    for page_num in range(1, 4): 
#base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type=Rugs&page={page_num}"
# scarves info:
#base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type=Scarves&page={page_num}"
#for page_num in range(1, 28):   
#blanket info:  
#base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type=Afghans+%26+Blankets&page={page_num}"
#for page_num in range(24, 100): 


# URL: yarnspirations
#base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type=Pillows+%26+Poufs&page={page_num}"


def download_yarnspirations(project_type, total_page_num):
# Loop through pages
    base_url = "https://www.yarnspirations.com/collections/patterns?filter.p.m.global.project_type={project_type}&page={page_num}"

    for page_num in range(1,total_page_num + 1):  # page range
        print(f"Scraping page {page_num}...")
        driver.get(base_url.format(project_type=project_type, page_num=page_num))
        
        # wait til the "Free Pattern" buttons are loaded
        try:
            # wait for elements to be present (located by XPath)
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@class, 'card-button--full')]"))
            )
        except Exception as e:
            print(f"Error while waiting for patterns to load: {e}")
            continue
        
        # find "Free Pattern" buttons by XPath or class
        pattern_links = driver.find_elements(By.XPATH, "//a[contains(@class, 'card-button--full')]")
        
        # debug to see how many links are found on current page
        print(f"Found {len(pattern_links)} pattern links on page {page_num}")
        
        for pattern_link in pattern_links:
            pdf_url = pattern_link.get_attribute('href')
            
            # check if the link is a PDF link
            if pdf_url and pdf_url.endswith('.pdf'):
                # extract the file name from the URL
                file_name = pdf_url.split('/')[-1]
                
                # download the PDF
                download_pdf(pdf_url, file_name)
            else:
                print(f"Skipping non-PDF or invalid link: {pdf_url}")

download_yarnspirations('Skirts', 2)
# quit chromedriver
driver.quit()
