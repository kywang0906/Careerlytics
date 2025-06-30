import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, NoSuchElementException
from urllib.parse import urljoin
from selenium_stealth import stealth

COMPANY = "apple"
ROLE = "data scientist"
ABBR = "ds"
QUERY = f'site:linkedin.com/in/ intitle:"{COMPANY.capitalize()}" AND intitle:"{ROLE.capitalize()}"'
NEW_URLS_TO_FETCH = 1200
EXISTING_URLS_FILE = f"dataset/{COMPANY}/{COMPANY}_{ABBR}_urls.txt"
OUTPUT_FILENAME = "{COMPANY}_{ABBR}_urls.txt"

def load_existing_urls(filename):
    loaded_urls = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    loaded_urls.add(url)
        print(f"Successfully loaded {len(loaded_urls)} URLs from '{filename}'.")
    except FileNotFoundError:
        print(f"Warning: '{filename}' doesn't exist.")
    except Exception as e:
        print(f"Something wrong happended when loading '{filename}'. Error: {e}")
    return loaded_urls

def scroll_to_element(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element)
    time.sleep(0.5)

def is_navigation_successful(driver, old_first_result_href):
    """Check whether the next page is loaded by comparing the first href attribute"""
    try:
        WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.b_algo h2 > a")))
        new_first_result_href = driver.find_element(By.CSS_SELECTOR, "li.b_algo h2 > a").get_attribute('href')
        if new_first_result_href and new_first_result_href != old_first_result_href:
            return True
    except (TimeoutException, NoSuchElementException):
        print("Failed. Search results not found in new page.")
        return False
    return False

def main():
    existing_urls = load_existing_urls(EXISTING_URLS_FILE)
    
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    if 'PROXY' in locals() and PROXY:
        options.add_argument(f'--proxy-server={PROXY}')
        
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Fail to start WebDriver: {e}")
        return
        
    stealth(driver, languages=["en-US", "en"], vendor="Google Inc.", platform="Win32", webgl_vendor="Intel Inc.", renderer="Intel Iris OpenGL Engine", fix_hairline=True)

    newly_fetched_urls = set()
    current_page = 1

    try:
        print("Visit Google...")
        driver.get("https://www.google.com")
        time.sleep(random.uniform(2, 4))
        
        print(f"\nGoing to Bing and search: {QUERY}")
        driver.get("https://bing.com")
        time.sleep(random.uniform(1, 3))
        search_box = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "sb_form_q")))
        search_box.send_keys(QUERY)
        time.sleep(random.uniform(0.5, 1.5))
        search_box.send_keys(Keys.ENTER)
        print("loading...")
        time.sleep(random.uniform(3, 5))
        
        while len(newly_fetched_urls) < NEW_URLS_TO_FETCH:
            print(f"\nProcessing results in page {current_page}. (Goal: {NEW_URLS_TO_FETCH} new URLs. Currently found: {len(newly_fetched_urls)})")
            
            try:
                first_result_link = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.b_algo h2 > a")))
                old_first_result_href = first_result_link.get_attribute('href')
                print(f"當前頁面的第一個結果標記為: {old_first_result_href[:70]}...")
            except (TimeoutException, NoSuchElementException):
                print("Results not found in current page. Fetching process terminated.")
                break

            result_links = driver.find_elements(By.CSS_SELECTOR, "li.b_algo h2 > a")
            for link in result_links:
                try:
                    url_text = link.get_attribute('href')
                    
                    if url_text and "linkedin.com/in/" in url_text:
                        if url_text not in existing_urls and url_text not in newly_fetched_urls:
                            newly_fetched_urls.add(url_text)
                            print(f"New URLs found (Collected: {len(newly_fetched_urls)}/{NEW_URLS_TO_FETCH}): {url_text}")
                            if len(newly_fetched_urls) >= NEW_URLS_TO_FETCH:
                                break
                except Exception as e:
                    print(f"Error when fetching single URL: {e}")

            if len(newly_fetched_urls) >= NEW_URLS_TO_FETCH:
                break

            try:
                print("\n--- Start turning pages ---")
                next_page_xpath = "//a[@aria-label='下一頁' or @aria-label='Next page'] | //a[@title='下一頁' or @title='Next page'] | //li[contains(@class, 'b_pagNext')]/a | //a[contains(@class, 'sb_pagN')]"
                navigated = False

                try:
                    button_a = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, next_page_xpath)))
                    scroll_to_element(driver, button_a)
                    button_a.send_keys(Keys.ENTER)
                    time.sleep(3)
                    if is_navigation_successful(driver, old_first_result_href):
                        navigated = True
                except Exception:
                    print(f"Fail to turn page.")
                
                if navigated:
                    current_page += 1
                    time.sleep(random.uniform(2, 4))
                else:
                    print("Cannot turn to the next page.")
                    break
            except Exception as e:
                print(f"Unknown error when turning page: {e}")
                break
    except Exception as e:
        print(f"Unknown error: {e}")
    finally:
        if 'driver' in locals() and driver:
            print("\nWebdriver closing...")
            driver.quit()
    
    print(f"\n--- Conclusion ---")
    print(f"Goal: {NEW_URLS_TO_FETCH} fetched urls")
    print(f"From '{EXISTING_URLS_FILE}' loaded {len(existing_urls)} existing urls")
    print(f"New URLs fetched: {len(newly_fetched_urls)}")
    
    # Save file
    if newly_fetched_urls:
        try:
            with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
                for url_item in newly_fetched_urls:
                    f.write(url_item + "\n")
            print(f"\n[Success] {len(newly_fetched_urls)} new URLs saved to file: {OUTPUT_FILENAME}")
        except Exception as e:
            print(f"\n[Fail] File not saved to {OUTPUT_FILENAME}. Error: {e}")

        print("\nFirst 20 URLs list preview:")
        for i, url_item in enumerate(list(newly_fetched_urls)[:20]):
            print(f"{i+1}. {url_item}")
        if len(newly_fetched_urls) > 20:
            print(f"... ({len(newly_fetched_urls) - 20} URLs remaining. Check file: {OUTPUT_FILENAME}.)")

if __name__ == "__main__":
    main()