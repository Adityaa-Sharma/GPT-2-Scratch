from bs4 import BeautifulSoup
import requests
import csv
import time
import string
from typing import List
from datetime import datetime
import os

# Define a constant for the output file
POEMS_FILE = 'all_poems_collection.txt'

def scrape_poems(url, author_name, total_poems_scraped):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"\nScraping poems for author: {author_name}")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        poem_list = soup.find('ul', class_='list-poems nuevolist orderprimero')
        
        if not poem_list:
            print(f"No poems found for author: {author_name}")
            return total_poems_scraped
        
        poem_links = poem_list.find_all('a', rel='nofollow')
        author_poem_count = len(poem_links)
        print(f"Found {author_poem_count} poems for {author_name}")
        
        with open(POEMS_FILE, 'a', encoding='utf-8') as file:
            for i, link in enumerate(poem_links, 1):
                poem_url = link['href']
                if not poem_url.startswith('http'):
                    poem_url = 'https:' + poem_url
                
                try:
                    time.sleep(2)
                    poem_response = requests.get(poem_url, headers=headers)
                    poem_response.raise_for_status()
                    
                    poem_soup = BeautifulSoup(poem_response.text, 'html.parser')
                    
                    title = poem_soup.find('h1', class_='title-poem')
                    title_text = title.text.strip() if title else "Untitled"
                    
                    poem_content = poem_soup.find('div', class_='poem-entry')
                    poem_text = poem_content.text.strip() if poem_content else "No content found"
                    
                    # Write only title and content to file with minimal formatting
                    file.write(f"{title_text}\n")
                    file.write(f"{poem_text}\n\n")
                    
                    total_poems_scraped += 1
                    print(f"[{total_poems_scraped}] Scraped: {title_text} by {author_name}")
                    
                except Exception as e:
                    print(f"Error scraping poem at {poem_url}: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"An error occurred while scraping {author_name}'s poems: {str(e)}")
    
    return total_poems_scraped

def get_poets(url: str) -> List[dict]:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        poem_list = soup.find('ul', class_='list-poems')
        if not poem_list:
            return []
            
        poet_links = poem_list.find_all('a')
        
        poets = []
        
        for link in poet_links:
            poets.append({
                'name': link.text.strip(),
                'url': link['href']
            })
            
        return poets
        
    except requests.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []
    except Exception as e:
        print(f"Error parsing the content: {e}")
        return []

if __name__ == "__main__":
    start_time = datetime.now()
    poets_list = []
    total_poems_scraped = 0
    char = [i for i in string.ascii_lowercase]

    print("Starting the scraping process...")
    
    # Create or clear the poems file at the start
    with open(POEMS_FILE, 'w', encoding='utf-8') as file:
        file.write(f"Poetry Collection\nStarted: {datetime.now()}\n\n")
    
    # First, collect all poets
    for i in char:
        print(f"\nFetching poets for letter: {i.upper()}")
        result = get_poets(f"https://mypoeticside.com/{i}-browse")
        poets_list.extend(result)
        print(f"Found {len(result)} poets for letter {i.upper()}")
    
    total_poets = len(poets_list)
    print(f"\nTotal poets found: {total_poets}")
    
    # Then scrape poems for each poet
    for index, poet in enumerate(poets_list, 1):
        print(f"\nProcessing poet {index}/{total_poets}: {poet['name']}")
        total_poems_scraped = scrape_poems(poet['url'], poet['name'], total_poems_scraped)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Append final statistics to the file
    with open(POEMS_FILE, 'a', encoding='utf-8') as file:
        file.write("\n" + "="*50 + "\n")
        file.write("Collection Statistics:\n")
        file.write(f"Total poets processed: {total_poets}\n") 
        file.write(f"Total poems collected: {total_poems_scraped}\n")
        file.write(f"Collection completed: {end_time}\n")
        file.write(f"Total time taken: {duration}\n")
    
    print("\n" + "="*50)
    print("Scraping Complete!")
    print(f"Total poets processed: {total_poets}")
    print(f"Total poems scraped: {total_poems_scraped}")
    print(f"Average poems per poet: {total_poems_scraped/total_poets:.2f}")
    print(f"Time taken: {duration}")
    print(f"All poems have been saved to: {POEMS_FILE}")
    print("="*50)