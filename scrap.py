from bs4 import BeautifulSoup
import requests
import csv
import time

def scrape_poems( url):
    # Base URL
    
    # Headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Create BeautifulSoup object
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the poem list container
        poem_list = soup.find('ul', class_='list-poems nuevolist orderprimero')
        
        if not poem_list:
            print("Could not find the poem list container")
            return
        
        # Get all poem links
        poem_links = poem_links = poem_list.find_all('a', rel='nofollow')
        
        # Create/open a text file to store the poems
        with open('combine_poems1.txt', 'w', encoding='utf-8') as file:
            
            # Iterate through each poem link
            for i, link in enumerate(poem_links, 1):
                poem_url = link['href']
                if not poem_url.startswith('http'):
                    poem_url = 'https:'+ poem_url
                
                try:
                    # Add delay between requests to be polite
                    time.sleep(2)
                    
                    # Get the poem page
                    poem_response = requests.get(poem_url, headers=headers)
                    poem_response.raise_for_status()
                    
                    # Parse the poem page
                    poem_soup = BeautifulSoup(poem_response.text, 'html.parser')
                    
                    # Get poem title
                    title = poem_soup.find('h1', class_='title-poem')
                    title_text = title.text.strip() if title else "Untitled"
                    
                    # Get poem content
                    poem_content = poem_soup.find('div', class_='poem-entry')
                    poem_text = poem_content.text.strip() if poem_content else "No content found"
                    
                    # Write to file with formatting
                    file.write(f"{'='*50}\n")
                    file.write(f"Poem {i}: {title_text}\n")
                    file.write(f"{'='*50}\n\n")
                    file.write(poem_text + "\n\n\n")
                    
                    print(f"Successfully scraped: {title_text}")
                    
                except Exception as e:
                    print(f"Error scraping poem at {poem_url}: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # urls=["https://mypoeticside.com/poets/charles-bukowski-poems","https://mypoeticside.com/poets/william-shakespeare-poems",
        #   "https://mypoeticside.com/poets/sylvia-plath-poems","https://mypoeticside.com/poets/john-milton-poems","https://mypoeticside.com/poets/t-s-eliot-poems","https://mypoeticside.com/poets/emily-dickinson-poems"
        #   ,"https://mypoeticside.com/poets/john-donne-poems","https://mypoeticside.com/poets/alfred-lord-tennyson-poems","https://mypoeticside.com/poets/rabindranath-tagore-poems","https://mypoeticside.com/poets/rudyard-kipling-poems"]
    urls=["https://mypoeticside.com/poets/rudyard-kipling-poems"]
    for url in urls:  
        scrape_poems(url)