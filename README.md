A Linkedin Scraper developed using Bright Data API

Send request for scrapping first 1000 profiles from a csv file: python bright_data.py --send_request  -i Connections_Mri.csv -st 0 -et 1000  
This will create a csv file which will contain snapshot id for downloading data request.
Downloading data after request completion: python bright_data.py --download_data -i results/req.csv
