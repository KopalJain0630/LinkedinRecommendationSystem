1. A Linkedin Scraper developed using Bright Data API Key

a. Send request for scrapping first 1000 profiles from a csv file: python bright_data.py --send_request  -i Connections.csv -st 0 -et 1000  
b. This will create a csv file which will contain snapshot id for downloading data request.
c. Downloading data after request completion: python bright_data.py --download_data -i results/req.csv

2. The dataframe must be converted to json files for further processing

a. The dataframe will look like id_03700_03750.csv. This contains details of 50 linkedin profiles.
b. Use csv_to_json.ipynb to create json files of each profile, which will be saved in json_files.
c. Use process_json_files to clean and structure the data to extract relevant information, which will be saved in final_user_profiles.

3. These profile data are used to build the recommendation system using langchain

a. rag_faiss.ipynb shows the steps undertaken to build the recommendation system using langchain, OpenAI and faiss library.
b. app.py is the main application file which will be used to run the recommendation system using streamlit.
c. The recommendations are made as cold-start problem, considering 4000 profiles extracted by the scraper. The 4000 profiles are there in json_files folder.

