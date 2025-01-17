# Profile Recommendation using Cosine Similarity, RAG and GNN

## Running the file Locally

To run these notebooks in any development environment, ensure you have the required libraries installed. Below is a list of libraries which will be used in this notebook:

### Required Libraries

json: For json files handling
numpy: For numerical operations and array handling
scikit-learn: A robust toolkit for machine learning algorithms
pytorch: For PyTorch deep learning framework
torch-geometric: For graph neural network functionality
streamlit: For building web-based user interfaces
langchain: For LLM-based chatbot functionality
langchain-core: For LLM-based chatbot functionality
langchain-community: For LLM-based chatbot functionality
langchain-groq: For LLM-based chatbot functionality
groq: For LLM-based chatbot functionality
openai: For LLM-based chatbot functionality
requests: For making HTTP requests


## Installation Instructions

If you do not already have these libraries installed, you can install them by running the following command:

```sh
pip install json numpy scikit-learn pytorch torch-geometric streamlit langchain langchain-core langchain-community langchain-groq groq openai requests
```
This command installs all the required packages in your environment. Once installed, you can seamlessly run the notebooks locally, leveraging these tools for data processing, model training, visualization, and optimization

## Setup Instructions
Please ensure that your folders and code files are organized correctly. Upload the required folder containing the json data files and any code file into the same folder. 

1. A Linkedin Scraper developed using Bright Data API Key

To send request for scrapping first 1000 profiles from a csv file, run the following command in the terminal:

```sh
python bright_data.py --send_request -i Connections.csv -st 0 -et 1000
```

This will create a csv file which will contain snapshot id for downloading data request. For downloading data after request completion, run the following command in the terminal:

```sh
 python bright_data.py --download_data -i results/req.csv
 ```

 2. Conversion of dataframe to json files for further processing

The dataframe will look like id_03700_03750.csv. This contains details of 50 linkedin profiles. Run all cells of csv_to_json.ipynb to create json files of each profile, which will be saved in json_files. Then run the process_json_files.ipynb file to clean and structure the data to extract relevant information, which will be saved in final_user_profiles folder.

3. Recommendation system using langchain

The file rag_faiss.ipynb shows the steps undertaken to build the recommendation system using langchain, OpenAI and faiss library. To run the recommendation system using streamlit, run the following command in the terminal:

```sh
streamlit run app.py
```

The recommendations are made as cold-start problem, considering 4000 profiles extracted by the scraper. The 4000 profiles are there in json_files folder.

4. Recommendation System using TF-IDF-based feature extraction and Cosine Similarity

Given a user profile, the system recommends similar profiles based on textual features like skills, position, current company, experience, certifications and education. To generate the recommendations, run the following command in the terminal:

```sh
python similarity.py
```

5. Recommendation System using RAG-based Adjacency Matrix and GNN

The file build_adjmatrix.ipynb shows the steps undertaken to curate an adjacency matrix using the user profiles. The adjacency_matrix.csv and user_data.json files are created using this code. To run the recommendation system using using GNN, run the following command in the terminal:

```sh
python gnn.py
```