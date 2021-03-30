# Disaster-Response-Pipeline

### Table of Contents

1. [Project Summary](#summary)
2. [Running the Scripts and Web App](#scripts)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Summary <a name="summmary"></a>
In this project, a disaster dataset from Figure Eight was used to build a model for an API that classifies disaster messages.

A dataset containing real messages that were sent during disaster events was used to create a machine learning pipeline to categorize these events, so that messages could be sent to an appropriate disaster relief agency.

The project also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Running the Scripts and Web App <a name="Scripts"></a>

1. To run the scripts and display the web app, you can execute these commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database
      `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
- To run ML pipeline that trains classifier and saves
      `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## File Descriptions <a name="files"></a>

Data <br />
DisasterResponse.db: Database created in SQLite that was cleaned and transformed to be used in this project.<br />
disaster_categories.csv and disaster_messages.csv : Datasets provided by FigureEight. <br />
process_data.py : Python script that executes the ETL (Execute, Transform and Load) process in the provided data. <br />

App
templates: templates files for the webapp. <br />
run.py: run the web app used to display the plots and the Machine Learning pipeline results.<br />

Models <br />
train_classifier.py: Python Script that executes the Machine Learing pipeline, train the model and save the results in a .pkl file. <br />


## Results<a name="results"></a>

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
