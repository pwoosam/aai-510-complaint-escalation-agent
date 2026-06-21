# AAI-510 Complaint Escalation Agent

This project was developed for **AAI-510: Agent Systems and Tool Use** in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**Project Status:** Complete

## Project Overview

The Complaint Escalation Agent is designed to analyze customer complaints, retrieve relevant historical complaint information and internal operational policies, and generate escalation recommendations and customer response drafts.

The project combines Databricks AI tools, retrieval-augmented generation (RAG), vector search, and large language models to support customer service operations.

## Team Members

* Dina Othman 
* Patrick Woo-Sam 
* Cameron Aljilani 

## Project Components

### Data Engineering

* CFPB Consumer Complaints dataset ingestion
* Data cleaning and preprocessing
* PII masking using regex-based techniques
* Retrieval text generation
* Vector Search index preparation

### Agent Engineering

* Router model for complaint classification
* Reasoning model for escalation decisions
* Customer response generation
* Tool integration with Databricks Vector Search and SQL Unity Catalog Functions

### Evaluation

* MLflow tracing
* LLM-as-a-Judge evaluation
* Performance and cost analysis

## Technologies

* Databricks
* Unity Catalog
* Delta Tables
* Databricks Vector Search
* MLflow
* Python
* PySpark
* Large Language Models (LLMs)

## Repository Structure

* `agent_workflow.ipynb` - Notebook for testing, tracing, and evaluating example complaints using the agentic workflow.
* `agent_workflow.py` - Agentic workflow for evaluating customer complaints
* `agent_workflow_setup.ipynb` - Notebook to create `predictions` table, `customer_transactions` table, and `get_customer_transactions` SQL UC Function/Tool
* `CFPB_Data_Engineering.ipynb` - Data engineering pipeline
* `export_traces.ipynb` - Exports MLflow experiment traces to excel
* `mlflow_traces` - Destination folder for exported MLflow experiment traces
* `internal_policy_playbook.txt` - Internal escalation procedures
* `README.md` - Project documentation


## Dataset

The project uses the CFPB Consumer Complaints dataset as the primary source of customer complaint information. Internal escalation policies are used as a secondary retrieval source for operational guidance.

CFPB Consumer Complaints Dataset (Kaggle):
https://www.kaggle.com/datasets/sherrytp/consumer-complaints/data

The raw dataset is not included in this repository due to file size limitations. Users should download the dataset from Kaggle and upload it to Databricks before running the data engineering pipeline.

## License

This repository is intended for academic use as part of the USD MS-AAI program.
