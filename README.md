# Musanze Harvest Intelligence Service (SWE 3513 - Assignment 2)

* **Student Name:** Joel Eliezer Youto Mongar Jr
* **Registration Number:** 25/27585
* **Institutional Username:** ug2527585@ines.ac.rw
* **Verification Code:** AI-A2-7585

## Project Overview
An individual machine learning pipeline and FastAPI web service built for a potato cooperative in Musanze. Generates a reproducible synthetic dataset based on registration number seed `7585`, handles deliberate missing values and duplicates, trains regression, classification, and clustering models, and serves inference locally.

## Setup & Execution Commands
bash
python -m venv .venv
pip install -r requirements.txt
python run_all.py --student-id ug2527585@ines.ac.rw --seed 7585
python predict.py --json samples/valid.json
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000

