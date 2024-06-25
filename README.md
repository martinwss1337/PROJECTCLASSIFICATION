# Email Spam Classification with LIME Explainability

Email Spam Classification with LIME Explainability is a Python tool that classifies emails as spam or not spam and highlights influential words in the classification decision using LIME (Local Interpretable Model-agnostic Explanations) for the best model, support vector machine with term frequency - inverse document frequency.

## Installation

First, clone the repository:

```bash
git clone https://github.com/martinwss1337/PROJECTCLASSIFICATION.git
cd PROJECTCLASSIFICATION

```
Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```
Then install the required packages:

```bash
pip install -r requirements.txt
```
Usage
To run the application, use:

```bash
python GUI/gui.py
```
Interact with the GUI:


1- Select the classification method and model.

2-Input your email text.

3-Click "Check Input Email" to see the classification (and highlighted explanation if SVM with tf-idf is selected).