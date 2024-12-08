# Crop Recommendation System

### Project Overview
The Crop Recommendation System uses machine learning and deep learning models to predict the most suitable crops based on soil and environmental data. It generates detailed recommendation reports, empowering farmers and agricultural professionals with actionable insights for sustainable farming.

---

### Installation Guide

#### Prerequisites
Ensure the following tools are installed on your system:
- Python 3.8 or higher (3.10 is recommended)
- Git
- pip (Python package manager)

---

#### 1. Clone the Repository
To get the project code, run the following command:
```bash
git clone https://github.com/Gravitar07/crop-recommendation-system.git
cd crop-recommendation-system
```

---

#### 2. Set Up a Virtual Environment (Optional but Recommended)
Create and activate a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```

---

#### 3. Install Dependencies
Install the required libraries and packages:
```bash
pip install -r requirements.txt
```

---

#### 4. Prepare the Dataset
- Download the dataset from the specified source (e.g., Kaggle or any provided link).
- Save the dataset file in the `data/` directory of the project.
- The dataset should be in a CSV format.
- Ensure the dataset structure aligns with the requirements of the ML model.

---

#### 5. Ensure to add API KEY
- Ensure you created a .env file in root folder and add your API key for GROQ
- The .env file should contain the following:
    - GROQ_API_KEY = "your api key"

---

#### 6. Run the Project
To start the Crop Recommendation System:
1. Navigate to the project directory.
2. Run the application using the following command:
```bash
python app.py
```
3. Access the application via the local server URL (e.g., `http://localhost:5000`) in your web browser.

---

### Troubleshooting
If you encounter issues:
- Ensure all dependencies are correctly installed by re-running `pip install -r requirements.txt`.
- Verify that the dataset is in the correct format and placed in the `data/` directory.
- Check for Python version compatibility (Python 3.8 or higher is required).

---

### Future Enhancements
- Integration of a database for storing user inputs and historical recommendations.
- Deployment to a cloud platform for global accessibility.
- Multilingual support for recommendation reports.

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.
