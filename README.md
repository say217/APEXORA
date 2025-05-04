#                                                    APEXORA

**Apexora** is an advanced multimodal deep learning-based financial prediction platform. It accurately predicts both stock and cryptocurrency (e.g., Bitcoin) prices using RNN and LSTM models. Apexora provides real-time analytics, historical data, sentiment-based news analysis, and visually rich financial dashboards.

## MODEL PREDICTION VISUALIZATION

![image](https://github.com/user-attachments/assets/cce8b3a1-fc84-497c-9c34-b0546c33bdf8)

## FOR CRYPTOCURRENCY
![image](https://github.com/user-attachments/assets/47d7aac1-9264-43ea-b08d-4fc8d04348f5)

---

##  Features

-  **Stock Price Prediction** using deep learning (RNN, LSTM)
-  **Bitcoin & Cryptocurrency Price Forecasting**
-  **Real-Time Stock Price Monitoring**
-  **Downloadable Historical Stock Data** (based on user-defined year ranges in JSON format)
-  **News-Based Sentiment Analysis** for individual stocks
-  **Stock Comparison Dashboard** based on trends and news sentiment
-  **Live Financial & Business News Feed**
-  **Interactive Visualizations** using `matplotlib`, `ggplot`

---

##  Tech Stack

| Category         | Tools/Frameworks                         |
|------------------|-------------------------------------------|
| Deep Learning     | Keras, TensorFlow (RNN & LSTM)           |
| Data Handling     | pandas, yfinance                         |
| Visualization     | matplotlib, ggplot                       |
| Web Development   | Flask / Streamlit *(customizable)*       |
| APIs              | yfinance, NewsAPI, CryptoCompare         |

---
## JSON DATA OF COMPANY
![image](https://github.com/user-attachments/assets/1b86a9d3-c2f8-4bf5-8882-76b1e489f32d)




## 🗂️ Project Structure
```

PROJECT OVERVIEW

|----------MAIN WEBPAGE/- WELCOME PAGE-/
|
|
project/
│
├── app1/                            # First Flask App (App1)   [ Prediction main model for stock]
│   ├── venv/                        # Virtual environment for App1
│   ├── __init__.py                  # Initializes the app1
│   ├── routes.py                    # Routes for app1
│   ├── models.py                    # Database models for app1
│   ├── forms.py                     # Forms for app1 (e.g., for user input)
│   ├── templates/
│   │   ├── app1_template.html       # Template for app1
│   │   └── layout.html              # Base layout template
│   ├── static/
│   │   ├── img/
│   │   └── style.css                # Static files for app1
│   ├── config.py                    # Configuration settings for app1 (e.g., DB URI)
│   └── utils.py                     # Helper functions for app1
│   └── requirements.txt             # Dependencies for app1
│
├── app2/                            # Second Flask App (App2)    [ Prediction Model Crypto Currency]
│   ├── venv/                        # Virtual environment for App2
│   ├── __init__.py                  # Initializes the app2
│   ├── routes.py                    # Routes for app2
│   ├── models.py                    # Database models for app2
│   ├── forms.py                     # Forms for app2
│   ├── templates/
│   │   └── app2_template.html       # Template for app2
│   ├── static/
│   │   └── img/
│   ├── config.py                    # Configuration settings for app2
│   └── utils.py                     # Helper functions for app2
│   └── requirements.txt             # Dependencies for app2
|
|
├── app3/                            # Second Flask App (App3)    [ Live stock market chnages.. Stock data] 
│   ├── venv/                        # Virtual environment for App3
│   ├── __init__.py                  # Initializes the app3
│   ├── routes.py                    # Routes for app2
│   ├── models.py                    # Database models for app3
│   ├── forms.py                     # Forms for app3
│   ├── templates/
│   │   └── app2_template.html       # Template for app3
│   ├── static/
│   │   └── img/
│   ├── config.py                    # Configuration settings for app2
│   └── utils.py                     # Helper functions for app2
│   └── requirements.txt             # Dependencies for app2
|
├── app4/                            # Second Flask App (App4)    [ Data Extractions acording To the user ......] 
│   ├── venv/                        # Virtual environment for App3
│   ├── __init__.py                  # Initializes the app3
│   ├── routes.py                    # Routes for app2
│   ├── models.py                    # Database models for app3
│   ├── forms.py                     # Forms for app3
│   ├── templates/
│   │   └── app2_template.html       # Template for app3
│   ├── static/
│   │   └── img/
│   ├── config.py                    # Configuration settings for app2
│   └── utils.py                     # Helper functions for app2
│   └── requirements.txt             # Dependencies for app2
│
├── app5/                            # Second Flask App (App4)    [ Current news and Arcticles .....] 
│   ├── venv/                        # Virtual environment for App3
│   ├── __init__.py                  # Initializes the app3
│   ├── routes.py                    # Routes for app2
│   ├── models.py                    # Database models for app3
│   ├── forms.py                     # Forms for app3
│   ├── templates/
│   │   └── app2_template.html       # Template for app3
│   ├── static/
│   │   └── img/
│   ├── config.py                    # Configuration settings for app2
│   └── utils.py                     # Helper functions for app2
│   └── requirements.txt             # Dependencies for app2
|
|
├── auth/                            # Authentication Module (for user login/registration)
│   ├── __init__.py                  # Initializes the authentication module
│   ├── routes.py                    # Routes for user authentication
│   ├── models.py                    # Database models for authentication (user)
│   ├── forms.py                     # Forms for login/registration
│   ├── utils.py                     # Helper functions for auth (password hashing, etc.)
│
├── migrations/                      # Database migrations
│   └── versions/
│
├── main.py                          # Main entry point to run the app (app1 & app2 together)
├── config.py                        # Global configuration settings (e.g., database URI, secret keys)
├── requirements.txt                 # List of global dependencies
└── Dockerfile                       # Docker configuration for deployment



```


---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/apexora.git
cd apexora
