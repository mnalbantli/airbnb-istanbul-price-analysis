# Istanbul Airbnb Price Analysis & Price Estimator App

This project explores Airbnb listing prices in Istanbul using real-world data from Inside Airbnb. It includes data cleaning, feature engineering, exploratory data analysis (EDA), and machine learning modeling — all presented in a well-documented Jupyter notebook.  

In addition, a **Streamlit-powered app** has been built to let users interactively predict nightly prices based on listing features.

---

## Project Structure

```
airbnb-istanbul-price-analysis/
│
├── datasets/                # Raw CSV files
├── notebooks/               # EDA, modeling, feature importance
│   └── analysis.ipynb
├── streamlit_app/           # Mini app folder
│   ├── app.py               # Streamlit application
│   └── README.md            # App-specific documentation
├── rf_airbnb_model.pkl      # Trained Random Forest model (optional in repo)
├── README.md                # Project documentation (this file)
```

---

## Project Goals

- Clean and prepare Istanbul Airbnb listing data
- Explore factors affecting nightly price
- Visualize relationships (room type, review score, location)
- Build a regression model to predict prices
- Deploy a lightweight app for real-time predictions

---

## Live App Demo

Try the interactive version of this project in the Streamlit app:

[Istanbul Airbnb Price Estimator →](./streamlit_app/README.md)

---

## Tools & Tech Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `joblib`
- `folium`, `Streamlit`
- Model hosted on [Hugging Face](https://huggingface.co/mnalbantli/airbnb-istanbul-model)

---

## Dataset Source

This project is based on publicly available data from **Inside Airbnb**:

🔗 [https://insideairbnb.com/get-the-data.html](https://insideairbnb.com/get-the-data.html)

---

## About Me

I'm Mustafa, a data analytics graduate student at UTC who is passionate about combining behavioral psychology and data science to build smart, human-centered solutions.  

- 🔗 [LinkedIn](https://www.linkedin.com/in/YOUR-LINKEDIN-USERNAME)
- 💻 [GitHub](https://github.com/YOUR-GITHUB-USERNAME)

---

> If you like this project, feel free to ⭐ star the repo or share it!  
> Built with ❤️ for portfolio, growth, and impact.
