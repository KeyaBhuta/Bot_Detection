# Botnet Detection System

## Project Title
**Advanced Botnet Detection System using Machine Learning**

A comprehensive machine learning-based system for detecting botnet activities in network traffic using ensemble learning algorithms and a web-based interface for real-time analysis.

---

## Dataset Details

### UNSW-NB15 Dataset
- **Source**: UNSW-NB15 Network Traffic Dataset
- **Description**: A comprehensive network intrusion dataset created by the Australian Centre for Cyber Security (ACCS)
- **Dataset Size**: 82,332 network flow records
- **Features**: 45 features including network flow characteristics
- **Download Link**: 
  - Official: https://research.unsw.edu.au/projects/unsw-nb15-dataset
  - Alternative: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

### Key Features Used
The system utilizes flow-based features for botnet detection:
- **Temporal Features**: Connection duration (`dur`), inter-packet arrival times (`sinpkt`, `dinpkt`)
- **Packet Statistics**: Source/destination packet counts (`spkts`, `dpkts`), bytes (`sbytes`, `dbytes`)
- **Network Characteristics**: Time-to-live values (`sttl`, `dttl`), jitter (`sjit`, `djit`)
- **Traffic Patterns**: Load metrics (`sload`, `dload`), packet loss (`sloss`, `dloss`)
- **Protocol Information**: Protocol type (`proto`), service type (`service`), connection state (`state`)
- **TCP Metrics**: Round trip time (`tcprtt`), window sizes (`swin`, `dwin`), sequence numbers

### Data Preprocessing
- Missing value imputation using median values
- Categorical feature encoding using Label Encoding
- Feature scaling using StandardScaler
- Train-test split (80-20) with stratification
- Feature engineering: bytes per packet, packets per second, byte ratio

---

## Algorithm/Model Used

### Machine Learning Models Evaluated

The system trains and compares multiple machine learning algorithms:

1. **Random Forest Classifier**
   - Parameters: `n_estimators=100`, `max_depth=20`, `min_samples_split=10`, `min_samples_leaf=4`
   - Advantages: Handles non-linear relationships, feature importance analysis

2. **Gradient Boosting Classifier**
   - Parameters: `n_estimators=100`, `max_depth=10`, `learning_rate=0.1`
   - Advantages: Sequential learning, high predictive performance

3. **Logistic Regression**
   - Parameters: `max_iter=1000`
   - Advantages: Interpretable, fast training

4. **Decision Tree Classifier**
   - Parameters: `max_depth=20`, `min_samples_split=10`
   - Advantages: Simple, interpretable

### Model Selection
The best model is automatically selected based on **F1-Score** to balance precision and recall, which is crucial for security applications where both false positives and false negatives are important.

### Model Performance Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: Proportion of true botnet detections among all botnet predictions
- **Recall**: Proportion of actual botnets correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

---

## Results

### Model Evaluation Metrics

The system generates comprehensive evaluation metrics and visualizations:

#### Performance Metrics
- **Test Accuracy**: Varies based on the selected best model (typically 85-95%+)
- **Precision**: Measures the accuracy of botnet predictions
- **Recall**: Measures the ability to detect all botnet instances
- **F1-Score**: Balanced metric for model comparison
- **ROC-AUC**: Overall model discriminative ability

#### Visualizations Generated

1. **Model Comparison Chart**: Bar chart comparing train and test accuracy across all models
2. **Confusion Matrix**: Heatmap showing true positives, true negatives, false positives, and false negatives
3. **ROC Curves**: Comparison of ROC curves for all models with AUC scores
4. **Feature Importance**: Top 15 most important features for tree-based models

#### Generated Files
- `model_evaluation.png`: Comprehensive visualization dashboard
- `eda_visualizations.png`: Exploratory data analysis charts
- `botnet_model.pkl`: Trained model (best performing)
- `model_info.pkl`: Model performance metrics and metadata

### Sample Results Structure
```
Model Performance:
- Best Model: [Selected based on F1-Score]
- Test Accuracy: XX.XX%
- Precision: X.XXXX
- Recall: X.XXXX
- F1-Score: X.XXXX
- ROC-AUC: X.XXXX
```

---

## Conclusion

### Key Achievements
1. **Effective Detection**: The system successfully identifies botnet activities in network traffic using machine learning techniques
2. **Multiple Model Comparison**: Comprehensive evaluation of multiple algorithms ensures optimal model selection
3. **Real-time Application**: Flask-based web interface enables practical deployment for network security monitoring
4. **Feature Engineering**: Flow-based features effectively capture botnet behavioral patterns

### Insights
- Ensemble methods (Random Forest, Gradient Boosting) typically perform best for this task
- Flow-based features are highly effective for botnet detection
- Feature importance analysis reveals key indicators of botnet behavior
- The system provides a balance between detection accuracy and false positive rates

### Limitations
- Dataset-specific: Model performance depends on the training data characteristics
- Feature dependency: Requires specific network flow features to be present
- Static model: May need retraining for new botnet variants

---

## Future Scope

### Short-term Improvements
1. **Deep Learning Integration**
   - Implement LSTM/GRU networks for sequential pattern detection
   - Convolutional Neural Networks (CNNs) for feature extraction
   - Autoencoders for anomaly detection

2. **Real-time Streaming**
   - Integration with network monitoring tools (e.g., Wireshark, tcpdump)
   - Real-time packet capture and analysis
   - Stream processing using Apache Kafka or similar

3. **Enhanced Feature Engineering**
   - Graph-based features (network topology analysis)
   - Time-series features (temporal patterns)
   - Behavioral analysis features

### Long-term Enhancements
1. **Advanced ML Techniques**
   - Transfer learning for adapting to new botnet types
   - Federated learning for privacy-preserving collaborative detection
   - Reinforcement learning for adaptive detection strategies

2. **System Integration**
   - Integration with SIEM (Security Information and Event Management) systems
   - Automated response mechanisms (blocking, alerting)
   - Dashboard for network administrators

3. **Scalability**
   - Distributed computing for large-scale network analysis
   - Cloud deployment (AWS, Azure, GCP)
   - Microservices architecture

4. **Research Directions**
   - Zero-day botnet detection
   - Encrypted traffic analysis
   - IoT botnet detection
   - Multi-class classification (different botnet families)

5. **User Experience**
   - Mobile application for monitoring
   - API for third-party integrations
   - Advanced visualization and reporting
   - Historical analysis and trending

---

## References

### Academic Papers
1. Moustafa, N., & Slay, J. (2015). "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." *2015 Military Communications and Information Systems Conference (MilCIS)*.

2. Garcia, S., Grill, M., Stiborek, J., & Zunino, A. (2014). "An empirical comparison of botnet detection methods." *Computers & Security*, 45, 100-123.

3. Gu, G., Perdisci, R., Zhang, J., & Lee, W. (2008). "BotMiner: Clustering analysis of network traffic for protocol-and structure-independent botnet detection." *USENIX Security Symposium*.

### Datasets
- UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Kaggle UNSW-NB15: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

### Libraries and Frameworks
- **scikit-learn**: Machine learning algorithms and preprocessing
- **Flask**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib & seaborn**: Data visualization

### Documentation
- scikit-learn Documentation: https://scikit-learn.org/stable/
- Flask Documentation: https://flask.palletsprojects.com/
- UNSW-NB15 Dataset Documentation: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

---

## Installation and Usage

### Prerequisites
- Python 3.7+
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   cd Bot_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and prepare dataset**
   - Download UNSW-NB15 dataset
   - Place training CSV file in the project directory
   - Update the path in `botnet_preprocessing.ipynb` if needed

4. **Run data preprocessing**
   ```bash
   jupyter notebook botnet_preprocessing.ipynb
   ```
   Execute all cells to generate preprocessed data files

5. **Train the model**
   ```bash
   python botnet_model_training.py
   ```

6. **Run the Flask application**
   ```bash
   python app.py
   ```

7. **Access the web interface**
   - Open browser: http://localhost:5000
   - Upload a CSV file with network flow data
   - View detection results

### Project Structure
```
Bot_Detection/
├── app.py                          # Flask web application
├── botnet_model_training.py        # Model training script
├── botnet_preprocessing.ipynb     # Data preprocessing notebook
├── botnet_model.pkl               # Trained model
├── model_info.pkl                 # Model metadata
├── scaler.pkl                     # Feature scaler
├── feature_names.pkl              # Feature names
├── label_encoders.pkl             # Label encoders
├── model_evaluation.png           # Evaluation visualizations
├── eda_visualizations.png         # EDA charts
├── requirements.txt               # Python dependencies
├── templates/
│   └── index.html                 # Web interface
└── README.md                      # This file
```

---

## Authors
**AIML Students** - Practical Examination 2024

---

## License
This project is developed for educational purposes as part of an AIML (Artificial Intelligence and Machine Learning) course project.

---

## Acknowledgments
- UNSW Canberra for providing the UNSW-NB15 dataset
- scikit-learn community for excellent ML tools
- Flask team for the web framework

