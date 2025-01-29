# **AI Algorithms & Mathematics – Assignment 1**
This repository contains my solutions for **AIDI 1000 – AI Algorithms & Mathematics Assignment 1**. The assignment covers probability, statistics, and mathematical concepts essential for AI and machine learning applications.

## **📌 Assignment Overview**
The assignment includes a mix of probability theory, statistical modeling, and data analysis using Python. The key topics covered are:
- **Contingency Tables & Probability Calculations**
- **Binomial & Conditional Probability**
- **Normal Distribution & Z-Scores**
- **Bayesian Probability**
- **Spam Classification Using Probabilistic Models**

## **📊 Key Learning Outcomes**
This assignment strengthens **AI and statistical analysis** skills, covering:
- **Understanding probability distributions** and computing statistical probabilities.
- **Applying Bayesian reasoning** to make predictions.
- **Building basic spam detection models** using probability-based classification.
- **Visualizing data distributions** to interpret probability results.

## **📷 Example Visualization**
Here is an example of a probability distribution plot generated in this assignment:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, 0, 1), label="Standard Normal Distribution")
plt.axvline(x=0, color='r', linestyle='--', label="Mean (μ)")
plt.axvline(x=1, color='g', linestyle='--', label="One Std Dev (σ)")
plt.legend()
plt.title("Probability Density Function of Normal Distribution")
plt.xlabel("X")
plt.ylabel("Density")
plt.show()
```

## **✍️ Author**
```yaml
Name: Fanmei Wang
```

## **🔗 Next Steps**
```markdown
- Try experimenting with different datasets and probability distributions.
- Compare Bayesian models with other statistical classification methods.
- Extend this project by integrating it with real-world HR analytics scenarios.
```
