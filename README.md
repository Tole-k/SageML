# SageML
[![PyPI - Downloads](https://img.shields.io/pypi/dm/SageML)](https://pypi.org/project/SageML/)

**SageML** is an out-of-the-box AutoML solution designed to simplify the machine learning workflow. With minimal user input, SageML automates model selection, hyperparameter optimization, and provides a trained machine learning model ready for deployment.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection)
  - [Hyperparameter Optimization](#hyperparameter-optimization)
  - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Automatic Model Selection**: Chooses the best algorithm based on data characteristics with pre-trained neural network.
- **Hyperparameter Optimization**: Utilizes Optuna for efficient hyperparameter tuning.
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling automatically.
- **Interactive Interface**: User-friendly terminal interface with tutorials and step-by-step guidance.
- **Extensibility**: Modular architecture allows for easy customization and extension.
- **Compatibility**: Supports a wide range of algorithms from scikit-learn, CatBoost, XGBoost, and more.

## Installation

SageML is available on PyPI. You can install it using `pip`:

```bash
pip install sageml
```

> **Note**: For the latest features and updates, you might want to install from the GitHub repository.

## Quick Start

Here's how you can get started with SageML in just a few lines of code:

```python
from sageml import SageML
import pandas as pd
# Initialize SageML with your dataset
sageml = SageML(pd.read_csv('classified/data.csv'), target='target')

# Make predictions
predictions = sageml.predict(pd.read_csv('not/classified/data.csv'))
```

## Usage

### Data Preprocessing

SageML automatically preprocesses your data to make it suitable for machine learning algorithms.

- Handles missing values with appropriate imputation methods.
- Encodes categorical variables using techniques like One-Hot Encoding.
- Scales numerical features for algorithms sensitive to feature scales.

### Model Selection

- Analyzes data characteristics (e.g., number of features, class balance).
- Selects suitable algorithms from a pool that includes scikit-learn classifiers/regressors, CatBoost, XGBoost, etc.
- Supports both classification and regression tasks.

### Hyperparameter Optimization

- Utilizes Optuna for efficient hyperparameter optimization.
- Employs advanced features like pruning to reduce computation time.

### Model Evaluation

- Allows selection of evaluation metrics from scikit-learn or custom weighted sums.
- Supports cross-validation and hold-out validation strategies.

## Documentation

Detailed documentation should be available soon.

## Contributing

We welcome contributions from the community!

- **Bug Reports & Feature Requests**: Use the [GitHub Issues](https://github.com/Tole-k/SageML/issues) to report bugs or suggest features.

## License

SageML is licensed under the [GNU General Public License v3.0](LICENSE).

---

*Disclaimer: This project is under active development. Features and interfaces are subject to change.*
