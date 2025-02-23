Quick Start Guide
===============

Basic Usage
----------

1. Start the Flask server:

   .. code-block:: bash

      python main.py

2. Make predictions:

   .. code-block:: python

      import requests
      import json

      url = "http://127.0.0.1:5000/predict"
      data = {"description": "RGrid Machine Learning Challenge"}

      response = requests.post(url, json=data)
      prediction = response.json()["prediction"]
      print(f"Predicted condition: {prediction}")

API Reference
------------

Check the :doc:`../api/index` for detailed API documentation.