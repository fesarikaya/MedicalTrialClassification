Installation Guide
================

Prerequisites
------------

* Python 3.8+
* pip
* virtualenv (recommended)

Installation Steps
----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://gitlab.com/ferhat.sarikaya/ml-recruitment/ml-recruitment.git
      cd ml-recruitment

2. Create and activate virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

   .. code-block:: bash

      python environment_setup.py

4. Verify installation:

   .. code-block:: bash

      python -m pytest tests/