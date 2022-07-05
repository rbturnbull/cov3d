================================================================
cov3d
================================================================

.. start-badges

|fastapp badge|

.. |testing badge| image:: https://github.com/rbturnbull/cov3d/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/cov3d/actions

.. |docs badge| image:: https://github.com/rbturnbull/cov3d/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/cov3d
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/e5215101db772c68108372edc5f6519b/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/cov3d/coverage/

.. |fastapp badge| image:: https://img.shields.io/badge/MLOps-fastapp-fuchsia.svg
    :target: https://github.com/rbturnbull/fastapp
    
.. end-badges

A deep learning model to detect the presence and severity of COVID19 in patients from CT-scans. 
It has been submitted as part of the workshop `'AI-enabled Medical Image Analysis – Digital Pathology & Radiology/COVID19 (MIA-COV19D)' <https://mlearn.lincoln.ac.uk/eccv-2022-ai-mia/>`_
at the European Conference on Computer Vision (ECCV) in 2022.

.. warning::

    This project is part of a computer vision competition and has not been released for clinical use.

Installation
==================================

Install cov3d with pip:

.. code:: bash

    pip install git+https://github.com/rbturnbull/cov3d.git

Training
==================================

To train cov3d, use this command:

.. code:: bash

    cov3d train --directory <path/to/uncompressed/training/files> --directory ../ --training-csv train_partition_covid_categories.csv --validation-csv val_partition_covid_categories.csv

The training data and the two CSV files are part of the COV19-CT-DB Database which is available from MIA-COV19D workshop team.

More options for training the model can be seen with the command:

.. code:: bash

    cov3d train --help

Inference
==================================

If you have a trained model, you can infer the presence and severity of COVID19 from a CT scan with the command:

.. code:: bash

    cov3d infer --pretrained <path/to/model> --scan <path/to/ct-scan>

More options for inference can be found with the command:

.. code:: bash

    cov3d infer --help


.. Further information
.. ==================================

.. Read the paper for more information: 


Credits
==================================

* Robert Turnbull <robert.turnbull@unimelb.edu.au>
* Created using fastapp (https://github.com/rbturnbull/fastapp)

