=========
trickster
=========

.. description-marker-do-not-remove

Library and experiments for attacking machine learning in discrete domains.

This is an accompanying code to the paper "`Evading classifiers in discrete domains with provable
optimality guarantees <https://arxiv.org/abs/1810.10939>`__" by B. Kulynych, J. Hayes, N. Samarin,
and C. Troncoso, 2018. Cite as follows:

.. code-block:: bibtex

    @article{KulynychHST18,
      author    = {Bogdan Kulynych and
                   Jamie Hayes and
                   Nikita Samarin and
                   Carmela Troncoso},
      title     = {Evading classifiers in discrete domains with provable optimality guarantees},
      journal   = {CoRR},
      volume    = {abs/1810.10939},
      year      = {2018},
      url       = {http://arxiv.org/abs/1810.10939},
      archivePrefix = {arXiv},
      eprint    = {1810.10939},
    }

.. end-description-marker-do-not-remove

Setup
=====

Library
-------

.. lib-setup-marker-do-not-remove

Install the trickster library as a Python package:

::

    pip install -e git+git://github.com/spring-epfl/trickster#egg=trickster

.. end-lib-setup-marker-do-not-remove

Experiments
-----------

.. exp-setup-marker-do-not-remove

Python packages
~~~~~~~~~~~~~~~

Install the required Python packages:

::

    pip install -r requirements.txt

System packages
~~~~~~~~~~~~~~~

On Ubuntu, you need these system packages:

::

    apt install parallel unzip

Datasets
~~~~~~~~

To download the datasets, run this:

::

    make data

The datasets include:

- Zafar Gilani's `Twitter bot classification dataset <https://www.cl.cam.ac.uk/~szuhg2/data.html>`__

.. end-exp-setup-marker-do-not-remove

Development
===========

See `DEVELOPMENT.rst <DEVELOPMENT.rst>`__
