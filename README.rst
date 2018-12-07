.. image:: https://raw.githubusercontent.com/spring-epfl/trickster/master/trickster.svg?sanitize=true
   :width: 100px
   :alt: Trickster

=========
trickster
=========

.. description-marker-do-not-remove

Library and experiments for attacking machine learning in discrete domains `using graph search
<https://arxiv.org/abs/1810.10939>`__.

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

Citing
======

.. citing-marker-do-not-remove

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

.. end-citing-marker-do-not-remove

Acknowledgements
================

.. acks-marker-do-not-remove

This work is funded by the NEXTLEAP project within the European Union’s Horizon 2020 Framework Programme for Research and Innovation (H2020-ICT-2015, ICT-10-2015) under grant agreement 688722.

.. end-acks-marker-do-not-remove
