.. image:: https://raw.githubusercontent.com/spring-epfl/trickster/master/trickster.svg?sanitize=true
   :width: 100px
   :alt: Trickster

=========
trickster
=========

|travis| |docs|

.. |docs| image:: https://readthedocs.org/projects/trickster-lib/badge/?version=latest
   :target: https://trickster-lib.readthedocs.io/en/latest/
   :alt: Docs

.. |travis| image:: https://travis-ci.org/spring-epfl/trickster.svg?branch=master
   :target: https://travis-ci.org/spring-epfl/trickster
   :alt: Travis

.. description-marker-do-not-remove

Library and experiments for attacking machine learning in discrete domains `using graph search
<https://arxiv.org/abs/1810.10939>`__.

.. end-description-marker-do-not-remove

See the `documentation <https://trickster-lib.readthedocs.io/en/latest/>`__ on Readthedocs, or jump
directly to the `guide <https://trickster-lib.readthedocs.io/en/latest/guide.html>`__.

Setup
=====

Library
-------

.. lib-setup-marker-do-not-remove

Install the trickster library as a Python package:

::

    pip install -e git+git://github.com/spring-epfl/trickster#egg=trickster

Note that trickster requires Python **3.6**.

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

- UCI `German credit dataset <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>`__
- Zafar Gilani's `Twitter bot classification dataset <https://www.cl.cam.ac.uk/~szuhg2/data.html>`__
- Tao Wang's `knndata <http://home.cse.ust.hk/~taow/wf/data/>`__

.. end-exp-setup-marker-do-not-remove

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
