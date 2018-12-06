Some development notes
======================

Running tests
-------------

::

    export PYTHONPATH=.
    pytest

Code formatting
---------------

The codebase is formatted using `black <https://github.com/ambv/black>`__.
Install black using pipsi (just pip will also work):

::

    pipsi install black

Run the following to format the code:

::

    make format
