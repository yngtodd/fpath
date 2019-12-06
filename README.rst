=====
FPATH
=====

Fpath, short for fast path, is a library for deep learning models 
on clinical pathology reports. Its focus is speed, therefore we 
hold off on the Python runtime and keep with the C++ interface of Pytorch.

Building
========

Application
-----------
.. code-block:: console

    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
    cmake --build .
    

Documentation (Doxygen)
-----------------------
.. code-block:: console

    cd doc/doxygen
    doxygen Doxyfile
 
    
Documenation (Sphinx)
---------------------
.. _Breathe: https://breathe.readthedocs.io/en/latest/

.. code-block:: console

    $ sphinx-build -b html doc/sphinx doc/sphinx/_build/html
    
Doxygen documentation can be integreted into Sphinx using the `Breathe`_
extension.


Running
=======

Application
-----------

.. code-block:: console

    cd build
    ./fpath


Test Suite
----------
.. code-block:: console

    cd build
    test/fpath_test
