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

    $ mkdir build && cd build
    $ cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
    $ cmake --build .
    

Documentation (Doxygen)
-----------------------
.. code-block:: console

    $ cd doc/doxygen
    $ doxygen Doxyfile
 
    
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

1. `kernel_size`: the size of the kernel filter (square)
2. `num_epochs`: the number of epochs to pass over the data
3. `batch_size`: minibatch size
4. `data_path`: path to the MNIST data

.. code-block:: console

    $ cd build
    $ ./fpath --kernel_size <int> --num_epochs <int> --batch_size <int> --data_path <string>


Test Suite
----------
.. code-block:: console

    $ cd build
    $ test/fpath_test
