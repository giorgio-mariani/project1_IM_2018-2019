Quick-Start
===========

Dependencies Installation
-------------------------
The system makes use of several external libraries, which can be easily installed using **pip** and the **requirements.txt** file:

	``pip install -r requirements.txt``

.. note::

	**Python 2.7** is required in order to run the system, consequently, the appropriate version of **pip** should be used. 


Running the System
------------------
The system can be executed by invoking the following instruction from the choise operating system's command line terminal:

	``python estimate.py [-i][-b] <config.txt>``

**<config.txt>** indicates a configuration file containing parameters and file-paths necessary for the estimation process.
A more in detail explanation of such file can be found at :ref:`config-file`.
The options **-i** and **-b** are used to communicate to the system which steps should be executed:

	* **-i**, it executes only the initialization phase and stores the output depth-maps in the output folder.
	* **-b**, it executes the bundle optimization phase; it requires previously estimated depth-maps as input. 
	* **-ib**, it executes both the initialization and the bundle optimization phases.

An overview of the modules used in this project can be seen in :ref:`index`.

Inputs Used by the System
----------------------------
.. In order to work correctly the systems requires several data.

.. _camfile:

Camera File
^^^^^^^^^^^
This file contains information about the camera parameters; in particular, it stores information 
about *intrisic matrix*, *rotations*, and *translation* of the camera for each picture in the sequence.

The camera file itself is quite simple; for each image the intrinsic matrix, 
rotation matrix, and position must be included in plain text, one after the other, without 
interleaving blank lines.

The code below shows an example of such format: the first three rows are the *intrinsic matric*,
the second three rows are the *rotation* matrix, and the final row is the *position*.
::

     1139.7929     0.0000000     479.50000
     0.0000000     1139.7929     269.50000
     0.0000000     0.0000000     1.0000000
     1.0000000     0.0000000     0.0000000
     0.0000000     1.0000000     0.0000000
     0.0000000     0.0000000     1.0000000
     0.0000000     0.0000000     0.0000000

This pattern must be repeated for each image in the target sequence, and two blank lines must be
used between data of different images. 

.. important::
	The first line of the camera file must be a number, indicating the number of camera frames in the
	file itself. A blank line must follow.

A complete example, using a sequence of length three is

::

     3

     1139.7929     0.0000000     479.50000
     0.0000000     1139.7929     269.50000
     0.0000000     0.0000000     1.0000000
     1.0000000     0.0000000     0.0000000
     0.0000000     1.0000000     0.0000000
     0.0000000     0.0000000     1.0000000
     0.0000000     0.0000000     0.0000000


     1139.7929     0.0000000     479.50000
     0.0000000     1139.7929     269.50000
     0.0000000     0.0000000     1.0000000
     0.9999972     -0.001567     0.0017456
     0.0015681     0.9999985     -0.000634
     -0.001744     0.0006373     0.9999982
     -0.320146     0.0141622     -0.049300


     1139.7929     0.0000000     479.50000
     0.0000000     1139.7929     269.50000
     0.0000000     0.0000000     1.0000000
     0.9999884     -0.002738     0.0039581
     0.0027459     0.9999945     -0.001806
     -0.003953     0.0018177     0.9999905
     -0.720470     -0.028543     -0.089562

The location of the camera file is defined by the configuration parameter 
``"camera_file"``.


.. _picdir:

Pictures Directory
^^^^^^^^^^^^^^^^^^
Directory which contains the picture sequence to be used during the estimation process.
The images must be named using the syntax ``img_<num>``, with ``<num>`` numeric string 
which assumes values from ``0000`` to ``9999``. The images filenames must be organized in a 
contiguous fashion: no gaps should be present between the smallest and largest filename numbers.
All the image files (used for estimation) in the directory should have same format and resolution.

The image format, as well as the location of the directory are indicated by the configuration parameters
``"pictures_file_extension"`` and ``"pictures_directory"`` respectively. 

.. _depdir:

Depth-maps Directory
^^^^^^^^^^^^^^^^^^^^
Directory containing a sequence of depth-maps. These can then be used during execution of 
the function :func:`compute_energy.compute_energy_data` to perform the *bundle optimization* step.

.. note::

  Depth-maps directories are generally the result of a previous invokation of the system.

These depth-maps are stored as **.npy** files.
Depth-maps stored inside this directory must be named ``depth_<num>``, with ``<num>`` numeric string 
which assumes values from ``0000`` to ``9999``. As with the images, the depth-maps filenames must be organized in a 
contiguous fashion: no gaps should be present between the smallest and largest filename numbers.

The location of the directory is indicated by the configuration parameter ``"depthmaps_directory"``.

.. _outdir:

Outputs of the System
---------------------
The output of the system is a directory containing the estimated depth-maps.
Names and format of such files follows the same convention as described in :ref:`depdir`.
The location of the output directory is indicated by the configuration parameter ``"output_directory"``.


