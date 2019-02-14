
.. _config-file:

Configuration File
===================
File containig the information used by the system in order to estimate and store depth-maps from a sequence of images. The file must contain a single **JSON** objects; the fields of said objects are then used by the system during execution.

.. note::
	Some of these parameters are required by the system and must be present in the **JSON** object.

Required Parameters
-------------------
The parameters that **must** be in the configuration file are

+-------------------------+-----------------------+---------------+
| **Required Parameters** | **Description**       |   **type**    | 
+=========================+=======================+===============+
|                         | File containing the   |  **string**   |
|    "camera_file"        | the camera matrices   |               |
|                         | (:ref:`camfile`).     |               |
+-------------------------+-----------------------+---------------+
|                         | Directory containing  |  **string**   |
|   "pictures_directory"  | the images used       |               |
|                         | during depth-map      |               |
|                         | estimation            |               |
|                         | (:ref:`picdir`).      |               |
+-------------------------+-----------------------+---------------+
|   "output_directory"    | Directory in which the|  **string**   |
|                         | output depth-map will |               |
|                         | be saved (            |               |
|                         | :ref:`outdir`).       |               |
+-------------------------+-----------------------+---------------+
|"pictures_file_extension"| File extension used   |   **string**  |
|                         | by the images in the  |               |
|                         | pictures directory.   |               |
+-------------------------+-----------------------+---------------+
|     "height"            | Height of the output  |   **int**     |
|                         | depth-maps. It can be |               |
|                         | different from the    |               |
|                         | height of the original|               |
|                         | image.                |               |
|                         |                       |               |
+-------------------------+-----------------------+---------------+
|     "width"             | Width of the output   |   **int**     |
|                         | depth-maps. It can be |               |
|                         | different from the    |               |
|                         | width  of the original|               |
|                         | image.                |               |
+-------------------------+-----------------------+---------------+
|    "start_frame"        |starting frame from    |    **int**    |
|                         |which estimate         |               |
|                         |depth-maps.            |               |
+-------------------------+-----------------------+---------------+
|    "end_frame"          |Ending frame for       |   **int**     |
|                         |depth-map estimation.  |               |
|                         |                       |               |
+-------------------------+-----------------------+---------------+
|     "depth_min"         |Minimum depth-value    |    **float**  |
|                         |admissible by the      |               |
|                         |system.                |               |
+-------------------------+-----------------------+---------------+
|     "depth_max"         |Maximum depth-value    |  **float**    |
|                         |admissible by the      |               |
|                         |system.                |               |
+-------------------------+-----------------------+---------------+
|     "depth_levels"      |Number of discrete     |   **int**     |
|                         |depth labels used      |               |
|                         |during estimation.     |               |
|                         |These labels are       |               |
|                         |distribuited uniformely|               |
|                         |between the values     |               |
|                         |*"depth_min"*  and     |               | 
|                         |*"depth_max"*.         |               |
+-------------------------+-----------------------+---------------+

Optional Parameters
-------------------
The remaining optional parameters are

+-------------------------+-------------------------------------------+---------------+
| **Optional Parameters** | **Description**                           |   **type**    | 
+=========================+===========================================+===============+
|                         |Directory containing                       |  **string**   |
|  "depthmaps_directory"  |the depth-maps used                        |               |
|                         |during *bundle*                            |               |
|                         |*optimization*                             |               |
|                         |(:ref:`depdir`).                           |               |
+-------------------------+-------------------------------------------+---------------+
|    "window_side"        |Size of the window used when estimating    |    **int**    |
|                         |photo/geometry consistency of pixels in    |               |
|                         |:func:`compute_energy.compute_energy_data`.|               |
|                         |A bigger window size means a greater number|               |
|                         |of samples.                                |               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+
|                         |Parameter used in function                 |  **float**    |
|   "sigma_c"             |:func:`compute_energy.compute_energy_data`.|               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+
|   "sigma_d"             |Parameter used in function                 |  **float**    |
|                         |:func:`compute_energy.compute_energy_data`.|               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+
|     "eta"               |Parameter used in function                 |   **float**   |
|                         |:func:`lbp.lbp`.                           |               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+
|     "w_s"               |Parameter used in function                 |   **float**   |
|                         |:func:`compute_energy.lambda_factor`.      |               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+
|     ""epsilon""         |Parameter used in function                 |   **float**   |
|                         |:func:`compute_energy.lambda_factor`.      |               |
|                         |                                           |               |
+-------------------------+-------------------------------------------+---------------+

