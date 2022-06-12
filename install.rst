Installation
============

.. note::
    We do not recommend installation as a root user on your system Python.
    Please setup a `Anaconda or Miniconda <https://conda.io/projects/conda/en/latest/user-guide/install>`_ environment or create a `Docker image <https://www.docker.com/>`_.


We provide two ways for installation the environment: (1) one step installation from bash file; (2) step by step installation. 


One step installation from bash file 
------------------------------------

    .. code-block:: none
        
        source init.sh 


Step by step installation
-------------------------

# . Download openpoints

        .. code-block:: none
            
            git submodule update --init --recursive


#. Ensure that your CUDA is setup correctly (optional):
    #. Add CUDA to :obj:`$PATH` and :obj:`$CPATH` (note that your actual CUDA path may vary from :obj:`/usr/local/cuda`):

        .. code-block:: none
            
            # add the following to you `~/.bashrc`
            cuda=cuda-11.1
            export CUDADIR=/usr/local/$cuda
            export PATH=$CUDADIR/bin:$PATH
            export NUMBAPRO_NVVM=$CUDADIR/nvvm/lib64/libnvvm.so
            export NUMBAPRO_LIBDEVICE=$CUDADIR/nvvm/libdevice/
            export NVCCDIR=$CUDADIR/bin/nvcc
            export LD_LIBRARY_PATH=$CUDADIR/lib64:$LD_LIBRARY_PATH
            export CPATH=$CUDADIR/include:$CPATH
            export CUDA_HOME=$CUDADIR
            export CUDA_BIN_PATH=$CUDADIR


    #. Verify that :obj:`nvcc` is accessible from terminal:

        .. code-block:: none

            nvcc --version
            >>> 11.1


#. Install PyTorch
        .. code-block:: none

            conda deactivate
            conda env remove --name openpoints
            conda create -n openpoints -y python=3.7 numpy=1.20 numba
            conda activate openpoints

            conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
            
    #. Ensure that PyTorch and system CUDA versions match in the major level (both of them are 11.x):

        .. code-block:: none

            python -c "import torch; print(torch.version.cuda)"
            >>> 11.1

            nvcc --version
            >>> 11.1


#. Install relevant packages:

    .. code-block:: none


        pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
        pip install -r requirements.txt



#. Install cpp extensions: 

    .. code-block:: none

        cd openpoints/cpp/pointnet2_batch
        python setup.py install
        cd ../../../

