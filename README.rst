|SLURM| |preempt| |PyTorch| |wandb| |pre-commit| |black|

PyTorch Experiment Template
===========================

This repository gives a fully-featured template or skeleton for new PyTorch
experiments for use on the Vector Institute cluster.
It supports:

- multi-node, multi-GPU jobs using DistributedDataParallel (DDP),
- preemption handling which gracefully stops and resumes the job,
- logging experiments to Weights & Biases,
- configuration seemlessly scales up as the amount of resources increases.

If you want to run the example experiment, you can clone the repository as-is
without modifying it. Then see the `Installation`_ and
`Executing the example experiment`_ sections below.

If you want to create a new repository from this template, you should follow
the `Creating a git repository using this template`_ instructions first.


Creating a git repository using this template
---------------------------------------------

.. highlight:: bash

When creating a new repository from this template, these are the steps to follow:

#. *Don't click the fork button.*
   The fork button is for making a new template based in this one, not for using the template to make a new repository.

#.
    #.  **New GitHub repository**.

        You can create a new repository on GitHub from this template by clicking the `Use this template <https://github.com/scottclowe/pytorch-experiment-template/generate>`_ button.

        Then clone your new repository to your local system [pseudocode]::

          git clone git@github.com:your_org/your_repo_name.git
          cd your_repo_name

    #.  **New repository not on GitHub**.

        Alternatively, if your new repository is not going to be on GitHub, you can download `this repo as a zip <https://github.com/scottclowe/pytorch-experiment-template/archive/master.zip>`_ and work from there.

        Note that this zip does not include the .gitignore and .gitattributes files (because GitHub automatically omits them, which is usually helpful but is not for our purposes).
        Thus you will also need to download the `.gitignore <https://raw.githubusercontent.com/scottclowe/pytorch-experiment-template/master/.gitignore>`__ and `.gitattributes <https://raw.githubusercontent.com/scottclowe/pytorch-experiment-template/master/.gitattributes>`__ files.

        The following shell commands can be used for this purpose on \*nix systems::

          git init your_repo_name
          cd your_repo_name
          wget https://github.com/scottclowe/pytorch-experiment-template/archive/master.zip
          unzip master.zip
          mv -n pytorch-experiment-template-master/* pytorch-experiment-template-master/.[!.]* .
          rm -r pytorch-experiment-template-master/
          rm master.zip
          wget https://raw.githubusercontent.com/scottclowe/pytorch-experiment-template/master/.gitignore
          wget https://raw.githubusercontent.com/scottclowe/pytorch-experiment-template/master/.gitattributes
          git add .
          git commit -m "Initial commit"
          git rm LICENSE

        Note that we are doing the move with ``mv -n``, which will prevent the template repository from clobbering your own files (in case you already made a README.rst file, for instance).

        You'll need to instruct your new local repository to synchronise with the remote ``your_repo_url``::

          git remote set-url origin your_repo_url
          git push -u origin master

#.  Delete the LICENSE file and replace it with a LICENSE file of your own choosing.
    If the code is intended to be freely available for anyone to use, use an `open source license`_, such as `MIT License`_ or `GPLv3`_.
    If you don't want your code to be used by anyone else, add a LICENSE file which just says:

    .. code-block:: none

        Copyright (c) CURRENT_YEAR, YOUR_NAME

        All rights reserved.

    Note that if you don't include a LICENSE file, you will still have copyright over your own code (this copyright is automatically granted), and your code will be private source (technically nobody else will be permitted to use it, even if you make your code publicly available).

#.  Edit the file ``template_experiment/__meta__.py`` to contain your author and repo details.

    name
        The name as it would be on PyPI (users will do ``pip install new_name_here``).
        It is `recommended <PEP-8_>`__ to use a name all lowercase, runtogetherwords but if separators are needed hyphens are preferred over underscores.

    path
        The path to the package. What you will rename the directory ``template_experiment``.
        `Should be <PEP-8_>`__ the same as ``name``, but now hyphens are disallowed and should be swapped for underscores.
        By default, this is automatically inferred from ``name``.

    license
        Should be the name of the license you just picked and put in the LICENSE file (e.g. ``MIT`` or ``GPLv3``).

    Other fields to enter should be self-explanatory.

#.  Rename the directory ``template_experiment`` to be the ``path`` variable you just added to ``__meta__.py``.::

      # Define PROJECT_HYPH as your actual project name (use hyphens instead of underscores or spaces)
      PROJECT_HYPH=your-actual-project-name-with-hyphens-for-spaces

      # Automatically convert hyphens to underscores to get the directory name
      PROJECT_DIRN="${PROJECT_HYPH//-/_}"
      # Rename the directory
      mv template_experiment "$PROJECT_DIRN"

#.  Change references to ``template_experiment`` and ``template-experiment``
    to your path variable.

    This can be done with the sed command::

        sed -i "s/template_experiment/$PROJECT_DIRN/" \
            "$PROJECT_DIRN/*.py" setup.py docs/source/conf.py
        sed -i "s/template-experiment/$PROJECT_HYPH/" \
            "$PROJECT_DIRN/*.py" slurm/*.slrm

    Which will make changes in the following places.

    .. highlight:: python

    - In ``setup.py``, `L51 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/setup.py#L51>`__::

        exec(read("template_experiment/__meta__.py"), meta)

    - In ``__meta__.py``, `L2,4 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/__meta__.py#L2-4>`__::

        name = "template-experiment"

    - In ``docs/source/conf.py``, `L27 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/docs/source/conf.py#L27>`__::

        from template_experiment import __meta__ as meta  # noqa: E402 isort:skip

    - In ``train.py``, `L17-18 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/train.py#L17-18>`__::

        from template_experiment import data_transformations, datasets, encoders, utils
        from template_experiment.evaluation import evaluate

    - In ``train.py``, `L1149 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/train.py#L1149>`__::

        group.add_argument(
            "--wandb-project",
            type=str,
            default="template-experiment",
            help="Name of project on wandb, where these runs will be saved.",
        )

    - In ``slurm/train.slrm``, `L16 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/train.slrm#L16>`__::

        #SBATCH --job-name=template-experiment    # Set this to be a shorthand for your project's name.

    - In ``slurm/train.slrm``, `L20 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/train.slrm#L20>`__::

        PROJECT_NAME="template-experiment"

    - In ``slurm/notebook.slrm``, `L16 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/notebook.slrm#L16>`__::

        PROJECT_NAME="template-experiment"

    .. highlight:: bash

#.  Swap out the contents of ``README.rst`` with an initial description of your project.
    If you prefer, you can use markdown (``README.md``) instead of rST.::

      git rm README.rst
      # touch README.rst
      touch README.md

#.  Commit and push your changes::

      git add .
      git commit -m "Initialise project from template repository"
      git push

.. _PEP-8: https://www.python.org/dev/peps/pep-0008/
.. _open source license: https://choosealicense.com/
.. _MIT License: https://choosealicense.com/licenses/mit/
.. _GPLv3: https://choosealicense.com/licenses/gpl-3.0/


Installation
------------

I recommend using miniconda to create an environment for your project.
By using one virtual environment dedicated to each project, you are ensured
stability - if you upgrade a package for one project, it won't affect the
environments you already have established for the others.

Vector one-time set-up
~~~~~~~~~~~~~~~~~~~~~~

Run this code block to install miniconda before you make your first environment
(you don't need to re-run this every time you start a new project).::

    # Login to Vector
    ssh USERNAME@v.vectorinstitute.ai
    # Enter your password and 2FA code to login.
    # Run the rest of this code block on the gateway node of the cluster that
    # you get to after establishing the ssh connection.

    # Make a screen session for us to work in
    screen;

    # Download miniconda to your ~/Downloads directory
    mkdir -p $HOME/Downloads;
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O "$HOME/Downloads/miniconda.sh";
    # Install miniconda to the home directory, if it isn't there already.
    if [ ! -d "$HOME/miniconda/bin" ]; then
        if [ -d "$HOME/miniconda" ]; then rm -r "$HOME/miniconda"; fi;
        bash $HOME/Downloads/miniconda.sh -b -p "$HOME/miniconda";
    fi;

    # Add conda to the PATH environment variable
    export PATH="$HOME/miniconda/bin:$PATH";

    # Automatically say yes to any check from conda (optional)
    conda config --set always_yes yes

    # Set the command prompt prefix to be the name of the current venv
    conda config --set env_prompt '({name}) '

    # Add conda setup to your ~/.bashrc file
    conda init;

    # Now exit this screen session (you have to exit the current terminal
    # session after conda init, and exiting the screen session achieves that
    # without closing the ssh connection)
    exit;

Follow this next step if you want to use `Weights and Biases`_ to log your experiments.
Weights and Biases is an online service for tracking your experiments which is
free for academic usage.
To set this up, you need to install the wandb pip package, and you'll need to
`create a Weights and Biases account <wandb-signup_>`_ if you don't already have one.::

    # (On v.vectorinstitute.ai)
    # You need to run the conda setup instructions that miniconda added to
    # your ~/.bashrc file so that conda is on your PATH and you can run it.
    # Either create a new screen session - when you launch a new screen session,
    # bash automatically runs source ~/.bashrc
    screen;
    # Or stay in your current window and explicitly yourself run
    source ~/.bashrc
    # Either way, you'll now see "(miniconda)" at the left of your command prompt,
    # indicating miniconda is on your PATH and using your default conda environment.

    # Install wandb
    pip install wandb

    # Log in to wandb at the command prompt
    wandb login
    # wandb asks you for your username, then password
    # Then wandb creates a file in ~/.netrc which it uses to automatically login in the future

.. _Weights and Biases: https://wandb.ai/
.. _wandb-signup: https://wandb.ai/login?signup=true


Project one-time set-up
~~~~~~~~~~~~~~~~~~~~~~~

Run this code block once every time you start a new project from this template.
Change ENVNAME to equal the name of your project. This code will then create a
new virtual environment to use for the project.::

    # (On v.vectorinstitute.ai)
    # You need to run the conda setup instructions that miniconda added to
    # your ~/.bashrc file so that conda is on your PATH and you can run it.
    # Either create a new screen session - when you launch a new screen session,
    # bash automatically runs source ~/.bashrc
    screen;
    # Or stay in your current window and explicitly yourself run
    source ~/.bashrc
    # Either way, you'll now see "(miniconda)" at the left of your command prompt,
    # indicating miniconda is on your PATH and using your default conda environment.

    # Now run the following one-time setup per virtual environment (i.e. once per project)

    # Pick a name for the new environment.
    # It should correspond to the name of your project (hyphen separated, no spaces)
    ENVNAME=template-experiment

    # Create a python3.x conda environment, with pip installed, with this name.
    conda create -y --name "$ENVNAME" -q python=3 pip

    # Activate the environment
    conda activate "$ENVNAME"
    # The command prompt should now have your environment at the left of it, e.g.
    # (template-experiment) slowe@v3:~$


Resuming work on an existing project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this code block when you want to resume work on an existing project.::

    # (On v.vectorinstitute.ai)
    # Run conda setup in ~/.bashrc if you it hasn't already been run in this
    # terminal session
    source ~/.bashrc
    # The command prompt should now say (miniconda) at the left of it.

    # Activate the environment
    conda activate template-experiment
    # The command prompt should now have your environment at the left of it, e.g.
    # (template-experiment) slowe@v3:~$


Executing the example experiment
--------------------------------

The following commands describe how to setup and run the example repository
in its unmodified state.

To run the code in a repository you have
`created from this template <Creating a git repository using this template_>`_,
replace ``template-experiment`` with the name of your package and
``template_experiment`` with the name of your package directory, etc.

Set-up
~~~~~~

#. If you haven't already, then follow the Vector one-time set-up as above.

#. Then clone the repository::

        git clone git@github.com:scottclowe/pytorch-experiment-template.git
        cd pytorch-experiment-template

#. Run the project one-time set-up, as above (using template-experiment as the
   environment name).

#. With the project's conda environment activated, install the package and its
   training dependencies::

        pip install --editable .[train]

   This step will typically take 5-10 minutes to run.

#. Check the installation by running the help command::

        python template_experiment/train.py -h

   This should print the help message for the training script.


Example commands
~~~~~~~~~~~~~~~~

- To run the default training command locally::

        python template_experiment/train.py

- Run the default training command with on the cluster with SLURM.
  First, ssh into the cluster and cd to the project repository.
  You don't need to activate the project's conda environment.
  Then use sbatch to add your SLURM job to the queue::

        sbatch slurm/train.slrm

- You can supply arguments to sbatch by including them before the path to the
  SLURM script.
  Arguments set on the command prompt like this will override the arguments in
  ``slurm/train.slrm``.
  This is useful for customizing the job name, for example::

        sbatch --job-name=exp_cf10_rn18 slurm/train.slrm

  I recommend you should pretty much always customize the name of your job.
  The custom job name will be visible in the output of ``squeue -u "$USER"``
  when browsing your active jobs (helpful if you have multiple jobs running
  and need to check on their status or cancel one of them).
  When using this codebase, the custom job name is also used in the path to the
  checkpoint, the path to the SLURM log file, and the name of the job on wandb.

- Any arguments you include after ``slurm/train.slrm`` will be passed through to train.py.

  For example, you can specify to use a pretrained model::

        sbatch --job-name=exp_cf10_rn18-pt slurm/train.slrm --dataset=cifar10 --pretrained

  change the architecture and dataset::

        sbatch --job-name=exp_cf100_vit-pt \
            slurm/train.slrm --dataset=cifar100 --model=vit_small_patch16_224 --pretrained

  or change the learning rate of the encoder::

        sbatch --job-name=exp_cf10_rn18-pt_enc-lr-0.01 \
            slurm/train.slrm --dataset=cifar10 --pretrained --lr-encoder-mult=0.01

- You can trivially scale up the job to run across multiple GPUs, either by
  changing the gres argument to use more of the GPUs on the node (up to 8 GPUs
  per node on the t4v2 partition, 4 GPUs per node otherwise)::

        sbatch --job-name=exp_cf10_rn18-pt_4gpu --gres=gpu:4 slurm/train.slrm --pretrained

  or increasing the number of nodes being requested.::

        sbatch --job-name=exp_cf10_rn18-pt_2x1gpu --nodes=2 slurm/train.slrm --pretrained

  or both::

        sbatch --job-name=exp_cf10_rn18-pt_2x4gpu --nodes=2 --gres=gpu:4 slurm/train.slrm --pretrained

  In each case, the amount of memory and CPUs requested in the SLURM job will
  automatically be scaled up with the number of GPUs requested.
  The total batch size will be scaled up by the number of GPUs requested too.

As you run these commands, you can see the results logged on wandb at
https://wandb.ai/your-username/template-experiment


Jupyter notebook
~~~~~~~~~~~~~~~~

You can use the script ``slurm/notebook.slrm`` to launch a Jupyter notebook
server on one of the interactive compute nodes.
This uses the methodology of https://support.vectorinstitute.ai/jupyter_notebook

You'll need to install jupyter into your conda environment to launch the notebook.
After activating the environment for this project, run::

    pip install -r requirements-notebook.txt

To launch a notebook server and connect to it on your local machine, perform
the following steps.

#. Run the notebook SLURM script to launch the jupyter notebook::

        sbatch slurm/notebook.slrm

   The job will launch on one of the interactive nodes, and will acquire a
   random port on that node to serve the notebook on.

#. Wait for the job to start running. You can monitor it with::

        squeue -u "$USER"

   Note the job id of the notebook job. e.g.:

   .. code-block:: none

        (template-experiment) slowe@v2:~/pytorch-experiment-template$ squeue -u "$USER"
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          10618891 interacti      jnb    slowe  R       1:07      1 gpu026

   Here we can see our JOBID is 10618891, and it is running on node gpu026.

#. Inspect the output of the job with::

        cat jnb_JOBID.log

   e.g.::

        cat jnb_10618891.log

   The output will contain the port number that the notebook server is using,
   and the token as follows:

   .. code-block:: none

        To access the server, open this file in a browser:
            file:///ssd005/home/slowe/.local/share/jupyter/runtime/jpserver-7885-open.html
        Or copy and paste one of these URLs:
            http://gpu026:47201/tree?token=f54c10f52e3dad08e19101149a54985d1561dca7eec96b29
            http://127.0.0.1:47201/tree?token=f54c10f52e3dad08e19101149a54985d1561dca7eec96b29

   Here we can see the job is on node gpu026 and the notebook is being served
   on port 47201.
   We will need to use the token f54c10f52e3dad08e19101149a54985d1561dca7eec96b29
   to log in to the notebook.

#. On your local machine, use ssh to forward the port from the compute node to
   your local machine::

        ssh USERNAME@v.vectorinstitute.ai -N -L 8887:gpu026:47201

   You need to replace USER with your Vector username, gpu026 with the node
   your job is running on, and 47201 with the port number from the previous
   step.
   In this example, the local port being forwarded to is 8887.

#. Open a browser on your local machine and navigate to http://localhost:8887
   (or whatever port you chose in the previous step).::

        sensible-browser http://localhost:8887

   You should see the Jupyter notebook interface.
   Copy the token from the URL shown in the log file and paste it into the
   ``"Password or token: [ ] Log in"`` box.

#. Once you are done, make sure to end the job running the notebook with::

        scancel JOBID

   e.g.::

        scancel 10618891

   This will free up the interactive GPU node for other users to use.

Note that you can skip the need to copy the access token if you
`set up Jupyter notebook to use a password <jnb-password_>`_ instead.

.. _jnb-password: https://saturncloud.io/blog/how-to-autoconfigure-jupyter-password-from-command-line/


Features
--------

This template includes the following features.


Scalable training script
~~~~~~~~~~~~~~~~~~~~~~~~

The SLURM training script ``slurm/train.slrm`` will interface with the python
training script ``template_experiment/train.py`` to train a model on multiple
GPUs across, multiple nodes, using DistributedDataParallel_ (DDP).

The SLURM script is configured to scale up the amount of RAM and CPUs requested
with the GPUs requested.

The arguments to the python script control the batch size per GPU, and the
learning rate for a fixed batch size of 128 samples.
The total batch size will automatically scale up when deployed on more GPUs,
and the learning rate will automatically scale up linearly with the total batch
size. (This is the linear scaling rule from `Training ImageNet in 1 Hour`_.)

.. _DistributedDataParallel: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
.. _Training ImageNet in 1 Hour: https://arxiv.org/abs/1706.02677


Preemptable
~~~~~~~~~~~

Everything is set up to resume correctly if the job is interrupted by
preemption.


Checkpoints
~~~~~~~~~~~

The training script will save a checkpoint every epoch, and will resume from
this if the job is interrupted by preemption.

The checkpoint for a job will be saved to the directory
``/checkpoint/USERNAME/PROJECT_JOBNAME_JOBID``, along with a record of the
conda environment and frozen pip requirements.

Any print statements from the training script will be saved to the file
``slogs/JOBNAME_JOBID-ARRAYID_0-0.out``.


Weights and Biases
~~~~~~~~~~~~~~~~~~

`Weights and Biases`_ (wandb) is an online service for tracking your
experiments which is free for academic usage.

This template repository is set up to automatically log your experiments, using
the same job label across both SLURM and wandb.

If the job is preempted, the wandb logging will resume to the same wandb job
ID instead of spawning a new one.


RNG state
~~~~~~~~~

All RNG states are configured based on the overall seed that is set with the
``--seed`` argument to ``train.py``.
The default is ``--seed=0``, so all experiments will be reproducible by default.

When running on SLURM with slurm/train.slrm, the seed is set to equal the
`job array ID <slurm-job-array_>`_. This lets you run the same job with multiple
seeds in one command.
Our default job array in ``slurm/train.slrm`` is ``--array=0``, so only one job
will be launched, and that job will use the default seed of ``0``.

To launch the same job 5 times, each with a different seed (0, 1, 2, 3, and 4)::

    sbatch --array=0-4 slurm/train.slrm

or to use seeds 42 and 888::

    sbatch --array=42,888 slurm/train.slrm

The seed is used to set the following RNG states:

- Each epoch gets its own RNG seed (derived from the overall seed and the epoch
  number).
  The RNG state is set with this seed at the start of each epoch. This makes it
  possible to resume from preemption without needing to save all the RNG states
  to the model checkpoint and restore them on resume.

- Each GPU gets its own RNG seed, so any random operations such as dropout
  or random masking in the training script itself will be different on each
  GPU, but deterministically so.

- The dataloader workers each have distinct seeds from each other for torch,
  numpy and python's random module, so randomly selected augmentations won't be
  replicated across workers.
  (Pytorch only sets up its own worker seeds correctly, leaving numpy and
  random mirrored across all workers.)

**Caution:** To get *exactly* the same model produced when training with the
same seed, you will need to run the training script with the ``--deterministic``
flag to disable cuDNN's non-deterministic operations *and* use precisely the
same number of GPU devices and CPU workers on each attempt.
Without these steps, the model will be *almost* the same (because the initial
seed for the model parameters was the same, and the training trajectory was
very similar), but not *exactly* the same, due to (a) non-deterministic cuDNN
operations (b) the batch size increasing with the number of devices
(c) any randomized augmentation operations depending on the identity of the CPU
worker, which will each have an offset seed.

.. _slurm-job-array: https://slurm.schedmd.com/job_array.html


Prototyping mode, with distinct val/test sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Initial experiments and hyperparameter searches should be performed without
seeing the final test performance. They should be run only on a validation set.
Unfortunately, many datasets do not come with a validation set, and it is easy
to accidentally use the test set as a validation set, which can lead to
overfitting the model selection on the test set.

The image datasets implemented in ``template_experiment/datasets.py`` come with
support for creating a validation set from the training set, which is separate
from the test set. You should use this (with flag ``prototyping``) during the
initial model development steps and for any hyperparameter searches.

Your final models should be trained without ``--prototyping`` enabled, so that
the full training set is used for training and the best model is produced.



.gitignore
~~~~~~~~~~

A `.gitignore`_ file is used specify untracked files which Git should ignore and not try to commit.

Our template's .gitignore file is based on the `GitHub defaults <default-gitignores_>`_.
We use the default `Python .gitignore`_, `Windows .gitignore`_, `Linux .gitignore`_, and `Mac OSX .gitignore`_ concatenated together.
(Released under `CC0-1.0 <https://github.com/github/gitignore/blob/master/LICENSE>`__.)

The Python .gitignore specifications prevent compiled files, packaging and sphinx artifacts, test outputs, etc, from being accidentally committed.
Even though you may develop on one OS, you might find a helpful contributor working on a different OS suddenly issues you a new PR, hence we include the gitignore for all OSes.
This makes both their life and yours easier by ignoring their temporary files before they even start working on the project.

.. _.gitignore: https://git-scm.com/docs/gitignore
.. _default-gitignores: https://github.com/github/gitignore
.. _Python .gitignore: https://github.com/github/gitignore/blob/master/Python.gitignore
.. _Windows .gitignore: https://github.com/github/gitignore/blob/master/Global/Windows.gitignore
.. _Linux .gitignore: https://github.com/github/gitignore/blob/master/Global/Linux.gitignore
.. _Mac OSX .gitignore: https://github.com/github/gitignore/blob/master/Global/macOS.gitignore


.gitattributes
~~~~~~~~~~~~~~

The most important reason to include a `.gitattributes`_ file is to ensure that line endings are normalised, no matter which OS the developer is using.
This is largely achieved by the line::

    * text=auto

which `ensures <gitattributes-text_>`__ that all files Git decides contain text have their line endings normalized to LF on checkin.
This can cause problems if Git misdiagnoses a file as text when it is not, so we overwrite automatic detection based on file endings for some several common file endings.

Aside from this, we also gitattributes to tell git what kind of diff to generate.

Our template .gitattributes file is based on the `defaults from Alexander Karatarakis <alexkaratarakis/gitattributes_>`__.
We use the `Common .gitattributes`_ and `Python .gitattributes`_ concatenated together.
(Released under `MIT License <https://github.com/alexkaratarakis/gitattributes/blob/master/LICENSE.md>`__.)

.. _.gitattributes: https://git-scm.com/docs/gitattributes
.. _gitattributes-text: https://git-scm.com/docs/gitattributes#_text
.. _alexkaratarakis/gitattributes: https://github.com/alexkaratarakis/gitattributes
.. _Common .gitattributes: https://github.com/alexkaratarakis/gitattributes/blob/master/Common.gitattributes
.. _Python .gitattributes: https://github.com/alexkaratarakis/gitattributes/blob/master/Python.gitattributes


Black
~~~~~

Black_ is an uncompromising Python code formatter.
By using it, you cede control over minutiae of hand-formatting.
But in return, you no longer have to worry about formatting your code correctly, since black will handle it.
Blackened code looks the same for all authors, ensuring consistent code formatting within your project.

The format used by Black makes code review faster by producing the smaller diffs.

Black's output is always stable.
For a given block of code, a fixed version of black will always produce the same output.
However, you should note that different versions of black will produce different outputs.
If you want to upgrade to a newer version of black, you must change the version everywhere it is specified:

- requirements-dev.txt, `L1 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/requirements-dev.txt#L1>`__
- .pre-commit-config.yaml, `L14 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.pre-commit-config.yaml#L14>`__,
  `L28 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.pre-commit-config.yaml#L28>`__, and
  `L47 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.pre-commit-config.yaml#L47>`__

.. _black: https://github.com/psf/black


pre-commit
~~~~~~~~~~

The template repository comes with a pre-commit_ stack.
This is a set of git hooks which are executed every time you make a commit.
The hooks catch errors as they occur, and will automatically fix some of these errors.

To set up the pre-commit hooks, run the following code from within the repo directory::

    pip install -r requirements-dev.txt
    pre-commit install

Whenever you try to commit code which is flagged by the pre-commit hooks, the commit will not go through.
Some of the pre-commit hooks (such as black_, isort_) will automatically modify your code to fix the issues.
When this happens, you'll have to stage the changes made by the commit hooks and then try your commit again.
Other pre-commit hooks will not modify your code and will just tell you about issues which you'll then have to manually fix.

You can also manually run the pre-commit stack on all the files at any time::

    pre-commit run --all-files

To force a commit to go through without passing the pre-commit hooks use the ``--no-verify`` flag::

    git commit --no-verify

The pre-commit stack which comes with the template is highly opinionated, and includes the following operations:

- Code is reformatted to use the black_ style.
  Any code inside docstrings will be formatted to black using blackendocs_.
  All code cells in Jupyter notebooks are also formatted to black using black_nbconvert_.

- All Jupyter notebooks are cleared using nbstripout_.

- Imports are automatically sorted using isort_.

- flake8_ is run to check for conformity to the python style guide PEP-8_, along with several other formatting issues.

- setup-cfg-fmt_ is used to format any setup.cfg files.

- Several `hooks from pre-commit <pre-commit-hooks_>`_ are used to screen for non-language specific git issues, such as incomplete git merges, overly large files being commited to the repo, bugged JSON and YAML files.
  JSON files are also prettified automatically to have standardised indentation.
  Entries in requirements.txt files are automatically sorted alphabetically.

- Several `hooks from pre-commit specific to python <pre-commit-py-hooks_>`_ are used to screen for rST formatting issues, and ensure noqa flags always specify an error code to ignore.

Once it is set up, the pre-commit stack will run locally on every commit.
The pre-commit stack will also run on github with one of the action workflows, which ensures PRs are checked without having to rely on contributors to enable the pre-commit locally.

.. _black_nbconvert: https://github.com/dfm/black_nbconvert
.. _blackendocs: https://github.com/asottile/blacken-docs
.. _flake8: https://gitlab.com/pycqa/flake8
.. _isort: https://github.com/timothycrosley/isort
.. _nbstripout: https://github.com/kynan/nbstripout
.. _PEP-8: https://www.python.org/dev/peps/pep-0008/
.. _pre-commit: https://pre-commit.com/
.. _pre-commit-hooks: https://github.com/pre-commit/pre-commit-hooks
.. _pre-commit-py-hooks: https://github.com/pre-commit/pygrep-hooks
.. _setup-cfg-fmt: https://github.com/asottile/setup-cfg-fmt


Automated documentation
~~~~~~~~~~~~~~~~~~~~~~~

The script ``docs/conf.py`` is based on the Sphinx_ default configuration.
It is set up to work well out of the box, with several features added in.

GitHub Pages
^^^^^^^^^^^^

If your repository is publicly available, the docs workflow will automatically deploy your documentation to `GitHub Pages`_.
To enable the documentation, go to the ``Settings > Pages`` pane for your repository and set Source to be the ``gh-pages`` branch (root directory).
Your automatically compiled documentation will then be publicly available at https://USER.github.io/PACKAGE/.

Since GitHub pages are always publicly available, the workflow will check whether your repository is public or private, and will not deploy the documentation to gh-pages if your repository is private.

The gh-pages documentation is refreshed every time there is a push to your default branch.

Note that only one copy of the documentation is served (the latest version).
For more mature projects, you may wish to host the documentation readthedocs_ instead, which supports hosting documentation for multiple package versions simultaneously.

.. _GitHub Pages: https://pages.github.com/
.. _readthedocs: https://readthedocs.org/

Building locally
^^^^^^^^^^^^^^^^

You can build the web documentation locally with::

   make -C docs html

And view the documentation like so::

   sensible-browser docs/_build/html/index.html

Or you can build pdf documentation::

   make -C docs latexpdf

On Windows, this becomes::

    cd docs
    make html
    make latexpdf
    cd ..

Other documentation features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Your README.rst will become part of the generated documentation (via a link file ``docs/source/readme.rst``).
  Note that the first line of README.rst is not included in the documentation, since this is expected to contain badges which you want to render on GitHub, but not include in your documentation pages.

- If you prefer, you can use a README.md file written in GitHub-Flavored Markdown instead of README.rst.
  This will automatically be handled and incorporate into the generated documentation (via a generated file ``docs/source/readme.rst``).
  As with a README.rst file, the first line of README.md is not included in the documentation, since this is expected to contain badges which you want to render on GitHub, but not include in your documentation pages.

- Your docstrings to your modules, functions, classes and methods will be used to build a set of API documentation using autodoc_.
  Our ``docs/conf.py`` is also set up to automatically call autodoc whenever it is run, and the output files which it generates are on the gitignore list.
  This means you will automatically generate a fresh API description which exactly matches your current docstrings every time you generate the documentation.

- Docstrings can be formatted in plain reST_, or using the `numpy format`_ (recommended), or `Google format`_.
  Support for numpy and Google formats is through the napoleon_ extension (which we have enabled by default).

- You can reference functions in the python core and common packages and they will automatically be hyperlinked to the appropriate documentation in your own documentation.
  This is done using intersphinx_ mappings, which you can see (and can add to) at the bottom of the ``docs/conf.py`` file.

- The documentation theme is sphinx-book-theme_.
  Alternative themes can be found at sphinx-themes.org_, sphinxthemes.com_, and writethedocs_.

.. _autodoc: http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
.. _Google format: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
.. _intersphinx: http://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
.. _numpy format: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy-style-python-docstrings
.. _Sphinx: https://www.sphinx-doc.org/
.. _sphinx-book-theme: https://sphinx-book-theme.readthedocs.io/
.. _sphinx-themes.org: https://sphinx-themes.org
.. _sphinxthemes.com: https://sphinxthemes.com/
.. _reST: http://docutils.sourceforge.net/rst.html
.. _writethedocs: https://www.writethedocs.org/guide/tools/sphinx-themes/


Consolidated metadata
~~~~~~~~~~~~~~~~~~~~~

Package metadata is consolidated into one place, the file ``template_experiment/__meta__.py``.
You only have to write the metadata once in this centralised location, and everything else (packaging, documentation, etc) picks it up from there.
This is similar to `single-sourcing the package version`_, but for all metadata.

This information is available to end-users with ``import template_experiment; print(template_experiment.__meta__)``.
The version information is also accessible at ``template_experiment.__version__``, as per PEP-396_.

.. _PEP-396: https://www.python.org/dev/peps/pep-0396/#specification
.. _single-sourcing the package version: https://packaging.python.org/guides/single-sourcing-package-version/


setup.py
~~~~~~~~

The ``setup.py`` script is used to build and install your package.

Your package can be installed from source with::

    pip install .

or alternatively with::

    python setup.py install

But do remember that as a developer, you should install your package in editable mode, using either::

    pip install --editable .

or::

    python setup.py develop

which will mean changes to the source will affect your installed package immediately without you having to reinstall it.

By default, when the package is installed only the main requirements, listed in ``requirements.txt`` will be installed with it.
Requirements listed in ``requirements-dev.txt``, ``requirements-docs.txt``, and ``requirements-test.txt`` are optional extras.
The ``setup.py`` script is configured to include these as extras named ``dev``, ``docs``, and ``test``.
They can be installed along with::

    pip install .[dev]

etc.
Any additional files named ``requirements-EXTRANAME.txt`` will also be collected automatically and made available with the corresponding name ``EXTRANAME``.
Another extra named ``all`` captures all of these optional dependencies.

Your README file is automatically included in the metadata when you use setup.py build wheels for PyPI.
The rest of the metadata comes from ``template_experiment/__meta__.py``.

Our template setup.py file is based on the `example from setuptools documentation <setuptools-setup.py_>`_, and the comprehensive example from `Kenneth Reitz <kennethreitz/setup.py_>`_ (released under `MIT License <https://github.com/kennethreitz/setup.py/blob/master/LICENSE>`__), with further features added.

.. _kennethreitz/setup.py: https://github.com/kennethreitz/setup.py
.. _setuptools-setup.py: https://setuptools.readthedocs.io/en/latest/setuptools.html#basic-use


GitHub Actions Workflows
~~~~~~~~~~~~~~~~~~~~~~~~

GitHub features the ability to run various workflows whenever code is pushed to the repo or a pull request is opened.
This is one service of several services that can be used to continually run the unit tests and ensure changes can be integrated together without issue.
It is also useful to ensure that style guides are adhered to

Two workflows are included:

docs
    The docs workflow ensures the documentation builds correctly, and presents any errors and warnings nicely as annotations.
    If your repository is public, publicly available html documentation is automatically deployed to the gh-pages branch and https://USER.github.io/PACKAGE/.

pre-commit
    Runs the pre-commit stack.
    Ensures all contributions are compliant, even if a contributor has not set up pre-commit on their local machine.

.. _Codecov: https://codecov.io/
.. _ci-packaging: https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
.. _github-secrets: https://docs.github.com/en/actions/reference/encrypted-secrets


Contributing
------------

Contributions are welcome! If you can see a way to improve this template:

- Clone this repo
- Create a feature branch
- Make your changes in the feature branch
- Push your branch and make a pull request

Or to report a bug or request something new, make an issue.


.. highlight:: python

.. |SLURM| image:: https://img.shields.io/badge/scheduler-SLURM-40B1EC
   :target: https://slurm.schedmd.com/
   :alt: SLURM
.. |preempt| image:: https://img.shields.io/badge/preemption-supported-brightgreen
   :alt: preemption
.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-DDP-EE4C2C?logo=pytorch&logoColor=EE4C2C
   :target: https://pytorch.org/
   :alt: pytorch
.. |wandb| image:: https://img.shields.io/badge/Weights_%26_Biases-enabled-FFCC33?logo=WeightsAndBiases&logoColor=FFCC33
   :target: https://wandb.ai
   :alt: Weights&Biases
.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black
