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

When creating a new repository from this template, these are the steps to follow:

#. *Don't click the fork button.*
   The fork button is for making a new template based in this one, not for using the template to make a new repository.

#. Create repository.

   #.  **New GitHub repository**.

       You can create a new repository on GitHub from this template by clicking the `Use this template <https://github.com/scottclowe/pytorch-experiment-template/generate>`_ button.

       Then clone your new repository to your local system [pseudocode]:

       .. code-block:: bash

          git clone git@github.com:your_org/your_repo_name.git
          cd your_repo_name

   #.  **New repository not on GitHub**.

       Alternatively, if your new repository is not going to be on GitHub, you can download `this repo as a zip <https://github.com/scottclowe/pytorch-experiment-template/archive/master.zip>`_ and work from there.

       Note that this zip does not include the .gitignore and .gitattributes files (because GitHub automatically omits them, which is usually helpful but is not for our purposes).
       Thus you will also need to download the `.gitignore <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.gitignore>`__ and `.gitattributes <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.gitattributes>`__ files.

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

#.  Rename the directory ``template_experiment`` to be the ``path`` variable you just added to ``__meta__.py``:

    .. code-block:: bash

      # Define PROJ_HYPH as your actual project name (use hyphens instead of underscores or spaces)
      PROJ_HYPH=your-actual-project-name-with-hyphens-for-spaces

      # Automatically convert hyphens to underscores to get the directory name
      PROJ_DIRN="${PROJ_HYPH//-/_}"
      # Rename the directory
      mv template_experiment "$PROJ_DIRN"

#.  Change references to ``template_experiment`` and ``template-experiment``
    to your path variable.

    This can be done with the sed command:

    .. code-block:: bash

        sed -i "s/template_experiment/$PROJ_DIRN/" "$PROJ_DIRN"/*.py setup.py slurm/*.slrm
        sed -i "s/template-experiment/$PROJ_HYPH/" "$PROJ_DIRN"/*.py setup.py slurm/*.slrm

    Which will make changes in the following places.

    - In ``setup.py``, `L51 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/setup.py#L51>`__::

        exec(read("template_experiment/__meta__.py"), meta)

    - In ``__meta__.py``, `L2,4 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/__meta__.py#L2-4>`__::

        name = "template-experiment"

    - In ``train.py``, `L17-18 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/train.py#L17-18>`__::

        from template_experiment import data_transformations, datasets, encoders, utils
        from template_experiment.evaluation import evaluate

    - In ``train.py``, `L1321 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/template_experiment/train.py#L1321>`__::

        group.add_argument(
            "--wandb-project",
            type=str,
            default="template-experiment",
            help="Name of project on wandb, where these runs will be saved.",
        )

    - In ``slurm/train.slrm``, `L19 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/train.slrm#L19>`__::

        #SBATCH --job-name=template-experiment    # Set this to be a shorthand for your project's name.

    - In ``slurm/train.slrm``, `L23 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/train.slrm#L23>`__::

        PROJECT_NAME="template-experiment"

    - In ``slurm/notebook.slrm``, `L16 <https://github.com/scottclowe/pytorch-experiment-template/blob/master/slurm/notebook.slrm#L16>`__::

        PROJECT_NAME="template-experiment"

#.  Swap out the contents of ``README.rst`` with an initial description of your project.
    If you prefer, you can use markdown (``README.md``) instead of rST:

    .. code-block:: bash

      git rm README.rst
      # touch README.rst
      touch README.md && sed -i "s/.rst/.md/" MANIFEST.in

#.  Add your changes to the repo's initial commit and force-push your changes:

    .. code-block:: bash

      git add .
      git commit --amend
      git push --force

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
(you don't need to re-run this every time you start a new project):

.. code-block:: bash

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
`create a Weights and Biases account <wandb-signup_>`_ if you don't already have one:

.. code-block:: bash

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
new virtual environment to use for the project:

.. code-block:: bash

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

Run this code block when you want to resume work on an existing project:

.. code-block:: bash

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

#. If you haven't already, then follow the `Vector one-time set-up`_
   instructions.

#. Then clone the repository:

   .. code-block:: bash

        git clone git@github.com:scottclowe/pytorch-experiment-template.git
        cd pytorch-experiment-template

#. Run the `Project one-time set-up`_ (using ``template-experiment`` as
   the environment name).

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

  or alternatively::

        template-experiment-train

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

  or increasing the number of nodes being requested::

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

        squeue --me

   Note the job id of the notebook job. e.g.:

   .. code-block:: none

        (template-experiment) slowe@v2:~/pytorch-experiment-template$ squeue --me
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          10618891 interacti      jnb    slowe  R       1:07      1 gpu026

   Here we can see our JOBID is 10618891, and it is running on node gpu026.

#. Inspect the output of the job with::

        cat jnb_JOBID.out

   e.g.::

        cat jnb_10618891.out

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

   You need to replace USERNAME with your Vector username, gpu026 with the node
   your job is running on, and 47201 with the port number from the previous
   step.
   In this example, the local port which the notebook is being forwarded to is
   port 8887.

#. Open a browser on your local machine and navigate to http://localhost:8887
   (or whatever port you chose in the previous step)::

        sensible-browser http://localhost:8887

   You should see the Jupyter notebook interface.
   Copy the token from the URL shown in the log file and paste it into the
   ``Password or token: [ ] Log in`` box.
   You should now have access to the remote notebook server on your local
   machine.

#. Once you are done working in your notebooks (and have saved your changes),
   make sure to end the job running the notebook with::

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
``/checkpoint/USERNAME/PROJECT__JOBNAME__JOBID`` (with double-underscores
between each category) along with a record of the conda environment and
frozen pip requirements used to run the job in ``environment.yml`` and
``frozen-requirements.txt``.


Log messages
~~~~~~~~~~~~

Any print statements and error messages from the training script will be saved
to the file ``slogs/JOBNAME__JOBID_ARRAYID.out``.
Only the output from the rank 0 worker (the worker which saves the
checkpoints and sends logs to wandb) will be saved to this file.
When using multiple nodes, the output from each node will be saved to a
separate file: ``slogs-inner/JOBNAME__JOBID_ARRAYID-NODERANK.out``.

You can monitor the progress of a job that is currently running by monitoring
the contents of its log file. For example:

.. code-block:: bash

    tail -n 50 -f slogs/JOBNAME__JOBID_ARRAYID.out


Weights and Biases
~~~~~~~~~~~~~~~~~~

`Weights and Biases`_ (wandb) is an online service for tracking your
experiments which is free for academic usage.

This template repository is set up to automatically log your experiments, using
the same job label across both SLURM and wandb.

If the job is preempted, the wandb logging will resume to the same wandb job
ID instead of spawning a new one.


Random Number Generator (RNG) state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All RNG states are configured based on the overall seed that is set with the
``--seed`` argument to ``train.py``.

When running ``train.py`` directly, the seed is **not** set by default, so
behaviour will not be reproducible.
You will need to include the argument ``--seed=0`` (for example), to make sure
your experiments are reproducible.

When running on SLURM with slurm/train.slrm, the seed **is** set by default.
The seed used is equal the `array ID <slurm-job-array_>`_ of the job.
This configuration lets you easily run the same job with multiple seeds in one
sbatch command.
Our default job array in ``slurm/train.slrm`` is ``--array=0``, so only one job
will be launched, and that job will use the default seed of ``0``.

To launch the same job 5 times, each with a different seed (1, 2, 3, 4, and 5)::

    sbatch --array=1-5 slurm/train.slrm

or to use seeds 42 and 888::

    sbatch --array=42,888 slurm/train.slrm

or to use a randomly selected seed::

    sbatch --array="$RANDOM" slurm/train.slrm

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
from the test set. You should use this (with flag ``--prototyping``) during the
initial model development steps and for any hyperparameter searches.

Your final models should be trained without ``--prototyping`` enabled, so that
the full training set is used for training and the best model is produced.


Optional extra package dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several requirements files in the root directory of the repository.
The idea is the requirements.txt file contains the minimal set of packages
that are needed to use the models in the package.
The other requirements files are for optional extra packages.

requirements-dev.txt
    Extra packages needed for code development (i.e. writing the codebase)

requirements-notebook.txt
    Extra packages needed for running the notebooks.

requirements-train.txt
    Extra packages needed for training the models.

The setup.py file will automatically parse any requirements files in the
root directory of the repository which are named like ``requirements-*.txt``
and make them available to ``pip`` as extras.

For example, to install the repository to your virtual environment with the
extra packages needed for training::

    pip install --editable .[train]

You can also install all the extras at once::

    pip install --editable .[all]

Or you can install the extras directly from the requirements files::

    pip install -r requirements-train.txt

As a developer of the repository, you will need to pip install the package
with the ``--editable`` flag so the installed copy is updated automatically
when you make changes to the codebase.


Automated code checking and formatting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The template repository comes with a pre-commit_ stack.
This is a set of git hooks which are executed every time you make a commit.
The hooks catch errors as they occur, and will automatically fix some of these errors.

To set up the pre-commit hooks, run the following code from within the repo directory::

    pip install -r requirements-dev.txt
    pre-commit install

Whenever you try to commit code which is flagged by the pre-commit hooks,
*the commit will not go through*. Some of the pre-commit hooks
(such as black_, isort_) will automatically modify your code to fix formatting
issues. When this happens, you'll have to stage the changes made by the commit
hooks and then try your commit again. Other pre-commit hooks, such as flake8_,
will not modify your code and will just tell you about issues in what you tried
to commit (e.g. a variable was declared and never used), and you'll then have
to manually fix these yourself before staging the corrected version.

After installing it, the pre-commit stack will run every time you try to make
a commit to this repository on that machine.
You can also manually run the pre-commit stack on all the files at any time::

    pre-commit run --all-files

To force a commit to go through without passing the pre-commit hooks use the ``--no-verify`` flag::

    git commit --no-verify

The pre-commit stack which comes with the template is highly opinionated, and
includes the following operations:

- All **outputs in Jupyter notebooks are cleared** using nbstripout_.

- Code is reformatted to use the black_ style.
  Any code inside docstrings will be formatted to black using blackendocs_.
  All code cells in Jupyter notebooks are also formatted to black using black_nbconvert_.

- Imports are automatically sorted using isort_.

- Entries in requirements.txt files are automatically sorted alphabetically.

- Several `hooks from pre-commit <pre-commit-hooks_>`_ are used to screen for
  non-language specific git issues, such as incomplete git merges, overly large
  files being commited to the repo, bugged JSON and YAML files.

- JSON files are also prettified automatically to have standardised indentation.

The pre-commit stack will also run on github with one of the action workflows,
which ensures the code that is pushed is validated without relying on every
contributor installing pre-commit locally.

This development practice of using pre-commit_, and standardizing the
code-style using black_, is popular among leading open-source python projects
including numpy, scipy, sklearn, Pillow, and many others.

If you want to use pre-commit, but **want to commit outputs in Jupyter notebooks**
instead of stripping them, simply remove the nbstripout_ hook from the
`.pre-commit-config.yaml file <https://github.com/scottclowe/pytorch-experiment-template/blob/master/.pre-commit-config.yaml#L31-L35>`__
and commit that change.

If you don't want to use pre-commit at all, you can uninstall it::

    pre-commit uninstall

and purge it (along with black and flake8) from the repository::

    git rm .pre-commit-config.yaml .flake8 .github/workflows/pre-commit.yaml
    git commit -m "DEV: Remove pre-commit hooks"

.. _black: https://github.com/psf/black
.. _black_nbconvert: https://github.com/dfm/black_nbconvert
.. _blackendocs: https://github.com/asottile/blacken-docs
.. _flake8: https://gitlab.com/pycqa/flake8
.. _isort: https://github.com/timothycrosley/isort
.. _nbstripout: https://github.com/kynan/nbstripout
.. _pre-commit: https://pre-commit.com/
.. _pre-commit-hooks: https://github.com/pre-commit/pre-commit-hooks
.. _pre-commit-py-hooks: https://github.com/pre-commit/pygrep-hooks


Additional features
-------------------

This template was forked from a more general `python template repository`_.

For more information on the features of the python template repository, see
`here <python-template-repository-features_>`_.

.. _`python template repository`: https://github.com/scottclowe/python-template-repo
.. _`python-template-repository-features`: https://github.com/scottclowe/python-template-repo#features


Contributing
------------

Contributions are welcome! If you can see a way to improve this template:

- Clone this repo
- Create a feature branch
- Make your changes in the feature branch
- Push your branch and make a pull request

Or to report a bug or request something new, make an issue.


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
