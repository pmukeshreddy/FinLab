***************
Getting Started
***************

Requirements
============

- Docker or equivalent (e.g. `colima <https://github.com/abiosoft/colima>`_
  or `podman <https://podman.io/>`_)
- `uv <https://docs.astral.sh/uv/>`_ package manager

Installation
============

From the project root, install the dependencies ::

    uv sync

Any secrets (e.g. LLM API keys) should be placed in a ``.env`` file in the
repository root.

The pipeline can then be run using uv with ::

    uv run task pipeline

See :ref:`configuration` for details on configuring the pipeline and
:ref:`running` for more details.
