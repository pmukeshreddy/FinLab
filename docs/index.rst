#####################
Finance Task Pipeline
#####################

**Agentic Financial Task Creation Pipeline**

The pipeline automates the extraction of
machine learning tasks from papers, generating tasks in the
`Harbor <https://www.harborframework.com/docs>`_ format for use
in Agentic benchmarking, and SFT and RL LLM training.

I built the RL training and evaluation integration and completed GRPO training
for **Qwen3.8-27B on cloud GPUs**, improving measured task success from **61% to
75%**: **14 percentage points, approximately 23% relative improvement**.
See :doc:`rl` for the architecture and workflow, and the
:download:`technical report <rl-results.md>` for results and verification scope.

.. toctree::
   :caption: Contents:
   :maxdepth: 2
   :includehidden:

   getting_started
   pipeline
   configuration
   architecture
   development
   rl
