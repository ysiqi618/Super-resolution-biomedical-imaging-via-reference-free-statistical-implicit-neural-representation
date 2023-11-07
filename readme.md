# Super-resolution biomedical imaging via reference-free statistical implicit neural representation
Manuscript: https://iopscience.iop.org/article/10.1088/1361-6560/acfdf1/pdf

# Instruction for running the code
1. Environment setup: create conda environment using the "mlinr_env.yml" file
2. Main code: "train_main_MLINR.py" is the main code to process a single image. "train_main_loopdataset.py" is the code to apply the MLINR method to the entire dataset and compute the statistics of evaluation metrics in the end.
3. Configure file: config files located in the "config" folder. We provide the configure file for the ultrasound data (CCA-US) reported in our paper as an example. Users may modify (1) the information of images and (2) "ker_size" and "ker_type" that model the HR-to-LR image degradation, based on user-specified applications.	
4. Dataset: we provide ultrasound images used in our paper as examples. Uses may put their own images into the "dataset" folder.
5. Script for running the code: located in the "sh_file" folder. For example, in the command line, type in "bash train_main_CCA_US.sh" to run the code for ultrasound image SR.

# Citation:
If you find the code useful, please consider citing the paper:

Ye, S., Shen, L., Islam, M. T., & Xing, L. (2023). Super-resolution biomedical imaging via reference-free statistical implicit neural representation. Physics in Medicine & Biology, 68(20), 205020.
