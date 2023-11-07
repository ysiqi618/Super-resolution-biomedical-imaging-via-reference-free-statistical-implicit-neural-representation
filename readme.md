#### MLINR code #################
#### Siqi Ye, Rad. Onc. Stanford #######
#### 2023-11-05  ###################

This folder contains code of MLINR for super-resolution.


1. environment: create conda enviroment using the "nerp.yml" file

2. Data format: .npz

3. modify config files located in the "config" folder. We provide the configure file for the CCA-US data reported in our paper as an example. Users may modify (1) the information of images and (2) "ker_size" and "ker_type" used forward_op.py to model the image degradation from HR to LR, for user-specified applications.	

4. run the training code: locate in the sh_file folder, run in the command window (for linux os). For example:
 bash train_main_CCA_US.sh

5. Citation: Ye, S., Shen, L., Islam, M. T., & Xing, L. (2023). Super-resolution biomedical imaging via reference-free statistical implicit neural representation. Physics in Medicine & Biology, 68(20), 205020.
