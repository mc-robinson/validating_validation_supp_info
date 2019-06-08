# Obtaining the Mayr et al. data and relevant preprocessing #

This directory contains the necessary steps to obtain and process the data for our analysis of the original article by Mayr and coworkers,
[Large-scale comparison of machine learning methods for drug target prediction on ChEMBL](https://pubs.rsc.org/en/content/articlelanding/2018/sc/c8sc00148k#!divAbstract).

**BibTeX:**

>@Article{bib:Mayr2018,\
>&nbsp;&nbsp;author="Mayr, Andreas and Klambauer, G{\\"u}nter and Unterthiner, Thomas and Steijaert, Marvin and Wegner, J{\\"o}rg K. and Ceulemans, Hugo and Clevert, Djork-Arn{\\'e} and Hochreiter, Sepp",\
>&nbsp;&nbsp;title={{Large-scale comparison of machine learning methods for drug target prediction on ChEMBL}},\
>&nbsp;&nbsp;journal="Chem. Sci.",\
>&nbsp;&nbsp;year="2018",\
>&nbsp;&nbsp;volume="9",\
>&nbsp;&nbsp;issue="24",\
>&nbsp;&nbsp;pages="5441-5451"\
>} 

Additional information on their code and dataset creation may be found at http://ml.jku.at/research/lsc/index.html or on their GitHub page https://github.com/ml-jku/lsc . 

Most of their their data can be downloaded at http://ml.jku.at/research/lsc/mydata.html 
However, for our analysis, we only used the data in the `dataPythonReduced` section. 
This data is optimized for those using Python and includes only the data that is directly 
necessary for running their simple feed-forward neural network algorithm. 
This also reduces the memory requirements.

We recommend directly downloading the zip file in the browser: https://ml.jku.at/research/lsc/chembl20/dataPythonReduced.zip
The file https://ml.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Deepchem.pckl will also need to be downoaded.

Once unzipped, this data is used to construct fingerprints for further analysis/testing of algorithms. Creating this data involves the running of two separate programs: `initial_processing.py` and `make_sparse_fps.py`. 

The main output of interest from these programs is included in the `data_for_replication` folder one level outside this directory. The whole process is summarized as follows:
1. Download and unzip https://ml.jku.at/research/lsc/chembl20/dataPythonReduced.zip , thus creating a `dataPythonReduced` folder inside of this directory.
2. Download https://ml.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Deepchem.pckl and place the file inside the recently created `./dataPythonReduced/` directory.
3. Run the Python script `intial_processing.py`
4. Run the Python script `make_sparse_fps.py` 

Please note that all of the work by Mayr and coworkers is licensed under the GNU General Public License v3.0 as detailed at http://www.bioinf.jku.at/research/lsc/LICENSE and https://github.com/ml-jku/lsc/blob/master/LICENSE .
We should note that most of the code in this directory, especially, is highly derivative of their work and includes code directly copied from their scripts as specified in the comments of our code.
All mistakes in the programs included herein are, of course, our own fault.


