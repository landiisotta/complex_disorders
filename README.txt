1) run create_ehr_cohorts.py --> cohort-ehr.csv, cohort-vocab.csv, cohort-person.csv, cohort-diseases.csv, cohort-mrns_icds.csv.
2) run filter_collection_frequencies.ipynb --> stop-words.csv, collection-frequencies.csv.
3) run preprocessing_ehr.py to remove the stop clinical terms (high/low occurrencees), 
   order the sequences w.r.t age in days,
   divide the sequences in subsequences of unique terms T (start with T=15) days long,
   shuffle the terms in the subsequences.
4) with stats_ehr-length.ipynb check the length of the shuffled sequences and select the MRN
   with < of 3 medical terms, save the list into a file (list_mrnToDrop.csv).
5) run data_preparation.py to create the new vocabulary (after dropping the stop words and the mrn with too short ehrs) --> 
   cohort-new_vocab.csv and the new ehr file with the new vocabulary (starts from 1) ---> cohort-new_ehr.csv. 
   Moreover a file with mrns and the diagnosed diseases is produced ---> cohort-mrn_diseases.csv, 
   the diseases are the ones specified in create_ehr_cohorts.csv  
6) set folder paths and model's input parameters in ../bin/utils.py REMARK: batch size should be set = 1 cause ehr sequences are trimmed in subsequences of length utils.L
   (last subsequence, if shorter that L is padded). 
7) run ../bin/main.py for training the architecture. Save the best model (loss < 0.001 or end of epochs) and evaluate it, save mrns and TR + eval ecoded vectors
   (../experiments/disease+date_time/encoded_vect.csv, mrns.csv).
8) run data_clustering_visualization.ipynb 
