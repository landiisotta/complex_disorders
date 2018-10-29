1) run create_ehr_cohorts.py --> cohort-ehr.csv, cohort-vocab.csv, cohort-person.csv.
2) run filter_collection_frequencies.ipynb --> stop-words.csv, collection-frequencies.csv.
3) run preprocessing_ehr.py to remove the stop clinical terms (high/low occurrencees), 
   order the sequences w.r.t age in days,
   divide the sequences in subsequences of unique terms T (start with T=15) days long,
   shuffle the terms in the subsequences.
4) with stats_ehr-length.ipynb check the length of the shuffled sequences and select the MRN
   with < of 3 medical terms, save the list into a file (list_mrnToDrop.csv).
5) run data_preparation.py to create an array with the ehr sequences ordered by length and a 
   new vocabulary (since we eliminated least occurrent terms). In padded_ehrs.csv --> array with padded sequences.
   In ordered_mrns.csv there is the list of mrns ordered w.r.t sequence length (decreasing order).
   In mt_to_ix.csv there is the new vocabulary.

6) set folder paths and model's input parameters in ../bin/utils.py 
7) run ../bin/main.py for training of the architecture. Save the best model (loss < 0.001 or end of epochs) and evaluate it, save mrns and TR + eval ecoded vectors
   (../experiments/disease+date_time/encoded_vect.csv, mrns.csv). [Remark: for the trimmed version run code data_trimming.ipynb in cohorts/date_time and then 
   move to bin_200 to run the architecture].
8) run data_clustering_visualization.ipynb 
