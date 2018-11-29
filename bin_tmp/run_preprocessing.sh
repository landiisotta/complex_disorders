#!/bin/bash
echo '1'
#python create_ehr_cohorts.py  && 
python filter_collection_frequencies.py && python preprocessing_ehr.py 
#python stats_ehr_lengths.py
#python data_preparation.py
