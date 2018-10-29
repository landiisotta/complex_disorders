0) DATA_FOLDER:~/data1/complex_disorders/data/autism/cohorts/2018-10-23-17-35-48

   Select: autism, alzheimer, multiple myeloma, parkinson with at least len(ehr)>=3;
   Key words: autism/autistic/pervasive developmental disorder; alzheimer; multiple myeloma.
   OTH: 2500
   No of Patients: 9162 (icd9 -- icd10 -- medication).
   Multiple myeloma = 2738
   Alzheimer = 2929
   ASD = 1044

   Tokens (Vocabulary mt_to_ix.csv):
   ASD = [2568, 7628, 7301, 2088, 6118, 5361, 7321, 8684, 11299]
   MM = [4019, 372, 2970, 3077, 8316]
   AD = [654, 11559]
   
   Tokens (Vocbulary TRIMMEDcohort-vocabulary)
   ASD=[804, 5214, 5331, 618, 3134, 2541, 4190, 5575, 9462], 
   MM=[2610, 1, 1150, 1617], 
   AD =[118, 9848]
   The architecture takes 18hrs to run.

   LSTM_baseline: parameters = {batch_size:16}
   
   Output folder: autism2018-10-24-17-29-14

1) Data diagnostics (patient numerosity and classes) can be found in data_diagnostics.ipynb in ../cohorts/2018-10-23-17-35-48

   After preprocessing: OTH (no overlapping): 1739 -- ASD: 938 -- MM: 2507 -- AD: 2828 (31 patients have overlapping diagnosis between ASD, MM, AD).
   Classes file: mrn_classes.csv [MRN, CLASS] with OTH=0, ASD=1, MM=2, AD=3.

2) run data_clustering_visualization.ipynb

3) Second option for NIPS --> trimming sequences to 200 tokens. If seq > 200 keep from first diagnosis to 200, otherwise keep 200 tokens (padding for seq < 200).
   Code in data_trimming.ipynb in ../cohorts/2018-10-23-17-35-48

In the case of LSTM (../bin_baseline)).


