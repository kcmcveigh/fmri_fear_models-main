# fmri_fear_models

This repository contains code related to the analysis of dynamic and static models for predicting fear experiences using fMRI data. The code implements a set of analyses comparing the predictive performance of dynamic models (LSTMs) to static classical ML models (SVR). Including lesion analyses to compare how the different networks of the brain contribute to prediction.


## Code Files

- `train_model.py`: Trains the LSTM models.
- `train_svm.py`: Trains the sklearn implementations of RBF SVR and linear SVR.
- `generate_lesion_stats_lstm.py`: Generates stats with each of the 7 canonical networks in the brain lesion as well as with no networks lesion.
- `generate_lesion_stats_svm.py`: Generates stats with each of the 7 canonical networks in the brain lesion as well as with no networks lesion.
- `helper.py`: Provides the helper functions for all other scripts

## Results

Models are saved in saved_models and results from generate_lesion_stats_*.py are saved in results

Our findings show that dynamic and static models perform similarly in predicting fear experiences. This suggests that, in the context of this study (where the input data was 20 seconds of fMRI data and fear ratings were provided once at the end of the video), incorporating dynamic information through LSTM models does not significantly improve prediction accuracy.



| Model      |   Vis |   SomMot |   DorsAttn |   SalVent |   Limbic |   Cont |   Default |   None |
|:-----------|------:|---------:|-----------:|----------:|---------:|-------:|----------:|-------:|
| LSTM       |  0.42 |     0.45 |       0.4  |      0.45 |     0.49 |   0.49 |      0.5  |   0.5  |
| SVR Linear |  0.36 |     0.33 |       0.3  |      0.33 |     0.44 |   0.42 |      0.45 |   0.44 |
| SVR RBF    |  0.4  |     0.44 |       0.34 |      0.45 |     0.5  |   0.49 |      0.49 |   0.5  |





## Data Availability

Please note that the data used in this analysis is not yet publicly available. It will be made available once the first set of papers on the data is published.



## Contact

For any questions or inquiries, please contact [Yiyu Wang](mailto:yiyu.wang6@gmail.com) or [Kieran McVeigh](kieran.mcveighc@gmail.com).
