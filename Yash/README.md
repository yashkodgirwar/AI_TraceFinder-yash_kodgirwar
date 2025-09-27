📌 AI TraceFinder – Progress Report (Yash Kodgirwar)
🔹 Work Completed

✅ Dataset collected and preprocessed

✅ Baseline feature extraction with pixel intensity flattening

✅ Trained models:-Pixel Intensity Baseline (XGBoost, RF, SVM)

SVM – Accuracy: 79%

Random Forest – Accuracy: 89%

XGBoost – Accuracy: 97% (saved model + label encoder)

✅ Confusion matrices and classification reports generated

Handcrafted Features ( FFT + LBP + PRNU + correlation):

SVM → ~72%
Random Forest → ~71%
XGBoost  → ~74%

Hybrid CNN (Residual Images + Handcrafted Features):

Achieved ~74.7% test accuracy
Models saved (scanner_hybrid.keras, scanner_hybrid_final.keras)



🔹 Next Steps:-
Implement Hybrid CNN + handcrafted features.

Finalize comparison table:

Pixel Intensity Baseline (XGBoost, RF, SVM)

Handcrafted Features + ML

CNN / Hybrid CNN
