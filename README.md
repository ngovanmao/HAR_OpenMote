1. Dataset:

2. Training:
	a. only using RH sensor: clfMysqlOpenmote_RH_DCT.py
	- output with classifer function to file: SVM_HAR_PA1.pkl (SVM model) and DCT_HAR_PA1.pkl (Decision tree model)
	
	b. using full five sensors:

3. Prediction:
	a. Only using RH sensor: predictionMySqlOpenmote_RH_DCT.py
	- Get the trained model from file that given above, then read data from database (mySQL).
	- Prediction and show the result in image file.

	b. Using full 5 sensors:

