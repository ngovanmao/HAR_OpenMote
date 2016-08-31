This project is to use OpenMote-cc2538 platform with accelerometers for detecting human activity.

It includes the dataset that was collected by myself.
1. Dataset:
	User wears five sensors on the body (left/right wrist, left/right ankle, and one in the chest). 
	The sensor data is collected with sampling rate 64 Hz, and stored in database at the server for further processing.

2. Training:
	a. only using RH sensor: clfMysqlOpenmote_RH_DCT.py
	- output with classifer function to file: SVM_HAR_PA1.pkl (SVM model) and DCT_HAR_PA1.pkl (Decision tree model)
	- Even though using only one sensor, the accuracy is still nearly 90%.
	
	b. using full five sensors:
	- Using five sensors has some issue about synchronize time between all sensors and missing data in one of them. Some tricks have been included in the source code to remove/ignore/fill in them.


3. Prediction and visualization:
	a. Only using RH sensor: predictionMySqlOpenmote_RH_DCT.py
	- Get the trained model from file that given above, then read data from database (mySQL).
	- Prediction and show the result in image file.

	b. Using full 5 sensors:

