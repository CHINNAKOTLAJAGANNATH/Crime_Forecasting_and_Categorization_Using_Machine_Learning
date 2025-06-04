import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
import pickle
app = Flask(__name__) #Initialize the flask App

decision = pickle.load( open('Decision.pkl', 'rb') )
Bagging = pickle.load( open('Bagging.pkl', 'rb') )
@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')
    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')




@app.route('/predict',methods=["POST"])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        Month = request.form['Month']
        Day = request.form['Day']
        dayOfWeek = request.form['dayOfWeek']
        crimeAgainst = request.form['crimeAgainst'] 
        near_place = request.form['near_place']
        latitude = request.form['latitude']
        longitude = request.form['longitude']
         
        
        
        Model = request.form['Model']
        
		# Clean the data by convert from unicode to float 
        
        sample_data = [Year,Month,Day,dayOfWeek,crimeAgainst,near_place,latitude,longitude]
        # clean_data = [float(i) for i in sample_data]
        # int_feature = [x for x in sample_data]
        int_feature = [float(i) for i in sample_data]
        print(int_feature)
    

		# Reshape the Data as a Sample not Individual Features
        
        ex1 = np.array(int_feature).reshape(1,-1)
        print(ex1)
		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)

        # Reloading the Model
        if Model == 'DecisionTreeClassifier':
           result_prediction = decision.predict(ex1)
           
            
        elif Model == 'BaggingClassifier':
          result_prediction = Bagging.predict(ex1)
    return render_template('prediction.html', prediction_text= result_prediction[0], model = Model)
@app.route('/chart')
def chart():
	return render_template('chart.html')  
 
if __name__ == "__main__":
    app.run(debug=True)
