from flask import Flask, render_template, request
import pickle
import numpy as np
department_mapping = {
    'sewing': 0,
    'finishing': 1,
    'quality': 2,
    'cutting': 3,
    'ironing': 4,
    'packing': 5
}

day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Saturday': 4,
    'Sunday': 5
}

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # Load your trained model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract values from form, convert to float, reshape to array
    quarter=request.form['quarter']
    department = department_mapping[request.form['department']]
    day = day_mapping[request.form['day']]
    team=request.form['team']
    targeted_productivity=request.form['targeted_productivity']
    smv=request.form['smv']
    over_time=request.form['over_time']
    incentive=request.form['incentive']
    idle_time=request.form['idle_time']
    idle_men=request.form['idle_men']
    no_of_style_change=request.form['no_of_style_change']
    no_of_workers=request.form['no_of_workers']
    month=request.form['month']


    total=[[int( quarter),int(department),int(day),int( team),float(targeted_productivity),float( smv),int(over_time),int( incentive),float(idle_time),int( idle_men),int(no_of_style_change),float( no_of_workers),int(month)]]
    print(total)
    
    prediction = model.predict(total)
    print(prediction)
    if prediction<=0.3:
        text="The Employee is Averagely Productive"
    elif prediction>0.3 and prediction<=0.8:
            text="The Employee is Mediumly Productive"
    else:
          text="The Employee is Highly Productive"       
    return render_template('result.html', prediction_text=text)

if __name__ == "__main__":
    app.run(debug=True)
