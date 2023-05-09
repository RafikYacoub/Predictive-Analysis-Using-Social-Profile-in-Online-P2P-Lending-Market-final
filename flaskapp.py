import flask; print(flask.__version__)
from flask import Flask, render_template, request
import os
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
import datetime
import joblib

X = []

today = datetime.datetime.now()

app = Flask(__name__)
app.debug = "development"
result = ""
print("I am in flask app")

@app.route('/', methods=['GET'])
def hello():
    print("I am In hello. Made some changes")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():


    print("Request.method:", request.method)
    print("Request.TYPE", type(request))
    print("In the process of making a prediction.")
    if request.method == 'POST':
        X = []

        clsmodel = joblib.load('classification_model.joblib')
        rgrmodel = joblib.load('regression_model.joblib')

        # clsmodel = pickle.load(open('clsmodel.pkl', 'rb'))
        # rgrmodel = pickle.load(open('rgrmodel.pkl', 'rb'))


        columns = ['CreditGrade', 'Term', 'BorrowerAPR', 'BorrowerRate',
            'LenderYield', 'ProsperScore', 'EmploymentStatus',
            'IsBorrowerHomeowner', 'CreditScore', 'CurrentCreditLines',
            'OpenCreditLines', 'TotalCreditLinespast7years',
            'OpenRevolvingAccounts', 'OpenRevolvingMonthlyPayment',
            'InquiriesLast6Months', 'TotalInquiries', 'RevolvingCreditBalance',
            'BankcardUtilization', 'AvailableBankcardCredit', 'TotalTrades',
            'TradesNeverDelinquent', 'DebtToIncomeRatio',
            'IncomeVerifiable', 'StatedMonthlyIncome', 'LoanMonthsSinceOrigination',
            'LoanOriginalAmount', 'MonthlyLoanPayment', 'LP_CustomerPayments',
            'LP_CustomerPrincipalPayments', 'LP_InterestandFees', 'LP_ServiceFees',
            'LP_CollectionFees', 'LP_GrossPrincipalLoss', 'LP_NetPrincipalLoss',
            'LP_NonPrincipalRecoverypayments', 'Investors', 'year', 'month',
            'YearsWithCredit', 'LowWage', 'MediumWage', 'HighWage']

        for column in columns:
            if column == 'StatedMonthlyIncome':
                    feature = request.form.get(column)
                    if  float(feature) < 2500 :
                        wage = 0
                    elif float(feature) < 8000 :
                        wage = 1
                    else:
                        wage = 2
            if column == 'LowWage':
                if wage == 0:
                    X.append(1)
                else:
                    X.append(0)
            elif column == 'MediumWage':
                if wage == 1:
                    X.append(1)
                else:
                    X.append(0)
            elif column == 'HighWage':
                if wage == 2:
                    X.append(1)
                else:
                    X.append(0)
            elif column == 'year':
                X.append(today.year)
            elif column == 'month':
                X.append(today.month)
            elif column == 'LoanMonthsSinceOrigination':
                X.append(0)
            else:
                value = request.form.get(column)
                print(value)
                X.append(float(value))
                
        #X = request.form
        print("X values = ",X)

        X = np.array(X)
        X = X.reshape(1, -1)
        test_arr = X



        yreg_pred = rgrmodel.predict(test_arr)
        ycls_pred = clsmodel.predict(test_arr)

        PROI = yreg_pred[:,0]
        ELA = yreg_pred[:,1]
        EMI = yreg_pred[:,2]

        FinancialRisk = ((0.05 * 0.5 * ELA) / (PROI * EMI * 12)) *100


        print("Cls Model Object: ", clsmodel)
        print("Rgr Model Object: ", rgrmodel)

        predicted = "Accepted" if ycls_pred else "Risky" 
        result = f"The model has predicted that the result is: {predicted} with PROI = {PROI}, ELA = {ELA}, EMI = {EMI} and Financial Risk = {FinancialRisk}%"
        return render_template('index.html', result=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)