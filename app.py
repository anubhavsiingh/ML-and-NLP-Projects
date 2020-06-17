from flask import Flask,render_template,url_for,request,redirect,flash
from flask_sqlalchemy import SQLAlchemy
from datetime import date
from sentiment import prediction
import pickle
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.secret_key = '1602'

model = pickle.load(open('model.pkl', 'rb'))

db=SQLAlchemy(app)

class User(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	nam = db.Column(db.Text(20),nullable=False,unique=False)
	ratin = db.Column(db.Integer,unique=False)
	tex = db.Column(db.Text(200),nullable=False,unique=False)

	def __init__(self,nam,ratin,tex):
		self.nam = nam
		self.ratin = ratin
		self.tex = tex

	def __repr__(self):
		return f"user('{self.nam}','{self.ratin}','{self.tex}')"

@app.route("/",methods=['GET', 'POST'])
def home():
	data = User.query.all()
	return render_template("new.html",data=data)

@app.route("/redirect", methods=['GET', 'POST'])
def hom():
	data = User.query.all()
	def cal(data):
		n = len(data)
		return data[n-1],data[n-2],data[n-3]
	x,y,z = cal(data)
	return render_template('new1.html', data= data, x=x,y=y,z=z)

@app.route('/review', methods=['GET', 'POST'])
def user():
	db.create_all()
	if request.method == 'POST':
		data = User(nam = request.form['name'],ratin= request.form['rating'],tex= request.form['review'])
		if(len(request.form['name'])!=0 and len(request.form['review'])!=0):
			db.session.add(data)
			db.session.commit()
			a = prediction(request.form['review'])
			true = model.predict(a)
			if true:
				flash("Your form has been submitted. We appreciate your support.")
			else:
				flash("Your form has been submitted. We will do better.")
			
			return redirect(url_for('hom'))
		else:
			flash("Either you haven't entered the name properly or review.")
	return redirect(url_for('home'))

if __name__=="__main__":
	app.run(debug=True)